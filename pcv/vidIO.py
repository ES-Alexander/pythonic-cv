#!/usr/bin/env python3

import cv2
import signal
import numpy as np
from time import perf_counter, sleep
from queue import Queue
from threading import Thread, Event
from pcv.interact import DoNothing, waitKey


class BlockingVideoWriter(cv2.VideoWriter):
    ''' A cv2.VideoWriter with a context manager for releasing.

    Generally suggested to use the non-blocking, threaded VideoWriter class
        instead, unless your application requires no wait time on completion
        but permits performance reduction throughout to write frames. If that's
        the case, try VideoWriter anyway, and come back to this if a notable
        backlog occurs (it will tell you).

    '''
    properties = {
        'quality'    : cv2.VIDEOWRITER_PROP_QUALITY,
        'framebytes' : cv2.VIDEOWRITER_PROP_FRAMEBYTES,
        'nstripes'   : cv2.VIDEOWRITER_PROP_NSTRIPES,
    }

    # functioning combinations are often hard to find - these hopefully work
    SUGGESTED_CODECS = {
        'avi' : ['H264','X264','XVID','MJPG'],
        'mp4' : ['avc1','mp4v'],
        'mov' : ['avc1','mp4v'],
        'mkv' : ['H264'],
    }

    def __init__(self, filename, fourcc, fps, frameSize, isColor=True,
                 apiPreference=None):
        ''' Initialise a BlockingVideoWriter with the given parameters.

        'filename' The video file to write to.
        'fourcc' the "four character code" representing the writing codec.
            Can be a four character string, or an int as returned by
            cv2.VideoWriter_fourcc. As functioning combinations of file
            extension + codec can be difficult to find, the helper method
            VideoWriter.suggested_codec is provided, accepting a filename
            (or file extension) and a list of previously tried codecs that
            didn't work and should be excluded. Suggested codecs are populated
            from VideoWriter.SUGGESTED_CODECS, if you wish to view the
            suggested options directly.
        'fps' is the framerate (frames per second) to save as. It is a constant
            float, and can only be set on initialisation. To have a video that
            plays back faster than the recording stream, set the framerate to
            higher than the input framerate. The VideoWriter.from_camera
            factory function is provided to create a video-writer directly
            from a camera instance, and allows measuring of the input framerate
            for accurate output results if desired.
        'frameSize' is the size of the input frames as a tuple of (rows, cols).
        'isColor' is a boolean specifying if the saved images are coloured.
            Defaults to True. Set False for greyscale input streams.
        'apiPreference' allows specifying which API backend to use. It can be
            used to enforce a specific reader implementation if multiple are
            available (e.g. cv2.CAP_FFMPEG or cv2.CAP_GSTREAMER). Generally
            this is not required, and if left as None it is ignored.

        '''
        self.filename       = filename
        self.fps            = fps
        self.is_color       = isColor
        self.frame_size     = frameSize
        self.api_preference = apiPreference
        self.set_fourcc(fourcc)

        super().__init__(*self._construct_open_args())

    def set_fourcc(self, fourcc):
        ''' Set fourcc code as an integer or an iterable of 4 chars. '''
        self.fourcc = fourcc # save for checking value
        if not isinstance(fourcc, int):
            # assume iterable of 4 chars
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._fourcc = fourcc

    def _construct_open_args(self):
        args = [self.filename, self._fourcc, self.fps, self.frame_size,
                self.is_color]
        if self.api_preference is not None:
            args = [args[0], self.api_preference, *args[1:]]
        return args
       
    def __enter__(self):
        ''' Re-entrant '''
        if not self.isOpened():
            self.open(*self._construct_open_args())
        return self

    def __exit__(self, *args):
        self.release()

    def get(self, property):
        ''' Returns 'property' value, or 0 if not supported by the backend.

        'property' can be a string key for the VideoWriter.properties
            dictionary or an integer from cv2.VIDEOWRITER_PROP_*

        self.get(str/int) -> float

        '''
        try:
            return super().get(self.properties[property.lower()])
        except AttributeError:
            return super().get(property)

    def set(self, property, value):
        ''' Attempts to set the specified property value.
        Returns True if the property is supported by the backend in use.

        'property' can be a string key for the VideoWriter.properties
            dictionary or an integer from cv2.VIDEOWRITER_PROP_*
        'value' should be a float

        self.set(str/int, float) -> bool

        '''
        try:
            return super().set(self.properties[property.lower()], value)
        except AttributeError:
            return super().set(property, value)

    @classmethod
    def suggested_codec(cls, filename, exclude=[]):
        extension = filename.split('.')[-1]
        try:
            return [codec for codec in cls.SUGGESTED_CODECS[extension.lower()]
                    if codec not in exclude][0]
        except IndexError:
            raise Exception('No codecs available, try a different extension')

    @classmethod
    def from_camera(cls, filename, camera, fourcc=None, isColor=True,
                    apiPreference=None, fps=-3):
        ''' Returns a VideoWriter based on the properties of the input camera.

        'filename' is the name of the file to save to.
        'camera' is the SlowCamera instance (or any of its subclasses).
        'fourcc' is the codec four-character code. If left as None is
            determined automatically from filename.
        'isColor' specifies if the video stream is colour or greyscale.
        'fps' can be set as a float, 'camera' to ask the camera for the value,
            or a negative integer to measure over that number of frames.
            If no processing is occurring, 'camera' is suggested, otherwise
            it is generally best to measure the frame output.
            Defaults to -3, to measure over 3 frames.

        '''
        if fourcc is None:
            fourcc = cls.suggested_codec(filename)
        frameSize = tuple(int(camera.get(dim)) for dim in ('width','height'))

        if fps == 'camera':
            fps = camera.get('fps')
        elif fps < 0:
            fps = camera.measure_framerate(-fps)
            
        return cls(filename, fourcc, fps, frameSize, isColor, apiPreference)

    def __repr__(self):
        return (f'{self.__class__.__name__}(filename={repr(self.filename)}, '
                f'fourcc={repr(self.fourcc)}, fps={self.fps}, '
                f'frameSize={self.frame_size}, isColor={self.is_colour}, '
                f'apiPreference={self.api_preference})')


class VideoWriter(BlockingVideoWriter):
    ''' A non-blocking thread-based video writer, using a queue. '''
    def __init__(self, *args, maxsize=0, verbose_exit=True, **kwargs):
        ''' Initialise the video writer.

        'maxsize' is the maximum allowed frame buildup before adding frames
            blocks execution. Defaults to 0 (no maximum). Set a meaningful
            number if you have fast processing, limited memory, and can't
            afford the time required to wait at the end once you've finished
            recording. Setting a number for this is helpful in early testing
            to get notified of cases where writing to disk is a bottleneck
            (you may get processing freezes from time to time as a result).
            Consistently slow write times may indicate a need for a more
            efficient file format, memory type, or just lower resolution in
            time or space (ie fewer fps or smaller images).
        'verbose_exit' is a boolean indicating if the writer should notify
            you on exit if a backlog wait will be required, and if so once it
            completes and how long it took. Defaults to True.

        *args and **kwargs are the same as those for BlockingVideoWriter.

        '''
        super().__init__(*args, **kwargs)
        self._initialise_writer(maxsize)
        self._verbose_exit = verbose_exit

    def _initialise_writer(self, maxsize):
        ''' Start the Thread for grabbing images. '''
        self.max_queue_size = maxsize
        self._write_queue = Queue(maxsize=maxsize)
        self._image_writer = Thread(name='writer', target=self._writer,
                                    daemon=True)
        self._image_writer.start()

    def _writer(self):
        ''' Write frames forever, until '''
        while "not finished":
            # retrieve an image, wait indefinitely if necessary
            img = self._write_queue.get()
            # write the image to file ('where' is specified outside)
            super().write(img)
            # inform the queue that a frame has been written
            self._write_queue.task_done()

    def write(self, img):
        ''' Send 'img' to the write queue. '''
        self._write_queue.put(img)

    def __exit__(self, *args):
        ''' Wait for writing to complete, and release writer. '''
        # assume not waiting
        waited = False

        # check if waiting required
        if self._verbose_exit and not self._write_queue.empty():
            print(f'Writing {self._write_queue.qsize()} remaining frames.')
            print('Force quitting may result in a corrupted video file.')
            waited = perf_counter()

        # finish writing all frames
        self._write_queue.join()

        # cleanup as normal
        super().__exit__(*args)

        # if wait occurred, inform of completion
        if waited and self._verbose_exit:
            print(f'Writing complete in {perf_counter()-waited:.3f}s.')


class OutOfFrames(StopIteration):
    def __init__(msg='Out of video frames', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


class UserQuit(StopIteration):
    def __init__(msg='User quit manually', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


class ContextualVideoCapture(cv2.VideoCapture):
    ''' A cv2.VideoCapture with a context manager for releasing. '''
    properties = {
        'fps'     : cv2.CAP_PROP_FPS,
        'mode'    : cv2.CAP_PROP_MODE,
        'width'   : cv2.CAP_PROP_FRAME_WIDTH,
        'height'  : cv2.CAP_PROP_FRAME_HEIGHT,
        'backend' : cv2.CAP_PROP_BACKEND,
    }
    # more properties + descriptions can be found in the docs:
    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d

    def __init__(self, id, *args, display='frame', delay=None, quit=ord('q'),
                 play_pause=ord(' '), pause_effects={}, play_commands={},
                 destroy=-1, **kwargs):
        ''' A pausable, quitable, iterable video-capture object
            with context management.

        'id' is the id that gets passed to the underlying VideoCapture object.
            it can be an integer to select a connected camera, or a filename
            to open a video.
        'display' is used as the default window name when streaming. Defaults
            to 'frame'.
        'delay' is the integer millisecond delay applied between each iteration
            to enable windows to update. If set to None, this is skipped and
            the user must manually call waitKey to update windows.
            Default is None, which allows headless operation without
            unnecessary waiting.
        'quit' is an integer ordinal corresponding to a key which can be used
            to stop the iteration loop. Only applies if delay is not None.
            Default is ord('q'), so press the 'q' key to quit when iterating.
        'play_pause' is an integer ordinal corresponding to a key which can be
            used to pause and resume the iteration loop. Only applies if delay
            is not None. Default is ord(' '), so press space-bar to pause/
            resume when iterating.
        'pause_effects' is a dictionary of key ordinals and corresponding
            handler functions. The handler will be passed self as its only
            argument, which gives it access to the 'get' and 'set' methods,
            as well as the 'status' and 'image' properties from the last 'read'
            call. This can be useful for logging, selecting images for
            labelling, or temporary manual control of the event/read loop.
            Note that this is only used while paused, and does not get passed
            quit or play_pause key events.
        'play_commands' is the same as 'pause_effects' but operates instead
            while playback/streaming is occurring. For live processing,
            this can be used to change playback modes, or more generally for
            similar scenarios as 'pause_effects'.
        'destroy' destroys any specified windows on context exit. Can be 'all'
            to destroy all active opencv windows, a string of a specific window
            name, or a list of window names to close. If left as -1, destroys
            the window specified in 'display'.

        '''
        super().__init__(id, *args, **kwargs)
        self._id             = id
        self.display         = display
        self._delay          = delay
        self._quit           = quit
        self._play_pause     = play_pause
        self._pause_effects  = pause_effects
        self._play_commands  = play_commands
        self._destroy        = destroy

        self._api_preference = kwargs.get('apiPreference', None)

    def __enter__(self, force=False):
        ''' Enter a re-entrant context for this camera. '''
        if force or not self.isOpened():
            if self._api_preference:
                self.open(self._id, self._api_preference)
            else:
                self.open(self._id)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        ''' Clean up on context exit.

        Releases the internal VideoCapture object, and destroys any windows
            specified at initialisation.

        '''
        # release VideoCapture object
        self.release()

        # clean up window(s) if specified on initialisation
        destroy = self._destroy
        try:
            if destroy == -1:
                cv2.destroyWindow(self.display)
            elif destroy == 'all':
                cv2.destroyAllWindows()
            elif isinstance(destroy, str):
                # a single window name
                cv2.destroyWindow(destroy)
            elif destroy is not None:
                # assume an iterable of multiple windows
                for window in destroy: cv2.destroyWindow(window)
            else:
                return # destroy is None
        except cv2.error as e:
            print('Failed to destroy window(s)', e)

        waitKey(3) # allow the GUI manager to update

    def __iter__(self):
        return self

    def __next__(self):
        # check if doing automatic waits
        if self._delay is not None:
            key = waitKey(self._delay)

            if key == self._quit:
                raise UserQuit
            elif key == self._play_pause:
                self._handle_pause()
            else:
                # pass self to a triggered user-defined key handler, or nothing
                self._play_commands.get(key, lambda cap: None)(self)

        # wait completed, get next frame if possible
        if self.isOpened():
            return self.read()
        raise OutOfFrames

    def _handle_pause(self):
        ''' Handle event loop and key-presses while paused. '''
        while "paused":
            key = waitKey(1)
            if key == self._quit:
                raise UserQuit
            if key == self._play_pause:
                break
            # pass self to a triggered user-defined key handler, or do nothing
            self._pause_effects.get(key, lambda cap: None)(self)

    def stream(self, mouse_handler=DoNothing()):
        ''' Capture and display stream on window specified at initialisation.

        'mouse_handler' is an optional MouseCallback instance determining
            the effects of mouse clicks and moves during the stream. Defaults
            to DoNothing.

        '''
        with mouse_handler:
            for read_success, frame in self:
                if read_success:
                    cv2.imshow(self.display, frame)
                else:
                    break # camera disconnected

    def headless_stream(self):
        ''' Capture and process stream without display. '''
        for read_success, frame in self:
            if not read_success: break # camera disconnected

    def record_stream(self, filename, show=True, mouse_handler=DoNothing()):
        ''' Capture and record stream, with optional display.

        'filename' is the file to save to.
        'show' is a boolean specifying if the result is displayed (on the
            window specified at initialisation).
        'mouse_handler' is an optional MouseCallback instance determining
            the effects of mouse clicks and moves during the stream. It is only
            useful if 'show' is set to True. Defaults to DoNothing.

        '''
        with VideoWriter.from_camera(filename, self) as writer, mouse_handler:
            for read_success, frame in self:
                if read_success:
                    if show:
                        cv2.imshow(self.display, frame)
                    writer.write(frame)
                else:
                    break # camera disconnected

    def get(self, property):
        ''' Return the value of 'property' if it exists, else 0.0. '''
        try:
            return super().get(self.properties.get(property, property))
        except TypeError: # property must be an unknown string
            return super().get(eval('cv2.CAP_PROP_'+property.upper()))

    def set(self, property, value):
        ''' Attempts to set 'property' to 'value', returning success. '''
        try:
            return super().set(self.properties.get(property, property), value)
        except TypeError: # 'property' must be an unknown string
            return super().set(eval('cv2.CAP_PROP_'+property.upper))

    def read(self, image=None):
        if image is not None:
            status, image = super().read(image)
        else:
            status, image = super().read()
        self.status, self.image = status, image
        return status, image


class SlowCamera(ContextualVideoCapture):
    ''' A basic, slow camera class for processing frames relatively far apart.

    Use 'Camera' instead unless you need to reduce power/CPU usage and the time
    to read an image is insignificant in your processing pipeline.

    '''
    def __init__(self, camera_id=0, *args, delay=1, **kwargs):
        ''' Create a camera capture instance with the given id.

        Arguments are the same as ContextualVideoCapture, but 'id' is replaced
            with 'camera_id', and 'delay' is set to 1 by default instead of
            None.

        '''
        super().__init__(camera_id, *args, delay=delay, **kwargs)

    def measure_framerate(self, frames):
        ''' Measure framerate for specified number of frames. '''
        count = 0
        for read_success, frame in self:
            if self.display:
                cv2.imshow(self.display, frame)
            count += 1
            if count == 1:
                start = perf_counter() # avoid timing opening the window
            if count > frames:
                # desired frames reached, set fps as average framerate
                return count / (perf_counter() - start)

    def __repr__(self):
        return f"{self.__class__.__name__}(camera_id={repr(self._id)})"


class Camera(SlowCamera):
    ''' A camera for always capturing the latest frame, fast.

    Use this instead of 'SlowCamera', unless you need to reduce power/CPU
    usage, and the time to read an image is insignificant in your processing
    pipeline.

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialise_grabber()

    def _initialise_grabber(self):
        ''' Start the Thread for grabbing images. '''
        self._finished = Event()
        self._image_grabber = Thread(name='grabber', target=self._grabber,
                                     daemon=True) # auto-kill when finished
        self._image_grabber.start()
        self._wait_for_grabber_start()

    def _grabber(self):
        ''' Grab images as fast as possible - only latest gets processed. '''
        while not self._finished.is_set():
            self.grab()

    def _wait_for_grabber_start(self):
        ''' Waits for a successful retrieve. Raises Exception after 50 attempts. '''
        for check in range(50):
            if self.retrieve()[0]: break
            sleep(0.1)
        else:
            raise Exception(f'Failed to start {self.__class__.__name__}')

    def __exit__(self, *args):
        self._finished.set()
        self._image_grabber.join()
        super().__exit__(*args)

    def read(self, image=None):
        ''' Read and return the latest available image. '''
        if image is not None:
            status, image = self.retrieve(image)
        else:
            status, image = self.retrieve()
        self.status, self.image = status, image
        return status, image


class LockedCamera(Camera):
    ''' A camera for semi-synchronously capturing a single image at a time.

    Like 'Camera' but uses less power+CPU by only capturing images on request.
    Allows specifying when each image should start being captured, then doing
    some processing while the image is being grabbed and decoded (and
    optionally pre-processed), before using it.

    Images may be less recent than achieved with Camera, depending on when the
    user starts the capture process within their processing pipeline, but can
    also be more recent if started near the end of the pipeline (at the risk of
    having to wait for the capturing to complete).

    '''
    def __init__(self, *args, preprocess=lambda img:img,
                 process=lambda img:img, **kwargs):
        ''' Create a camera capture instance with the given id.

        'preprocess' is an optional function which takes an image and returns
            a modified image, which gets applied to each frame on read.
            Defaults to no preprocessing.
        'process' is an optional function which takes an image and returns
            a modified image, which gets applied to each preprocessed frame
            after the next frame has been requested. Defaults to no processing.

        *args and **kwargs are the same as for Camera.

        '''
        super().__init__(*args, **kwargs)
        self._preprocess = preprocess
        self._process = process
        self._get_latest_image() # start getting the first image

    def _initialise_grabber(self):
        ''' Create locks and start the grabber thread. '''
        self._image_desired = Event()
        self._image_ready   = Event()
        super()._initialise_grabber()

    def _grabber(self):
        ''' Grab and preprocess images on demand, ready for later usage '''
        while not self._finished.is_set():
            self._wait_until_needed()
            # read the latest frame
            read_success, frame = super(ContextualVideoCapture, self).read()

            # apply any desired pre-processing and store for main thread
            self._preprocessed = self._preprocess(frame) if read_success \
                                 else None
            # inform that image is ready for access/main processing
            self._inform_image_ready()

    def _wait_for_grabber_start(self):
        ''' Not used - done automatically with Events. '''
        pass

    def _wait_until_needed(self):
        ''' Wait for main to request the next image. '''
        self._image_desired.wait()
        self._image_desired.clear()

    def _inform_image_ready(self):
        ''' Inform main that next image is available. '''
        self._image_ready.set()

    def _get_latest_image(self):
        ''' Ask camera handler for next image. '''
        self._image_desired.set()

    def _wait_for_camera_image(self):
        ''' Wait until next image is available. '''
        self._image_ready.wait()
        self._image_ready.clear()

    def __exit__(self, *args):
        self._finished.set()
        self._image_desired.set() # allow thread to reach finished check
        super().__exit__(*args)

    def read(self, image=None):
        ''' For optimal usage, tune _process to take the same amount of time
            as getting the next frame.
        '''
        self._wait_for_camera_image()
        preprocessed = self._preprocessed
        self._get_latest_image()
        if preprocessed is None:
            self.status, self.image = False, None
        else:
            self.image = self._process(preprocessed)
            if image is not None:
                image = self.image
            self.status = True
        return self.status, self.image


class VideoReader(LockedCamera):
    ''' A class for reading video files. '''
    properties = {
        **ContextualVideoCapture.properties,
        'frame'      : cv2.CAP_PROP_POS_FRAMES,
        'codec'      : cv2.CAP_PROP_FOURCC,
        'timestamp'  : cv2.CAP_PROP_POS_MSEC,
        'num_frames' : cv2.CAP_PROP_FRAME_COUNT,
        'proportion' : cv2.CAP_PROP_POS_AVI_RATIO,
    }

    FASTER, SLOWER, REWIND, FORWARD, RESET, RESTART = \
        (ord(key) for key in 'wsadrb')
    FORWARD_DIRECTION, REVERSE_DIRECTION = 1, -1
    MIN_DELAY = 1 # integer milliseconds

    def __init__(self, filename, *args, start=None, end=None, auto_delay=True,
                 fps=None, skip_frames=None, verbose=True, display='video',
                 **kwargs):
        ''' Initialise a video reader from the given file.

        For default key-bindings see 'auto_delay' details.

        'filename' is the string path of a video file. Depending on the file
            format some features may not be available.
        'start' and 'end' denote the respective times of playback, according
            to the specified fps. They can be integers of milliseconds, or
            strings of 'hours:minutes:seconds' (larger amounts can be left off
            if 0, e.g. '5:10.35' for no hours). If left as None, the video
            starts and ends at the first and last frames respectively.
            It is expected that 'start' < 'end', or playback ends immediately.
        'auto_delay' is a boolean specifying if the delay between frames should
            be automatically adjusted during playback to match the specified
            fps. Set to False if operating headless (not viewing the video), or
            if manual control is desired while iterating over the video.
              If set to False, sets 'destroy' to None if not otherwise set.
              If True enables playback control with 'w' increasing playback
            speed, 's' slowing it down, 'a' rewinding (only possible if
            'skip_frames' is True), and 'd' returning to forwards playback.
              The 'r' key can be pressed to reset to 1x speed and forwards
            direction playback. 'a' and 'd' can be used while paused to step
            back and forwards, regardless of skip_frames. 'b' can be used while
            playing or paused to jump the video back to its starting point.
            These defaults can be overridden using the 'play_commands' and
            'pause_effects' keyword arguments, supplying a dictionary of key
            ordinals that sets the desired behaviour. Note that the defaults
            are set internally, so to turn them off the dictionary must be
            used, with e.g. play_commands={ord('a'):lambda vid:None} to disable
            rewinding.
        'fps' is a float specifying the desired frames per second for playback.
            If left as None the fps is read from file, or if that fails is set
            to 25 by default. Value is ignored if 'auto_delay' is False.
        'skip_frames' allows frames to be manually set, as required by reverse
            or high speed playback. If left as None this is disallowed. If
            'auto_delay' is True, any integer value can be set (suggested 0),
            and the number of frames to skip at each iteration is determined
            as part of the delay tuning. If 'auto_delay' is False, an integer
            can be used as a consistent number of frames to skip at each
            iteration (e.g. only read every 10th frame). Note that enabling
            frame skipping can make playback jerky on devices and/or file
            formats with slow video frame setting times, and inconsistent
            skipping amounts with 'auto_delay' may cause issues with
            time-dependent processing.
        'verbose' is a boolean determining if status updates (e.g. initial fps,
            and playback speed and direction changes) are printed. Defaults to
            True.

        *args and **kwargs get passed up the inheritance chain, with notable
            keywords including the 'preprocess' and 'process' functions which
            take an image and return a processed result (see LockedCamera),
            the 'quit' and 'play_pause' key ordinals which are checked if
            'auto_delay' is True, and the 'play_commands' and 'pause_effects'
            dictionaries mapping key ordinals to desired functionality while
            playing and paused (see ContextualVideoCapture documentation for
            details).

        '''
        super().__init__(filename, *args, display=display, **kwargs)
        self.filename = filename
        self._fps = fps or self.fps or 25 # user-specified or auto-retrieved
        self._period = 1e3 / self._fps
        self._verbose = verbose
        self.status = True
        self._initialise_delay(auto_delay)
        self._initialise_playback(start, end, skip_frames)

    def _initialise_delay(self, auto_delay):
        ''' Determines the delay automatically, or leaves as None. '''
        if auto_delay:
            if self._fps == 0 or self._fps >= 1e3:
                self.verbos_print('failed to determine fps, setting to 25')
                self._period = 1e3 / 25
                # set a bit low to allow image read times
                self._delay = self._period - 5
            else:
                self._delay = int(self._period)
                self.verbose_print('delay set automatically to',
                                   f'{self._delay}ms from fps={self._fps}')
        else:
            self._delay = None
            if self._destroy == -1:
                self._destroy = None

    def _initialise_playback(self, start, end, skip_frames):
        ''' Set up playback settings as specified. '''
        self._wait_for_camera_image() # don't set frame while grabber running

        self._set_start(start)
        self._set_end(end)

        self._skip_frames     = skip_frames
        self._direction       = self.FORWARD_DIRECTION
        self._speed           = 1
        self._adjusted_period = self._period
        self._calculate_frames()

        self._play_commands = {
            self.FASTER  : self._speed_up,
            self.SLOWER  : self._slow_down,
            self.REWIND  : self._go_back,
            self.FORWARD : self._go_forward,
            self.RESET   : self._reset,
            self.RESTART : self.restart,
            **self._play_commands
        }

        # add step back and forward functionality if keys not already used
        self._pause_effects = {
            self.REWIND  : self.step_back,
            self.FORWARD : self.step_forward,
            self.RESTART : self.restart,
            **self._pause_effects
        }

        # ensure time between frames is ignored while paused
        class LogDict(dict):
            def get(this, *args, **kwargs):
                self.reset_delay()
                return dict.get(this, *args, **kwargs)

        self._pause_effects = LogDict(self._pause_effects)

        self._get_latest_image() # re-initialise as ready

    def _set_start(self, start):
        ''' Set the start of the video to user specification, if possible. '''
        self._frame = 0
        if start is not None:
            if self.set_timestamp(start):
                self.verbose_print(f'starting at {start}')
            else:
                self.verbose_print('start specification failed, '
                                   'starting at 0:00')
        self._start = self._frame

    def _set_end(self, end):
        ''' Set playback to end where specified by user. '''
        if end is not None:
            if isinstance(end, str):
                self._end = self.timestamp_to_ms(end)
            else:
                self._end = end
            self._end /= self._period # convert to number of frames
        else:
            self._end = self.get('num_frames') or np.inf

    def verbose_print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    # NOTE: key callbacks set as static methods for clarity/ease of reference
    #    VideoReader to be modified gets passed in (so that external functions
    #    can be used), so also having a reference to self would be confusing.

    @staticmethod
    def _speed_up(vid):
        ''' Increase the speed by 10% of the initial value. '''
        vid._speed += 0.1
        vid._register_speed_change()

    @staticmethod
    def _slow_down(vid):
        ''' Reduce the speed by 10% of the initial value. '''
        vid._speed -= 0.1
        vid._register_speed_change()

    def _register_speed_change(self):
        ''' Update internals and print new speed. '''
        self._calculate_period()
        self.verbose_print(f'speed set to {self._speed:.1f}x starting fps') 

    def _calculate_period(self):
        ''' Determine the adjusted period given the speed. '''
        self._adjusted_period = self._period / self._speed
        self._calculate_timestep()

    def _calculate_timestep(self):
        ''' Determine the desired timestep of each iteration. '''
        self._timestep = self._adjusted_period * self._frames

    def _calculate_frames(self):
        ''' Determine the number of frames to increment each iteration. '''
        self._frames = (1 + self._skip_frames
                        if self._skip_frames is not None
                        else 1)
        self._calculate_timestep()

    def reset_delay(self):
        ''' Resets the delay between frames.

        Use to avoid fast playback/frame skipping after pauses.

        '''
        self._prev = perf_counter() - (self._period - self.MIN_DELAY) / 1e3

    @staticmethod
    def _go_back(vid):
        ''' Set playback to backwards. '''
        if vid._skip_frames is not None:
            vid._direction = vid.REVERSE_DIRECTION
            vid.verbose_print('Rewinding')
        else:
            vid.verbose_print('Cannot go backwards without skip_frames=True')

    @staticmethod
    def _go_forward(vid):
        ''' Set playback to go forwards. '''
        vid._direction = vid.FORWARD_DIRECTION
        vid.verbose_print('Going forwards')

    @staticmethod
    def _reset(vid):
        ''' Restore playback to 1x speed and forwards. '''
        vid._speed = 1
        vid._direction = vid.FORWARD_DIRECTION
        vid._calculate_period()
        vid.verbose_print('Going forwards with speed set to 1x starting fps '
                          f'({vid._fps:.2f})')

    @staticmethod
    def step_back(vid):
        ''' Take a step backwards. '''
        # store existing state
        old_state = (vid._skip_frames, vid._direction, vid._verbose)

        # enable back-stepping if not currently permitted
        vid._skip_frames = 0
        # make sure no unnecessary prints trigger from playback keys
        vid._verbose = False 

        # go back a step
        vid._direction = vid.REVERSE_DIRECTION
        next(vid)

        # restore state
        vid._skip_frames, vid._direction, vid._verbose = old_state

    @staticmethod
    def step_forward(vid):
        ''' Take a step forwards. '''
        # store existing state
        old_state = (vid._direction, vid._verbose)

        # make sure no unnecessary prints trigger from playback keys
        vid._verbose = False

        # go forwards a step
        vid._direction = vid.FORWARD_DIRECTION
        next(vid)

        # restore state
        vid._direction, vid._verbose = old_state

    @staticmethod
    def restart(vid):
        ''' Attempts to continue playback from the start of the video.

        Respects user-defined start-point from initialisation.

        '''
        vid.set_frame(vid._start)

    @property
    def fps(self):
        ''' The constant FPS assumed of the video file. '''
        return self.get('fps')

    @property
    def frame(self):
        ''' Retrieve the current video frame. '''
        self._frame = int(self.get('frame'))
        return self._frame

    def set_frame(self, frame):
        ''' Attempts to set the frame number, returns success.

        'frame' is an integer >= 0. Setting past the last frame
            either has no effect or ends the playback.

        self.set_frame(int) -> bool

        '''
        if self.set('frame', frame):
            self._frame = frame
            return True
        return False

    @property
    def timestamp(self):
        ''' Returns the video timestamp if possible, else 0.0.

        Returns a human-readable time string, as hours:minutes:seconds.
        For the numerical ms value use self.get('timestamp') instead.

        self.timestamp -> str

        '''
        # cv2.VideoCapture returns ms timestamp -> convert to meaningful time
        seconds          = self.get('timestamp') / 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes   = divmod(minutes, 60)
        if hours:
            return f'{hours}:{minutes}:{seconds:.3f}'
        if minutes:
            return f'{minutes}:{seconds:.3f}'
        return f'{seconds:.3f}s'

    def set_timestamp(self, timestamp):
        ''' Attempts to set the timestamp as specified, returns success.

        'timestamp' can be a float/integer of milliseconds, or a string
            of 'hours:minutes:seconds', 'minutes:seconds', or 'seconds',
            where all values can be integers or floats.

        self.set_timestamp(str/float/int) -> bool

        '''
        ms = self.timestamp_to_ms(timestamp) if isinstance(timestamp, str) \
            else timestamp
        fps = self._fps
        if fps == 0:
            # fps couldn't be determined - set ms directly and hope
            return self.set('timestamp', ms)
        return self.set_frame(int(ms * fps / 1e3))

    @staticmethod
    def timestamp_to_ms(timestamp):
        ''' Converts a string timestamp of hours:minutes:seconds to ms.'''
        return 1000 * sum(60 ** index * float(period) for index, period \
                          in enumerate(reversed(timestamp.split(':'))))

    def __iter__(self):
        if self._delay is not None:
            self._prev  = perf_counter()
            self._error = 0
            self._delay = 1
        return self

    def __next__(self):
        if self._delay is not None:
            # auto-adjust to get closer to desired fps
            now = perf_counter()
            diff = 1e3 * (now - self._prev) # s to ms
            self._error += diff - self._timestep

            self._update_playback_settings()

            self._prev = now

        self._update_frame_tracking()
        read_success, frame = super().__next__()
        if not read_success:
            raise OutOfFrames
        return read_success, frame

    def _update_playback_settings(self):
        ''' Adjusts delay/frame skipping if error is sufficiently large. '''
        error_magnitude = abs(self._error)
        if error_magnitude > self.MIN_DELAY:
            # determine distribution of change
            if self._skip_frames is not None:
                # can only skip full frames, rest left to delay
                skip_frames_change, delay_change = \
                    divmod(error_magnitude, self._adjusted_period)
            else:
                delay_change = error_magnitude
            # can only delay in MIN_DELAY increments, remainder is error
            delay_change, new_error_mag = \
                divmod(delay_change, self.MIN_DELAY)

            # determine if going too slowly (+) or too fast (-)
            sign = np.sign(self._error)
            # implement delay (and skip frames) change
            # reducing delay increases speed
            self._delay -= int(sign * delay_change)
            if self._skip_frames is not None:
                # skipping additional frames increases speed
                self._skip_frames += int(sign * skip_frames_change)
                self._calculate_frames() # update internals

            self._error = sign * new_error_mag
            if self._delay < self.MIN_DELAY:
                self._error += self.MIN_DELAY - self._delay
                self._delay = self.MIN_DELAY

    def _update_frame_tracking(self):
        # frame skip with no auto-delay allows continual frame skipping
        # only set frame if necessary (moving one frame ahead isn't helpful)
        if self._skip_frames is not None and \
           (self._direction == -1 or self._frames != 1):
            self._image_ready.wait()
            self.set_frame(self._frame + self._frames * self._direction)
        else:
            self._frame += 1

        if self.status == False or self._frame > self._end \
           or self._frame < self._start:
            raise OutOfFrames

    def __repr__(self):
        return f"{self.__class__.__name__}(filename={repr(self.filename)})"


if __name__ == '__main__':
    with Camera(0) as cam:
        cam.stream()

