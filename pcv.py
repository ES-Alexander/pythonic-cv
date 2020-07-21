#!/usr/bin/env python3

import cv2
import numpy as np
from time import time
from threading import Thread, Lock


waitKey = lambda ms : cv2.waitKey(ms) & 0xFF

def downsize(img, ratio):
    ''' downsize 'img' by 'ratio'. '''
    return cv2.resize(img, tuple(np.array(img.shape[:2][::-1]) // ratio),
                      0, 0, cv2.INTER_AREA)

def channel_options(img, rank=False):
    ''' Create a composite image of img in all of opencv's colour channels

    |img| -> | blue       | green      | red         |
             | hue        | saturation | value       |
             | hue2       | luminosity | saturation2 |
             | lightness  | green-red  | blue-yellow |
             | lightness2 | u          | v           |

    'rank' is a boolean? specifying whether to also return a ranking of each
        channel by variability/sharpness/contrast/other? !NOT YET IMPLEMENTED!
        TODO
        -> make a string maybe, with several options available, or select
            multiple options in a list and get back an array or dataframe or
            something
        -> important to make nicely stackable to run on video and determine
           statistics on the best option for a given use case

    '''
    B,G,R = cv2.split(img)
    H,S,V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    H2,L2,S2 = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    L,a,b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    L3,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LUV))
    channels = (((B, 'blue'), (G, 'green'), (R, 'red')),
                ((H, 'hue'), (S, 'saturation'), (V, 'value')),
                ((H2, 'hue2'), (L2, 'luminosity'), (S2, 'saturation2')),
                ((L, 'lightness'), (a, 'green-red'), (b, 'blue-yellow')),
                ((L3,'lightness2'), (u, 'u'), (v, 'v')))
    out = []
    for row in channels:
        img_row = []
        for img, name in row:
            cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, 255, 1)
            img_row.append(img)
        out.append(cv2.hconcat(img_row))
    return cv2.vconcat(out)

class VideoWriter(cv2.VideoWriter):
    ''' A cv2.VideoWriter with a context manager for releasing. '''
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
        self.filename       = filename
        self.fps            = fps
        self.is_color       = isColor
        self.frame_size     = frameSize
        self.api_preference = apiPreference
        self.set_fourcc(fourcc)

        super().__init__(*self._construct_open_args())

    def set_fourcc(self, fourcc):
        ''' Set fourcc code as an integer or an iterable of 4 chars. '''
        if not isinstance(fourcc, int):
            # assume iterable of 4 chars
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.fourcc = fourcc

    def _construct_open_args(self):
        args = [self.filename, self.fourcc, self.fps, self.frame_size,
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
                    apiPreference=None):
        ''' Returns a VideoWriter based on the properties of the input camera.
        '''
        if fourcc is None:
            fourcc = cls.suggested_codec(filename)
        frameSize = tuple(int(camera.get(dim)) for dim in ('width','height')) 

        return cls(filename, fourcc, camera.get('fps'), frameSize,
                   isColor, apiPreference)


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

    def __init__(self, id, windows=None, *args, delay=None, quit=ord('q'),
                 play_pause=ord(' '), pause_effects={}, **kwargs):
        ''' A pausable, quitable, iterable video-capture object
            with context management.

        'windows' destroys any specified windows on context exit. Can be 'all'
            to destroy all active opencv windows, a string of a specific window
            name, or a list of window names to close.
        'id' is the id that gets passed to the underlying VideoCapture object.
            it can be an integer to select a connected camera, or a filename
            to open a video
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

        '''
        super().__init__(id, *args, **kwargs)
        self._windows       = windows
        self._delay         = delay
        self._quit          = quit
        self._play_pause    = play_pause
        self._pause_effects = pause_effects

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        ''' Clean up on context exit.

        Releases the internal VideoCapture object, and destroys any windows
            specified at initialisation.

        '''
        # release VideoCapture object
        self.release()

        # clean up window(s) if specified on initialisation
        windows = self._windows
        if windows == 'all':
            cv2.destroyAllWindows()
        elif isinstance(windows, str):
            # a single window name
            cv2.destroyWindow(windows)
        elif windows is not None:
            # assume an iterable of multiple windows
            for w in windows: cv2.destroyWindow(w)

    def __iter__(self):
        return self

    def __next__(self, key=None):
        # check if doing automatic waits
        if self._delay is not None:
            if key is None:
                key = waitKey(self._delay)
            # else wait has already occurred with key passed in

            if key == self._quit:
                raise UserQuit
            elif key == self._play_pause:
                self._handle_pause()

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

    def read(self):
        self.status, self.image = super().read()
        return self.status, self.image


class SlowCamera(ContextualVideoCapture):
    ''' A basic, slow camera class for processing frames relatively far apart.

    Use 'Camera' instead unless you need to reduce power/CPU usage and the time
    to read an image is insignificant in your processing pipeline.

    '''
    def __init__(self, camera_id=0, *args, delay=1, **kwargs):
        ''' Create a camera capture instance with the given id.

        '''
        super().__init__(camera_id, *args, delay=delay, **kwargs)
        self._id = camera_id

    def stream(self, window='frame'):
        for read_success, frame in self:
            cv2.imshow(window, frame)

    def __repr__(self):
        return f"{self.__class__.__name__}(camera_id={self._id:!r})"


class Camera(SlowCamera):
    ''' A camera for always capturing the latest frame, fast.

    Use this instead of 'SlowCamera', unless you need to reduce power/CPU
    usage, and the time to read an image is insignificant in your processing
    pipeline.

    '''
    def __init__(self, camera_id=0, *args, **kwargs):
        super().__init__(camera_id, *args, **kwargs)
        self._initialise_grabber()

    def _initialise_grabber(self):
        ''' Start the Thread for grabbing images. '''
        self._image_grabber = Thread(name='grabber', target=self._grabber,
                                     daemon=True) # auto-kill when finished
        self._image_grabber.start()

    def _grabber(self):
        ''' Grab images as fast as possible - only latest gets processed. '''
        while 'running':
            self.grab()

    def read(self):
        ''' Read and return the latest available image. '''
        return self.retrieve()


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
    def _initialise_grabber(self):
        ''' '''
        self._lock1 = Lock()
        self._lock2 = Lock()
        self._lock1.acquire() # start with main thread in control
        super()._initialise_grabber()

    def _grabber(self):
        ''' Grab images on demand, ready for later usage '''
        while "running":
            self._wait_until_needed()
            # read the latest frame
            read_success, frame = super(Camera, self).read()
            if not read_success:
                raise IOError('Failure to read frame from camera.')

            # apply any desired pre-processing and store for main thread
            self.image = self._preprocess(frame)
            # inform that image is ready for access/main processing
            self._inform_image_ready()

    def _wait_until_needed(self):
        ''' Wait for main to request the next image. '''
        self._lock2.acquire() # wait until previous image has been received
        self._lock1.acquire() # wait until next image is desired
        self._lock2.release() # inform that camera is taking image

    def _inform_image_ready(self):
        ''' Inform main that next image is available. '''
        self._lock1.release() # inform that next image is ready

    def _get_latest_image(self):
        ''' Ask camera handler for next image. '''
        self._lock1.release() # inform that next image is desired
        self._lock2.acquire() # wait until camera is taking image

    def _wait_for_camera_image(self):
        ''' Wait until next image is available. '''
        self._lock1.acquire() # wait until next image is ready
        self._lock2.release() # inform that image has been received

    def _preprocess(self, image):
        ''' Perform desired pre-processing in the grabber thread. '''
        return image

    def read(self):
        ''' Enable manual reading and iteration FOR TESTING ONLY.

        If your application can run without issue using this read command
        or by iterating over the instance, use SlowCamera instead as it will be
        marginally faster.

        Correct use of this class should at least have something occurring
        between the calls to the methods self._get_latest_image() and
        self._wait_for_camera_image() in order to gain benefit from the
        multithreading.

        '''
        self._get_latest_image()
        self._wait_for_camera_image()
        return True, self.image


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

    FASTER, SLOWER, REWIND, FORWARD, RESET, STEP_BACK, STEP_FORWARD = \
        (ord(key) for key in 'wsadrbf')
    MIN_DELAY = 1 # delay is integer milliseconds

    def __init__(self, filename, *args, start=None, end=None, auto_delay=True,
                 fps=None, skip_frames=None, preprocess=None,
                 display_window='video', **kwargs):
        ''' Initialise a video reader from the given file.

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
            if manual control is desired while iterating over the video. If
            True, also enables playback control with 'w' increasing playback
            speed, 's' slowing it down, 'a' rewinding (only possible if
            'skip_frames' is True), and 'd' returning to forwards playback.
            The 'r' key can be pressed to reset to 1x speed and forwards
            direction playback.
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
        'preprocess' is an optional function which takes an image and returns
            a modified image, which gets applied to each frame on read.
            Defaults to no preprocessing.
        'display_window' is the string name of the window to display the
            video in (if auto_delay is True) when iterating over the instance
            or using the 'play' method. Defaults to 'video'.

        *args and **kwargs get passed up the inheritance chain, with notable
            keywords including the 'quit' and 'play_pause' key ordinals which
            are checked if 'auto_delay' is True, and the 'pause_effects'
            dictionary mapping key ordinals to desired functionality while
            paused (see ContextualVideoCapture documentation for details).

        '''
        # destroy display_window unless asked not to
        kwargs = {'windows':display_window, **kwargs}
        super().__init__(filename, *args, **kwargs)
        self.filename = filename
        self._fps = fps or self.fps # user-specified or auto-retrieved
        self._period = 1e3 / self._fps
        self._initialise_delay(auto_delay)
        self._initialise_playback(start, end, skip_frames)
        self.display_window = display_window

        if preprocess is not None:
            self._preprocess = preprocess

        self._prev_frame = super().read()[1]

    def _initialise_delay(self, auto_delay):
        ''' Determines the delay automatically, or leaves as None. '''
        if auto_delay:
            if self._fps == 0 or self._fps >= 1e3:
                print('failed to determine fps, setting to 25')
                self._period = 1e3 / 25
                # set a bit low to allow image read times
                self._delay  = self._period - 5
            else:
                self._delay = int(self._period)
                print('delay set automatically to',
                      f'{self._delay}ms from fps={self._fps}')
        # else self._delay defaults to None

    def _initialise_playback(self, start, end, skip_frames):
        ''' Set up playback settings as specified. '''
        self._set_start(start)
        self._set_end(end)

        self._skip_frames = skip_frames
        self._direction = 1
        self._speed = 1
        self._adjusted_period = self._period
        self._calculate_frames()

        self._playback_commands = {
            self.FASTER  : self._speed_up,
            self.SLOWER  : self._slow_down,
            self.REWIND  : self._go_back,
            self.FORWARD : self._go_forward,
            self.RESET   : self._reset,
        }

        # add step back and forward functionality if keys not already used
        self._pause_effects = {
            self.STEP_BACK    : self.step_back,
            self.STEP_FORWARD : self.step_forward,
            **self._pause_effects
        }

        # ensure time between frames is ignored while paused
        class LogDict(dict):
            def get(this, *args, **kwargs):
                self._prev = time() - (self._period - self.MIN_DELAY) / 1e3
                return dict.get(this, *args, **kwargs)

        self._pause_effects = LogDict(self._pause_effects)

    def _set_start(self, start):
        ''' Set the start of the video to user specification, if possible. '''
        if start is not None:
            if self.set_timestamp(start):
                print(f'starting at {start}')
            else:
                print('start specification failed, starting at 0:00')
                self._frame = 0
        else:
            self._frame = 0

    def _set_end(self, end):
        ''' Set playback to end where specified by user. '''
        if end is not None:
            if isinstance(end, str):
                self._end = self.timestamp_to_ms(end)
            else:
                self._end = end
            self._end /= self._period # convert to number of frames
        else:
            self._end = np.inf

    def _speed_up(self):
        ''' Increase the speed by 10% of the initial value. '''
        self._speed += 0.1
        self._register_speed_change()

    def _slow_down(self):
        ''' Reduce the speed by 10% of the initial value. '''
        self._speed -= 0.1
        self._register_speed_change()

    def _register_speed_change(self):
        ''' Update internals and print new speed. '''
        self._calculate_period()
        print(f'speed set to {self._speed:.1f}x starting fps') 

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

    def _go_back(self):
        ''' Set playback to backwards. '''
        if self._skip_frames is not None:
            self._direction = -1
            print('Rewinding')
        else:
            print('Cannot go backwards without skip_frames=True')

    def _go_forward(self):
        ''' Set playback to go forwards. '''
        self._direction = 1
        print('Going forwards')

    def _reset(self):
        ''' Restore playback to 1x speed and forwards. '''
        self._speed = 1
        self._direction = 1
        self._calculate_period()
        print(f'Going forwards with speed set to 1x starting fps')

    @staticmethod
    def step_back(vid):
        ''' Take a step backwards. '''
        # store existing state
        old_skip = vid._skip_frames
        old_direction = vid._direction

        # enable back-stepping if not currently permitted
        vid._skip_frames = 0

        # go back a step
        vid._direction = -1
        next(vid)

        # restore state
        vid._direction = old_direction
        vid._skip_frames = old_skip

    @staticmethod
    def step_forward(vid):
        ''' Take a step forwards. '''
        # store existing state
        old_direction = vid._direction

        # go forwards a step
        vid._direction = 1
        next(vid)

        # restore state
        vid._direction = old_direction

    def _grabber(self):
        ''' Grab images on demand, ready for later usage '''
        try:
            super()._grabber()
        except IOError:
            self.image = None
            self._inform_image_ready()

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

        'frame' is an integer greater than 0. Setting past the last frame
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

    def read(self):
        self._get_latest_image()
        if self._delay is not None:
            # display previous image while getting the next one
            cv2.imshow(self.display_window, self._prev_frame)
        self._wait_for_camera_image()
        if self.image is None:
            raise OutOfFrames
        self._prev_frame = self.image
        return True, self.image

    def play(self):
        for read_success, frame in self: pass

    def __iter__(self):
        if self._delay is not None:
            self._prev  = time()
            self._error = 0
            self._delay = 1
        return self

    def __next__(self):
        if self._delay is not None:
            # auto-adjust to get closer to desired fps
            now = time()
            diff = 1e3 * (now - self._prev) # s to ms
            self._error += diff - self._timestep

            self._update_playback_settings()

            self._prev = now

            key = waitKey(self._delay)
            self._handle_playback(key)
        else:
            self._update_frame_tracking()

        return super().__next__(key=key)

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

            self._update_frame_tracking()

            self._error = sign * new_error_mag
            if self._delay < self.MIN_DELAY:
                self._error += self.MIN_DELAY - self._delay
                self._delay = self.MIN_DELAY

    def _handle_playback(self, key):
        self._playback_commands.get(key, lambda : None)()

    def _update_frame_tracking(self):
        # frame skip with no auto-delay allows continual frame skipping
        # only set frame if necessary (moving one frame ahead isn't helpful)
        if self._skip_frames is not None and \
           (self._direction == -1 or self._frames != 1):
            self.set_frame(self._frame + self._frames * self._direction)
        else:
            self._frame += 1

        if self._frame > self._end:
            raise OutOfFrames

    def __repr__(self):
        return f"VideoReader(filename={self.filename:!r})"


if __name__ == '__main__':
    with Camera(0) as cam, VideoWriter.from_camera('me.mp4', cam) as writer:
        print("press 'q' to quit.")
        for read_success, frame in cam:
            if read_success:
                cv2.imshow('frame', frame)
                writer.write(frame)
            else:
                break # camera disconnected
