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


class OutOfFrames(StopIteration):
    def __init__(msg='Out of video frames', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class ContextualVideoCapture(cv2.VideoCapture):
    ''' A cv2.VideoCapture with a context manager for releasing. '''
    properties={}

    def __init__(self, id, windows=None, *args, delay=None, quit=ord('q'),
                 play_pause=ord('p'), **kwargs):
        ''' Destroys window on context exit, if specified, or 'all'.

        'delay' is the integer millisecond delay applied between each iteration
            to enable windows to update. If set to None, this is skipped and
            the user must manually call waitKey to update windows.
            Default is 1 ms. For headless operation, set to None.
        'quit' is an integer ordinal corresponding to a key which can be used
            to stop the iteration loop. Only applies if delay is not None.
            Default is ord('q'), so press the 'q' key to quit when iterating.
        'play_pause' is an integer ordinal corresponding to a key which can be
            used to pause and resume the iteration loop. Only applies if delay
            is not None. Default is ord('p'), so press 'p' to pause/resume when
            iterating.

        '''
        super().__init__(id, *args, **kwargs)
        self._windows = windows
        self._delay   = delay
        self._quit    = quit
        self._play_pause = play_pause


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
            cv2.destroyWindow(windows)
        elif windows is not None:
            for w in windows: cv2.destroyWindow(w)

    def __iter__(self):
        return self

    def __next__(self):
        if self._delay is not None:
            key = waitKey(self._delay)
            if key == self._quit:
                raise StopIteration # user quit manually
            elif key == self._play_pause:
                while "paused":
                    key = waitKey(1)
                    if key == self._quit:
                        raise StopIteration
                    if key == self._play_pause:
                        break
        if self.isOpened():
            return self.read()
        raise StopIteration # out of frames

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



class SlowCamera(ContextualVideoCapture):
    ''' A basic, slow camera class for processing frames relatively far apart.

    Use 'Camera' instead unless you need to reduce power/CPU usage and the time
    to read an image is insignificant in your processing pipeline.

    '''
    properties={} # TODO camera-related properties
    def __init__(self, camera_id=0, *args, delay=1, **kwargs):
        ''' Create a camera capture instance with the given id.

        '''
        super().__init__(camera_id, *args, delay=delay, **kwargs)
        self._id = camera_id

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
            read_success, frame = ContextualVideoCapture.read(self)
            if not read_success:
                raise IOError('Failure to read frame from camera.')

            # apply any desired pre-processing and store for main thread
            self._frame = self._preprocess(frame)
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
        return True, self._frame


class VideoReader(LockedCamera):
    ''' A class for reading video files. '''
    properties = {
        'fps'        : cv2.CAP_PROP_FPS,
        'timestamp'  : cv2.CAP_PROP_POS_MSEC,
        'frame'      : cv2.CAP_PROP_POS_FRAMES,
        'proportion' : cv2.CAP_PROP_POS_AVI_RATIO,
        # TODO the rest
        # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    }

    def __init__(self, filename, *args, start=None, delay=-1, fps=None,
                 preprocess=None, **kwargs):
        ''' Initialise a video reader from the given file.

        'delay' can be set manually to ms between each iteration. If set to -1,
            is automatically set using the video FPS. Set to None if operating
            headless (not viewing the video)
        TODO fps documentation (auto-adjustment of delay)

        'preprocess' is an optional function which takes an image and returns
            a modified image. Defaults to no preprocessing.

        '''
        super().__init__(filename, *args, delay=delay, **kwargs)
        self._fps = fps or self.fps # user-specified or auto-retrieved
        self._period = 1e3 / self._fps

        # handle delay automatic option
        if delay == -1:
            if self._fps == 0 or self._fps >= 1e3:
                print('failed to determine fps, setting delay to 25ms')
                self._delay  = 25
                self._period = self._delay
            else:
                self._delay = int(self._period)
                print('delay set automatically to',
                      f'{self._delay}ms from FPS={self._fps}')

        self.filename = filename
        if start is not None:
            if self.set_timestamp(start):
                print(f'starting at {start}')
            else:
                print('start specification failed, starting at 0:00')

        if preprocess is not None:
            self._preprocess = preprocess
        self._prev_frame = super().read()[1]

    def _grabber(self):
        ''' Grab images on demand, ready for later usage '''
        try:
            super()._grabber()
        except IOError:
            self._frame = None
            self._inform_image_ready()

    @property
    def fps(self):
        ''' The constant FPS assumed of the video file. '''
        return self.get('fps')

    @property
    def frame(self):
        return self.get('frame')

    def set_frame(self, frame):
        ''' Attempts to set the frame number, returns success.

        'frame' is an integer that must be less than TODO MAX_FRAMES.

        self.set_frame(int) -> bool

        '''
        return self.set('frame', frame)

    @property
    def timestamp(self):
        ''' Returns the video timestamp if possible, else 0.0. '''
        # cv2.VideoCapture returns ms timestamp -> convert to meaningful time
        seconds          = self.get('timestamp') / 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes   = divmod(minutes, 60)
        output = ''
        if hours:
            return f'{hours}:{minutes}:{seconds:.3f}'
        if minutes:
            return f'{minutes}:{seconds:.3f}'
        return '{seconds:.3f}s'

    def set_timestamp(self, timestamp):
        ''' Attempts to set the timestamp as specified, returns success.

        'timestamp' can be a float/integer of milliseconds, or a string
            of 'hours:minutes:seconds', 'minutes:seconds', or 'seconds',
            where all values can be integers or floats.

        self.set_timestamp(str/float/int) -> bool

        '''
        if isinstance(timestamp, str):
            # convert string to seconds
            timestamp = sum(60 ** index * float(period) for index, period \
                            in enumerate(reversed(timestamp.split(':'))))
            timestamp *= 1000
        fps = self._fps
        if fps == 0:
            # fps couldn't be determined - set ms directly and hope
            return self.set('timestamp', timestamp)
        return self.set_frame(int(timestamp * fps / 1e3))

    @property
    def proportion(self):
        ''' Returns the progress of the video as a float between 0 and 1. '''
        return self.get('proportion')

    def set_proportion(self, proportion):
        ''' Attempts to set the video progress proportion, returns success.

        'proportion' is a float value between 0 and 1.

        self.set_proportion(float) -> bool

        '''
        return self.set('proportion', proportion)

    def read(self):
        self._get_latest_image()
        cv2.imshow('video', self._prev_frame)
        self._wait_for_camera_image()
        if self._frame is None:
            raise OutOfFrames
        self._prev_frame = self._frame
        return True, self._frame

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
            diff = 1e3 * (now - self._prev)
            self._error += diff - self._period
            mag = abs(self._error)
            if mag > 1:
                sign = np.sign(self._error)
                delay_change, new_error_mag = divmod(mag, 1)
                self._delay -= int(sign * delay_change)
                self._error = sign * (new_error_mag % self._period)
                if self._delay < 1:
                    self._error += 1 - self._delay
                    self._delay = 1
            self._prev = now
        return super().__next__()

    def __repr__(self):
        return f"VideoReader(filename={self.filename:!r})"


if __name__ == '__main__':
    with Camera(0) as cam:
        for read_success, frame in cam:
            cv2.imshow('frame', frame)
