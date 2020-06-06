#!/usr/bin/env python3

import cv2
from threading import Thread, Lock

waitKey = lambda ms : cv2.waitKey(ms) & 0xFF

class ContextualVideoCapture(cv2.VideoCapture):
    ''' A cv2.VideoCapture with a context manager for releasing. '''
    properties={}

    def __init__(self, id, windows=None, *args, **kwargs):
        ''' Destroys window on context exit, if specified, or 'all'. '''
        super().__init__(id, *args, **kwargs)
        self._windows = windows

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


class VideoReader(ContextualVideoCapture):
    ''' A class for reading video files. '''
    properties = {
        'fps'        : cv2.CAP_PROP_FPS,
        'timestamp'  : cv2.CAP_PROP_POS_MSEC,
        'frame'      : cv2.CAP_PROP_POS_FRAMES,
        'proportion' : cv2.CAP_PROP_POS_AVI_RATIO,
        # TODO the rest
        # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    }

    def __init__(self, filename, *args, start=None, **kwargs):
        ''' Initialise a video reader from the given file. '''
        super().__init__(filename, *args, **kwargs)
        self.filename = filename
        if start is not None:
            self.set_timestamp(start)
 
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
            timestamp = sum(60 ** index * period \
                            for index, period in timestamp.split(':'))
        return self.set('timestamp', timestamp)

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

    def __repr__(self):
        return f"VideoReader(filename={self.filename:!r})"


class SlowCamera(ContextualVideoCapture):
    ''' A basic, slow camera class for processing frames relatively far apart.

    Use 'Camera' instead unless you need to reduce power/CPU usage and the time
    to read an image is insignificant in your processing pipeline.

    '''
    properties={} # TODO camera-related properties
    def __init__(self, camera_id=0, *args, **kwargs):
        super().__init__(camera_id, *args, **kwargs)
        self._id = camera_id


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

    Like 'Camera' but uses less power+CPU by only capturing a single frame per
    iteration, and allows specifying when the next frame should start being
    captured.

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
