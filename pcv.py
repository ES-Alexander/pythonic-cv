#!/usr/bin/env python3

import cv2
from threading import Thread

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
        ''' Return 'property' if it exists, else raise Exception.
        TODO check raised exception and specify correctly.

        '''
        try:
            return super().get(self.properties.get(property, property))
        except: # figure out what to except for meaningful errors
            return super().get(eval('cv2.CAP_PROP_'+property.upper()))

    def set(self, property, value):
        ''' Attempts to set 'property' to 'value', returning success. '''
        try:
            return super().set(self.properties.get(property, property), value)
        except:
            return super().set(eval('cv2.CAP_PROP_'+property.upper))


class VideoReader(ContextualVideoCapture):
    ''' A class for reading video files. '''
    properties = {
        'fps'       : cv2.CAP_PROP_FPS,
        'timestamp' : cv2.CAP_PROP_POS_MSEC,
        'frame'     : cv2.CAP_PROP_POS_FRAMES, # TODO the rest
        # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    }

    def __init__(self, filename, *args, start=None, **kwargs):
        ''' Initialise a video reader from the given file. '''
        super().__init__(filename, *args, **kwargs)
        self.filename = filename
        if start is not None:
            self.set()
 
    @property
    def fps(self):
        return self.get('fps')

    @property
    def current_frame(self):
        return self.get('frame')

    @property
    def timestamp(self):
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
        ''' Sets the current timestamp as specified.

        'timestamp' can be a float/integer of milliseconds, or a string
            of 'hours:minutes:seconds', 'minutes:seconds', or 'seconds',
            where all values can be integers or floats.

        '''
        if isinstance(timestamp, str):
            timestamp = sum(60 ** index * period \
                            for index, period in timestamp.split(':'))
        self.set('timestamp', timestamp)


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
        # TODO start thread for repeated grab
        #   cycle with 'retrieve' called on the latest grab when calling
        #   __next__ or self.read

    def __next__(self):
        if self.isOpened():
            return self.retrieve()
        raise StopIteration


class LockedCamera(Camera):
    ''' A camera for asynchronously capturing a single image at a time.

    Like 'Camera' but uses less power+CPU by only capturing a single frame per
    iteration, and allows specifying when the next frame should start being
    captured.

    Images may be less recent than achieved with Camera, depending on when the
    user starts the capture process within their processing pipeline, but can
    also be more recent if started near the end of the pipeline (at the risk of
    having to wait for the capturing to complete).

    '''
    pass # TODO locked thread execution



