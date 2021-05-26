#!/usr/bin/env python3

import mss
import platform
import numpy as np
from pcv.vidIO import ContextualVideoCapture, SlowCamera, Camera, LockedCamera

Mss = mss.mss().__class__ # handle different operating systems
DARWIN = platform.system() == 'Darwin'


class ScreenSource(Mss):
    ''' A class to provide mss as a VideoSource backend. '''
    def __init__(self, monitor, *args, **kwargs):
        '''
        'monitor' can be an integer index, -1 for all monitors together, a
            PIL.grab-style tuple of (left, top, right, bottom), or a dictionary
            with keys 'left', 'top', 'width', and 'height'.
            NOTE: macOS values should be half of the actual screen resolution.

        '''
        self.__image = None
        super().__init__(*args, **kwargs)
        self.open(monitor)

    def get(self, property):
        return getattr(self, property)

    def set(self, property, value):
        return setattr(self, property, value)

    def open(self, monitor, api_preference=None):
        # more efficient to use dictionary form
        if isinstance(monitor, tuple):
            monitor = {
                'left'  : monitor[0],
                'top'   : monitor[1],
                'width' : monitor[2] - monitor[0],
                'height': monitor[3] - monitor[1]
            }
        elif isinstance(monitor, int):
            monitor = self.monitors[monitor+1]

        scale = 2 if DARWIN else 1 # handle macOS pixel doubling
        self.width = monitor['width'] * scale
        self.height = monitor['height'] * scale
        self._monitor = monitor
        self._open = True

    def isOpened(self):
        return self._open

    def grab(self):
        try:
            self.__image = super().grab(self._monitor)
            return True
        except mss.exception.ScreenShotError:
            self.__image = None
            return False

    def retrieve(self, image=None, *args):
        if self.__image is None:
            return False, None

        if image is not None:
            image[:] = np.array(self.__image)
        else:
            image = np.array(self.__image)

        return True, image

    def read(self, image=None):
        self.grab()
        return self.retrieve(image)

    def release(self):
        self.close()
        self._open = False


class ScreenWrap:
    ''' A VideoSource mixin to use ScreenSource instead of OpenCVSource. '''
    def __init__(self, monitor, *args, **kwargs):
        super().__init__(monitor, *args, source=ScreenSource, **kwargs)

    def __repr__(self):
        monitor = self._monitor
        return f'{self.__class__.__name__}({monitor=})'


class SlowScreen(ScreenWrap, SlowCamera): pass
class Screen(ScreenWrap, Camera): pass
class LockedScreen(ScreenWrap, LockedCamera): pass


if __name__ == '__main__':
    import cv2
    with LockedScreen(0, process=lambda img: \
                      cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)) as screen:
        # video-recording requires 3-channel (BGR) or single-channel
        #  (greyscale, isColor=False) to work
        screen.record_stream('screen-record.mp4')
