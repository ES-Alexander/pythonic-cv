#!/usr/bin/env python3

import cv2


waitKey = lambda ms : cv2.waitKey(ms) & 0xFF


class DoNothing:
    ''' A context manager that does nothing. '''
    def __init__(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class MouseCallback:
    ''' A context manager for temporary mouse callbacks. '''
    def __init__(self, window, handler, param=None,
                 restore=lambda *args: None, restore_param=None):
        ''' Initialise with the window, handler, and restoration command.

        'window' is the name of the window to set the callback for.
        'handler' is the function for handling the callback, which should take
            x, y, flags ('&'ed EVENT_FLAG bits), and an optional param passed
            in from the callback handler.
        'param' is any Python object that should get passed to the handler
            on each call - useful for tracking state.
        'restore' is the function to restore as the handler on context exit.
        'restore_param' is the handler param to restore on context exit.

        '''
        self.window        = window
        self.handler       = handler
        self.param         = param
        self.restore       = restore
        self.restore_param = restore_param

    def __enter__(self):
        cv2.setMouseCallback(self.window, self.handler, self.param)
        return self

    def __exit__(self, *args):
        cv2.setMouseCallback(self.window, self.restore, self.restore_param)
