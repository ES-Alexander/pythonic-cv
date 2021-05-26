class VideoSource:
    ''' A generic video source with swappable back-end. '''
    def __init__(self, source, *args, **kwargs):
        self.__source = source(*args, **kwargs)

    def get(self, property):
        ''' Return the value of 'property' if it exists, else 0.0. '''
        return self.__source.get(property)

    def set(self, property, value):
        ''' Attempts to set 'property' to 'value', returning success. '''
        return self.__source.set(property, value)

    def read(self, image=None):
        ''' Returns success, frame of reading the next frame.

        'image' an optional array to store the output frame in.

        '''
        return self.__source.read(image)

    def __getattr__(self, key):
        ''' On failure to find an attribute in this instance, check source. '''
        return getattr(self.__source, key)
