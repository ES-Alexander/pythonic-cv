#!/usr/bin/env python

from pcv.vidIO import VideoReader, OutOfFrames
import cv2

class VideoSwitcher:
    ''' A class for switching between multiple videos. '''
    def __init__(self, *filenames, track_along=True):
        ''' Opens multiple videos for switching between.

        When iterating, if a video completes it is re-started.

        'track_along' is a boolean specifying if all the videos track along
            together. Defaults to True, so when swapping to the next video it
            gets jumped ahead to the number of frames that were covered since
            it was last active. If set to False, swapping back to a video
            resumes where it left off.

        '''
        self.filenames   = filenames
        self.track_along = track_along
        self.readers     = [self._initialise(video) for video in filenames]
        self._counts     = [0 for video in filenames] # position of each video
        self._count      = 0 # total frames processed
        self.active      = 0 # current active video index

    def _initialise(self, video):
        ''' Initialise a video and its iterator. '''
        reader = VideoReader(video, destroy=None, verbose=False)
        iter(reader)
        return reader

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        ''' Set a new reader as active. Track frames as required. '''
        self._active = value
        new_reader = self.readers[self._active]
        if self.track_along:
            extra = self._count - self._counts[self._active]
            new_frame = (new_reader.frame + extra) % new_reader._end
            new_reader.set_frame(new_frame)
        new_reader.reset_delay() # avoid pause jump/negative delay

    @property
    def video(self):
        return self.readers[self.active]

    @video.setter
    def video(self, replacement):
        self.readers[self.active] = replacement

    def __enter__(self):
        ''' Ensure all VideoReaders are '''
        for reader in self.readers:
            reader.__enter__()

        return self

    def __exit__(self, *args):
        ''' Clean up all the VideoReaders at context end. '''
        for reader in self.readers:
            try:
                reader.__exit__(*args)
            except:
                continue

    def __iter__(self):
        return self

    def __next__(self):
        ''' Get the next video frame from the active video.

        Restarts the active video first if it was finished.

        '''
        self._count += 1
        self._counts[self.active] += 1
        try:
            return next(self.video)
        except OutOfFrames:
            return self._reinitialise_active_video()

    def _reinitialise_active_video(self):
        ''' Close and re-open the currently active video. '''
        self.video.__exit__(None, None, None)
        filename = self.filenames[self.active]
        self.video = self._initialise(filename)
        return next(self.video)

    def next_video(self):
        ''' Switch to the next video (circular buffer). '''
        self.active = (self.active + 1) % len(self.readers)

    def prev_video(self):
        ''' Switch to the previous video (circular buffer). '''
        self.active = (self.active - 1) % len(self.readers)

    def set_active(self, index):
        ''' Switch to a specified video index, modulo the number of videos. '''
        self.active = index % len(self.readers)


if __name__ == '__main__':
    from os import listdir
    from pcv.vidIO import Camera

    class CoverAnalyser:
        ''' Analyses a camera feed and switches videos when un/covered. '''
        def __init__(self, video_switcher):
            ''' Cycles through 'video_switcher' videos based on frame data. '''
            self._video_switcher = video_switcher
            self._covered = False

        def analyse(self, frame):
            ''' Switches to the next video if the camera is un/covered.

            Switches on both camera cover and un-cover events.

            Mean values of 40 and 60 were relevant for my camera when testing,
                and may need to be different depending on camera and lighting.

            '''
            mean = frame.mean()
            if (mean < 40 and not self._covered) \
               or (mean > 50 and self._covered):
                # switch to next video and toggle state
                self._video_switcher.next_video()
                self._covered ^= True # toggle state using boolean xor

    videos = (file for file in listdir() if file.endswith('.mp4'))

    print("press space to pause, 'q' to quit")
    with VideoSwitcher(*videos) as video_switcher:
        analyser = CoverAnalyser(video_switcher)
        with Camera(0) as cam:
            # zip together the videos and camera to progress in sync
            for (_, v_frame), (_, c_frame) in zip(video_switcher, cam):
                analyser.analyse(c_frame)
                cv2.imshow('video', v_frame)
