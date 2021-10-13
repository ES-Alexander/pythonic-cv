_________________________________
 Version: 1.2.2                  
 Author: ES Alexander            
 Release Date: 26/May/2021       
_________________________________

# About
OpenCV is a fantastic tool for computer vision, with significant Python support
through automatically generated bindings. Unfortunately some basic functionality
is frustrating to use, and documentation is sparse and fragmented as to how best to
approach even simple tasks such as efficiently processing a webcam feed.

This library aims to address frustrations in the OpenCV Python api that can be
fixed using pythonic constructs and methodologies. Solutions are not guaranteed to
be optimal, but every effort has been made to make the code as performant as
possible while ensuring ease of use and helpful errors/documentation.

# Requirements
This library requires an existing version of `OpenCV` with Python bindings to be 
installed (e.g. `python3 -m pip install opencv-python`). Some features (mainly
property access helpers) may not work for versions of OpenCV earlier than 4.2.0. 
The library was tested using Python 3.8.5, and is expected to work down to at least
Python 3.4 (although the integrated advanced features example uses matmul (@) for
some processing, which was introduced in Python 3.5).

`Numpy` is also used throughout, so a recent version is suggested (tested with 1.19.0).
`mss` is used for cross-platform screen-capturing functionality (requires >= 3.0.0)

# Installation
The library can be installed from pip, with `python3 -m pip install pythonic-cv`.

# Usage
New functionality is provided in the `pcv` module, as described below. All other 
opencv functionality should be accessed through the standard `cv2` import.

## Main Functionality
The main implemented functionality is handling video through a context manager, 
while also enabling iteration over the input stream. While iterating, key-bindings
have been set up for play/pause (`SPACE`) and stopping playback (`q`). A dictionary
of pause_effects can be passed in to add additional key-bindings while paused without
needing to create a subclass. In addition, video playback can be sped up with `w`,
slowed down with `s`, and if enabled allows rewinding with `a` and returning to
forwards playback with `d`. Forwards playback at 1x speed can be restored with `r`.
While paused, video can be stepped backwards and forwards using `a` and `d`. All
default key-bindings can be overwritten using the play_commands and pause_effects
dictionaries and the quit and play_pause variables on initialisation.

## Video I/O Details
For reading and writing video files, the `VideoReader` and `VideoWriter` classes should 
be used. For streaming, the classes `Camera`, `SlowCamera`, and `LockedCamera` are 
provided. The simplest of these is `SlowCamera`, which has slow iteration because image
grabbing is performed synchronously, with a blocking call while reading each frame. 
`Camera` extends `SlowCamera` with additional logic to perform repeated grabbing in a
separate thread, so processing and image grabbing can occur concurrently. `LockedCamera`
sits between the two, providing thread based I/O but with more control over when each 
image is taken. Analogues for all three camera classes are provided in the `pcv.screen`
module - `SlowScreen`, `Screen`, and `LockedScreen`. The interface is the same as that
for the corresponding camera class, except that a monitor index (-1 for all) or screen 
region is passed in instead of a camera id.

`Camera` is most useful for applications with processing speeds that require the most
up to date information possible and don't want to waste time decoding frames that are
grabbed too early to be processed (frame grabbing occurs in a separate thread, and only
the latest frame is retrieved (decoded) on read). `SlowCamera` should only be used where
power consumption or overall CPU usage are more important than fast processing, or in
hardware that is only capable of single-thread execution, in which case the
separate image-grabbing thread will only serve to slow things down.

`LockedCamera` is intended to work asynchronously like `Camera`, but with more control.
It allows the user to specify when the next image should be taken, which leads to less
wasted CPU and power usage on grabbing frames that aren't used, but with time for the 
image to be grabbed and decoded before the next iteration needs to start. The locking 
protocol adds a small amount of additional syntax, and starting the image
grabbing process too late in an iteration can result in waits similar to those in
`SlowCamera`, while starting the process too early can result in images being somewhat
out of date. Tuning can be done using the 'preprocess' and 'process' keyword arguments,
with an in-depth usage example provided in `something_fishy.py`. When used correctly
`LockedCamera` has the fastest iteration times, or if delays are used to slow down the
process it can have CPU and power usage similar to that of `SlowCamera`.

If using a video file to simulate a live camera stream, use `SlowCamera` or 
`LockedCamera` - `Camera` will skip frames.

There is also a `GuaranteedVideoWriter` class which guarantees the output framerate by
repeating frames when given input too slowly, and skipping frames when input is too
fast.

## Overview
![Overview of classes diagram](https://github.com/ES-Alexander/pythonic-cv/blob/master/Overview.png)

## Examples
### Basic Camera Stream
```python
from pcv.vidIO import Camera
from pcv.process import channel_options, downsize

# start streaming camera 0 (generally laptop webcam/primary camera), and destroy 'frame'
#   window (default streaming window) when finished.
# Auto-initialised to have 1ms waitKey between iterations, breaking on 'q' key-press,
#   and play/pause using the spacebar.
with Camera(0) as cam:
    cam.stream()

# stream camera 0 on window 'channels', downsized and showing all available channels.
with LockedCamera(0, display='channels', 
                  process=lambda img: channel_options(downsize(img, 4))) as cam:
    cam.stream()
```

### Stream and Record
```python
from pcv.vidIO import Camera

with Camera(0) as cam:
    print("press 'q' to quit and stop recording.")
    cam.record_stream('me.mp4')
```

### Screen Functionality
#### Main Monitor
```python
import cv2
from pcv.screen import LockedScreen

# stream monitor 0, and record to 'screen-record.mp4' file.
with LockedScreen(0, process=lambda img: \
                  cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) as screen:
    # video-recording requires 3-channel (BGR) or single-channel
    #  (greyscale, isColor=False) to work
    screen.record_stream('screen-record.mp4')
```
#### Screen Region
```python
from pcv.Screen import Screen

with Screen({'left': -10, 'top': 50, 'width': 100, 'height': 200}) as screen:
    screen.stream()
```

### VideoReader
```python
from pcv.vidIO import VideoReader
from pcv.process import downsize

# just play (simple)
# Press 'b' to jump playback back to the beginning (only works if pressed
#   before playback is finished)
with VideoReader('my_vid.mp4') as vid:
    vid.stream()
    
# start 15 seconds in, end at 1:32, downsize the video by a factor of 4
with VideoReader('my_vid.mp4', start='15', end='1:32', 
                 preprocess=lambda img: downsize(img, 4)) as vid:
    vid.stream()
    
# enable rewinding and super fast playback
# Press 'a' to rewind, 'd' to go forwards, 'w' to speed up, 's' to slow down
#    and 'r' to reset to forwards at 1x speed.
with VideoReader('my_vid.mp4', skip_frames=0) as vid:
    vid.stream()
    
# headless mode (no display), operating on every 10th frame
with VideoReader('my_vid.mp4', auto_delay=False, skip_frames=10,
                 process=my_processing_func) as vid:
    vid.headless_stream()
```

### Mouse Events
```python
# Example courtesy of @MatusGasparik in Issue#15
import cv2
from pcv.vidIO import VideoReader
from pcv.interact import MouseCallback

def on_mouse_click(event, x, y, flags, params=None):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Click', x, y)

# create a window beforehand so a mouse callback can be assigned [REQUIRED]
window = 'foo'
cv2.namedWindow(window)
with VideoReader('my_vid.mp4', display=window) as vid:
    vid.stream(mouse_handler=MouseCallback(vid.display, on_mouse_click))
```

### Advanced Examples
Check the `pcv/examples` folder for some examples of full programs using this library.

#### Something Fishy
This example is a relatively basic augmented reality example, which creates a tank of
fish that swim around on top of a video/webbcam feed. You can catch the fish (click and
drag a 'net' over them with your mouse), or tickle them (move around in your webcam feed).
There are several generally useful processing techniques included, so take a look 
through the code and find the functionality that's most interesting to you to explore
and modify.

To run the example use `python3 -m pcv.examples.something_fishy`, or optionally specify
the path to a newline separated text file of names (of your friends and family for 
example), and run with `python3 -m pcv.examples.something_fishy path/to/names.txt`.

#### Video Switcher
This was made in response to a question from `u/guillerubio` on reddit, to show how to
efficiently read multiple videos simultaneously while only displaying one, and switching
between videos based on what's happening in a webcam feed. 

The current video is switched out when the camera is covered/uncovered. Switching is
performed intelligently by only actually reading frames from a single 'active' video
at a time. The `VideoSwitcher` class allows tracking along all the videos simultaneously
based on how many frames have occurred since they were last active, as well as just
resuming from where each one left off when it was last active. When a video ends it gets
started again (in an infinite loop).

To run the example use `python3 -m pcv.examples.cam_video_switch` while in a directory
with the `.mp4` files you want to switch between.

#### Hand Writer
This uses Google's `mediapipe` library for hand detection, to show a relatively simple
AR workflow, where a pointed right hand index finger can be used to write/draw on a
camera feed. Once you've installed the library (`python3 -m pip install mediapipe`),
run with `python3 -m pcv.examples.hand_write` and start drawing.

The pointing detection algorithm is quite rudimentary, so may require some angling of
your hand before it detects that the pointer finger is straight and the other fingers
are closed. It currently intentionally only detects right hands, so your left hand can
also be in the frame without causing issues. The side detection assumes a front-facing
(selfie) camera, as is commonly the case with webcams and similar.
