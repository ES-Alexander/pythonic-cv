# About
OpenCV is a fantastic tool for computer vision, with full support in Python
through its automatically generated bindings. Unfortunately some basic functionality
is frustrating to use, and documentation is sparse and fragmented as to how best to
approach even simple tasks such as efficiently processing a webcam feed.  

This library aims to address frustrations in the openCV python api that can be
fixed using pythonic constructs and methodologies. Solutions are not guaranteed to
be optimal, but every effort has been made to make the code as performant as
possible while ensuring ease of use and helpful errors/documentation.

# Usage
New functionality is provided in the `pcv` module, providing classes such as `Camera`,
`SlowCamera`, and `VideoReader`. All other opencv functionality should be accessed
through the standard `cv2` import.  

The main implemented functionality is allowing video input in a context manager, 
while also enabling iteration over the input stream. For reading video files, the
`VideoReader` class should be used. For streaming, `SlowCamera` is a simple camera
stream class, with context management and iteration over each frame. Iteration is slow
because I/O is performed synchronously, with a blocking call while reading each frame.
`Camera` extends `SlowCamera` with additional logic to perform I/O in a separate thread,
and ensure the camera is always retrieving the latest frame (NOT YET IMPLEMENTED).

`Camera` is most useful for applications with processing speeds that require the most
up to date information possible and don't want to waste time decoding frames that are
grabbed too early to be processed (frame grabbing occurs in a separate thread, and only
the latest frame is retrieved (decoded) on read). `SlowCamera` should only be used where
power consumption or overall CPU usage are more important than fast processing, or in
hardware that is only capable of single-thread execution, in which case the
separate image-grabbing thread will only serve to slow things down.

`LockedCamera` is intended to work asynchronously like `Camera`, but with more control.
It allows the user to specify when the next image should be taken, which leads to less
CPU and power usage because only one image is taken per iteration, but the image can
still be grabbed and decoded before the next iteration needs to start. The locking 
protocol adds a small amount of additional complication, and starting the image
grabbing process too late in an iteration can result in waits similar to those in
`SlowCamera`, while starting the process too early can result in images being somewhat
out of date. When used correctly, however, `LockedCamera` has the fastest iteration
times, and CPU and power usage similar to that of `SlowCamera`.

## Examples
TODO
