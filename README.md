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
Implemented functionality is provided in the `pcv` module, providing classes such
as `Camera`, `CamCorder`, and `VideoReader`. All other opencv functionality should
be accessed through the standard `cv2` import.
