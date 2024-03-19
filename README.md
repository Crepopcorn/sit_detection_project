# sit_detection_project
## This project is for my Final Year Project.

### System Requirements

A computer with a minimum of two webcams (for front and side view analysis).

Python environment with PyQt5, OpenCV, MediaPipe, and other dependencies installed as outlined in the initial setup steps.

    Python: version 3.8 - 3.11
    OS: Desktop: Windows, Mac, Linux. IoT: Raspberry OS 64-bit.
    PIP: use latest version
    keras version 2.15.0

    #Install packages if they are not preinstalled on your system.
    pip install PyQt5
    pip install opencv-python
    pip install mediapipe
    pip install ultralytics
    pip install opencv-contrib-python

Camera Setup: Position two webcams such that one captures your front view and the other captures your side view. These will be referred to as "Camera 1 (Front View)" and "Camera 2 (Side View)" within the application.

### Getting Started

1) Connect a secondary camera or webcam to the user's computer, which already has its own built-in camera.

2) The second camera should be positioned on either the left or right side of the user.

3) Download the "main_code" folder to the user's computer.

4) Verify whether the system being used by the user meets the specified requirements. In certain situations, the user may need to either upgrade or downgrade the system in order to align with the requirements.

5) Run the pyqt_main.py code to display the programme.

6) Start the application. You will be greeted by a splash screen followed by the main window of the Sit Posture Recognition System.

main window layout:

![pyqt_main](https://github.com/Crepopcorn/sit_detection_project/assets/112138670/9e7394ba-e553-404c-a713-abebfac0d4e7)

settings dialog layout:

![pyqt_settings](https://github.com/Crepopcorn/sit_detection_project/assets/112138670/e129f578-bf55-4245-bac0-198d19cbc033)

For references, you can check the output video/example_video.mp4 in this github repository to see the demo of this GUI programme.
