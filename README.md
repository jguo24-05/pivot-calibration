Calibrating the position of a robot's tool center point with a stereo vision system

1. Run the charuco calibration to find each camera's internal parameters
2. Run the stereo calibration to find the external parameters of the 2 camera system
3. Run find_tcp to solve for the world coordinates of each detected point relative to the camera
