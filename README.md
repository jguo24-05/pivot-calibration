Calibrating the position of a robot's tool center point with a stereo vision system

1. Run the charuco calibration to find each camera's internal parameters
2. Run the stereo calibration to find the external parameters of the 2 camera system
3. Run find_tcp to solve for the world coordinates of each detected point relative to the camera

Make sure the 745 camera is plugged into the first USB slot, and 746 the second!

TODO: calibrate with ROS
try calibration again with fixed camera and old charuco boards
print out a measuring cube or smth and see if the algo holds up