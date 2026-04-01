import cv2
import pypylon.pylon as py
import json
import numpy as np

# TODO: finish this if we ever want to speed up the data collection process

def getCalibrationVideo():
    NUM_CAMERAS = 2
    tlf = py.TlFactory.GetInstance()
    
    devs = tlf.EnumerateDevices()
    cam_array = py.InstantCameraArray(NUM_CAMERAS)
    
    for idx, cam in enumerate(cam_array):
        cam.Attach(tlf.CreateDevice(devs[idx]))
    cam_array.Open()

    # Set the exposure time for each camera and store a unique 
    # number for each camera to identify the incoming images
    cam1 = cam_array[0]
    cam1.ExposureTime.SetValue(300000)
    cam1.SetCameraContext(0)
    cam2 = cam_array[1]
    cam2.ExposureTime.SetValue(300000)
    cam2.SetCameraContext(1)

    # store last framecount in array
    frame_counts = [0]*NUM_CAMERAS

    converter = py.ImageFormatConverter()
    # converting to opencv bgr format
    converter.OutputPixelFormat = py.PixelType_BGR8packed
    converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

    # TODO: add two video writers here
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

    cam_array.StartGrabbing()

    while True:
        with cam_array.RetrieveResult(5000) as res:
            if res.GrabSucceeded():

                # TODO: out.write(frame)        # Save frame to new video

    cam_array.StopGrabbing()
    cam_array.Close()

    


