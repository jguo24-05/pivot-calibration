import pypylon.pylon as py
import numpy as np
import cv2

def main():
    NUM_CAMERAS = 2
    tlf = py.TlFactory.GetInstance()
    di = py.DeviceInfo()
    devs = [di,]
    tlf.EnumerateDevices(devs)
    cam_array = py.InstantCameraArray(NUM_CAMERAS)
    
    for idx, cam in enumerate(cam_array):
        cam.Attach(tlf.CreateDevice(devs[idx]))
    cam_array.Open()

    # Set the exposure time for each camera and store a unique 
    # number for each camera to identify the incoming images
    _, cam1 = cam_array[0]
    cam1.ExposureTimeRaw = 5000
    cam1.SetCameraContext(0)
    _, cam2 = cam_array[1]
    cam2.ExposureTimeRaw = 10000
    cam2.SetCameraContext(1)
    
    # store last framecount in array
    frame_counts = [0]*NUM_CAMERAS

    converter = py.ImageFormatConverter()
    # converting to opencv bgr format
    converter.OutputPixelFormat = py.PixelType_BGR8packed
    converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

    cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 1", 500, 350)

    cv2.namedWindow("Window 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 2", 800, 650)

    cam_array.StartGrabbing()
    while True:
        with cam_array.RetrieveResult(1000) as res:
            if res.GrabSucceeded():
                img_nr = res.ImageNumber
                cam_id = res.GetCameraContext()
                frame_counts[cam_id] = img_nr
                grabResult = res.GetArray()

                # Access the image data
                image = converter.Convert(grabResult)
                color_image = image.GetArray()
                
                # do something with the image ....
                if (cam_id == 0):
                    cv2.imshow('Window 1', color_image)
                    
                elif (cam_id == 1):
                    cv2.imshow('Window 2', color_image)
                
                # check if q button has been pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    cam_array.StopGrabbing()
    cam_array.Close()

main()