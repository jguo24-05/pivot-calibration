import pypylon.pylon as py
from charuco_calibration import *
import numpy as np
import cv2
import json

#criteria used by checkerboard pattern detector.
#Change this if the code can't find the checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
rows = 7 #number of internal corners in a chessboard row
columns = 10 #number of checkerboard columns.
world_scaling = 1. #change this to the real world square size. Or not.
calib_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE   


def getCalibrationImages(filepath1, filepath2):
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

    # Camera Calibration
    with open(f"./calibration_data/cam745_calibration.json", 'r') as file:
        json_data = json.load(file)

    cameraMatrix745 = np.array(json_data['mtx'])
    distCoeffs745 = np.array(json_data['dist'])

    with open(f"./calibration_data/cam746_calibration.json", 'r') as file:
        json_data = json.load(file)

    cameraMatrix746 = np.array(json_data['mtx'])
    distCoeffs746 = np.array(json_data['dist'])
    
    # store last framecount in array
    frame_counts = [0]*NUM_CAMERAS

    converter = py.ImageFormatConverter()
    # converting to opencv bgr format
    converter.OutputPixelFormat = py.PixelType_BGR8packed
    converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

    cam_array.StartGrabbing()

    imageCounter = 0
    currentCam_id = 0

    while True:
        with cam_array.RetrieveResult(5000) as res:
            if res.GrabSucceeded():
                img_nr = res.ImageNumber
                cam_id = res.GetCameraContext()
                frame_counts[cam_id] = img_nr

                # Access the image data
                image = converter.Convert(res)
                color_image = image.GetArray()
                
                # Undistort the frame with our intrinsic camera parameters
                if (cam_id == 0):
                    _, color_image = undistort_image(color_image, cameraMatrix745, distCoeffs746)
                else:
                    _, color_image = undistort_image(color_image, cameraMatrix746, distCoeffs745)

                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                #find the checkerboard
                ret, corners = cv2.findChessboardCorners(gray, (rows, columns), flags = calib_flags)
            
                if ret == True:
                    img_copy = color_image.copy()
                    cv2.drawChessboardCorners(color_image, (rows,columns), corners, ret)

                    key = cv2.waitKey(1) & 0xFF
                    if (currentCam_id == cam_id and key == ord('s')):
                        if (currentCam_id == 0):
                            cv2.imwrite(f'{filepath1}/charuco{imageCounter//2}.png', img_copy)
                            currentCam_id = 1
                        else:
                            cv2.imwrite(f'{filepath2}/charuco{imageCounter//2}.png', img_copy)
                            currentCam_id = 0
                        imageCounter += 1

                if (cam_id == 0):
                    cv2.imshow('Window 1', color_image)
                    
                elif (cam_id == 1):
                    cv2.imshow('Window 2', color_image)
                
                # check if q button has been pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    cam_array.StopGrabbing()
    cam_array.Close()


getCalibrationImages("./charuco_images/external_745", "./charuco_images/external_746")