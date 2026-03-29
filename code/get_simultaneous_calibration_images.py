import pypylon.pylon as py
from charuco_calibration import *
import numpy as np
import cv2
import json

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # Dictionary ID
SQUARES_VERTICALLY = 5               # Number of squares vertically
SQUARES_HORIZONTALLY = 7             # Number of squares horizontally
SQUARE_LENGTH = 30                   # Square side length (in pixels)
MARKER_LENGTH = 24                   # ArUco marker side length (in pixels)
MARGIN_PX = 0                       # Margins size (in pixels)


def getCalibrationImages(filepath1, filepath2):
    CHECKERBOARD = (4,5)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params) 

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

                win_name = 'Window 1'
                if (cam_id == 1):
                    win_name = 'Window 2'

                # Access the image data
                image = converter.Convert(res)
                color_image = image.GetArray()
                
                # Undistort the frame with our intrinsic camera parameters
                if (cam_id == 0):
                    _, color_image = undistort_image(color_image, cameraMatrix745, distCoeffs746)
                else:
                    _, color_image = undistort_image(color_image, cameraMatrix746, distCoeffs745)

                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        
                if marker_ids is not None and len(marker_ids) > 1: # If at least two markers are detected
                    _, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, color_image, board)

                    if charucoIds is not None and len(charucoCorners) > 10:
                        img_copy = color_image.copy()
                        cv2.aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)
                        cv2.drawChessboardCorners(color_image, CHECKERBOARD, charucoCorners, True)

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