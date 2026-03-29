import numpy as np
import cv2
import glob
import json

# Reference: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # Dictionary ID
SQUARES_VERTICALLY = 5               # Number of squares vertically
SQUARES_HORIZONTALLY = 7             # Number of squares horizontally
SQUARE_LENGTH = 100                   # Square side length (in pixels)
MARKER_LENGTH = 70                   # ArUco marker side length (in pixels)
MARGIN_PX = 0                        # Margins size (in pixels)
WORLD_SCALING = 1

def stereo_calibrate(mtx1, dist1, mtx2, dist2, filepath1, filepath2):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementMinAccuracy = 0.01
    params.cornerRefinementMaxIterations = 100
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    c1_images_names = sorted(glob.glob(filepath1))
    c2_images_names = sorted(glob.glob(filepath2))
    
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
 
        marker_corners_745, marker_ids_745, _ = detector.detectMarkers(gray1)
        marker_corners_746, marker_ids_746, _ = detector.detectMarkers(gray2)
            
        if marker_ids_745 is not None and len(marker_ids_745) > 1: # If at least two markers are detected
            _, charuco_corners_745, charuco_ids_745 = cv2.aruco.interpolateCornersCharuco(marker_corners_745, marker_ids_745, gray1, board) 
            
            if charuco_ids is not None and len(charuco_corners) > 10:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

        objpoints.append(objp)
        imgpoints_left.append(corners1)
        imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T



with open(f"./calibration_data/cam745_calibration.json", 'r') as file:
        json_data = json.load(file)

cameraMatrix745 = np.array(json_data['mtx'])
distCoeffs745 = np.array(json_data['dist'])

with open(f"./calibration_data/cam746_calibration.json", 'r') as file:
        json_data = json.load(file)

cameraMatrix746 = np.array(json_data['mtx'])
distCoeffs746 = np.array(json_data['dist'])
    
R, T = stereo_calibrate(cameraMatrix745, distCoeffs745, cameraMatrix746, distCoeffs746, 
                        "./charuco_images/external_745", "./charuco_images/external_746")