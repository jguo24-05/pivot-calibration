import cv2
import numpy as np
import json

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # Dictionary ID
SQUARES_VERTICALLY = 5               # Number of squares vertically
SQUARES_HORIZONTALLY = 7             # Number of squares horizontally
SQUARE_LENGTH = 30                   # Square side length (in pixels)
MARKER_LENGTH = 24                   # ArUco marker side length (in pixels)
MARGIN_PX = 0                       # Margins size (in pixels)
json_file_path = './jank_calibration.json'

with open(json_file_path, 'r') as file: # Read the JSON file
    json_data = json.load(file)

mtx = np.array(json_data['mtx'])
dst = np.array(json_data['dist'])

image_path = './charuco_images/charuco1.png'
image = cv2.imread(image_path)
if (image is not None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
    image = cv2.undistort(image, mtx, dst, None, newcameramtx)

    all_charuco_ids = []
    all_charuco_corners = []

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
    if marker_ids is not None and len(marker_ids) > 0: # If at least one marker is detected
        # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
        ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
        if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
            all_charuco_corners.append(charucoCorners)
            all_charuco_ids.append(charucoIds)

        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(np.array(all_charuco_corners)[0], np.array(all_charuco_ids)[0], board, np.array(mtx), np.array(dst), np.empty(1), np.empty(1))

        Zx, Zy, Zz = tvec[0][0], tvec[1][0], tvec[2][0]
        fx, fy = mtx[0][0], mtx[1][1]

        print(f'Zz = {Zz}\nfx = {fx}')