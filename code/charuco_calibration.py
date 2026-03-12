import cv2
import numpy as np
import json
import glob

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # Dictionary ID
SQUARES_VERTICALLY = 5               # Number of squares vertically
SQUARES_HORIZONTALLY = 7             # Number of squares horizontally
SQUARE_LENGTH = 30                   # Square side length (in pixels)
MARKER_LENGTH = 24                   # ArUco marker side length (in pixels)
MARGIN_PX = 0                       # Margins size (in pixels)
OUTPUT_JSON = "./jank_calibration.json"


# TODO: test out charuco calibration to see if rms is better

def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    
    # Load images from directory
    images = glob.glob(img_dir)
    all_charuco_ids = []
    all_charuco_corners = []

    imgOne = cv2.imread(images[0])
    shape = np.empty((1, 2))
    if (imgOne is not None):
        shape = imgOne.shape[:2]
    else:
        return (None, None)
    
    # Loop over images and extraction of corners
    for image_file in images:
        image = cv2.imread(image_file)
        if (image is None):
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_copy = image.copy()
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
        
        if marker_ids is not None and len(marker_ids) > 0: # If at least one marker is detected
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board) 

            if charucoIds is not None and len(charucoCorners) > 3:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)
    
    # Calibrate camera with extracted information
    mtx = np.zeros((3, 3))
    dist = np.zeros((4, 1))
    result, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, shape, mtx, dist, None, None)
    print(f"Error: {result}")
    return mtx, dist

mtx, dist = get_calibration_parameters(img_dir='./charuco_images/charuco*.png')
if (mtx is not None and dist is not None):
    data = {"mtx": mtx.tolist(), "dist": dist.tolist()}

    with open(OUTPUT_JSON, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f'Data has been saved to {OUTPUT_JSON}')