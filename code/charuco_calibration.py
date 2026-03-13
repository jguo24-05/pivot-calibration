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

def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementMinAccuracy = 0.01
    params.cornerRefinementMaxIterations = 100

    detector = cv2.aruco.ArucoDetector(dictionary, params)
    
    # Load images from directory
    images = glob.glob(img_dir)
    all_charuco_ids = []
    all_charuco_corners = []

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.'

    imgOne = cv2.imread(images[0])
    shape = np.empty((1, 2))
    if (imgOne is not None):
        shape = imgOne.shape[:2]
    else:
        return (None, None)
    
    mtx = np.zeros((3, 3))
    dist = np.zeros((4, 1))
    
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
            objpts, imgpts = cv2.aruco.getBoardObjectAndImagePoints(board, charucoCorners, charucoIds)

            if charucoIds is not None and len(charucoCorners) > 10:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)
                objpoints.append(objpts)
                imgpoints.append(imgpts)
    
    # Calibrate camera with extracted information
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    rms, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, shape, mtx, dist, flags=calibration_flags,
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    print(f"Error: {rms}")
    return mtx, dist


def write_calibration_parameters(img_dir, OUTPUT_JSON):
    mtx, dist = get_calibration_parameters(img_dir)
    if (mtx is not None and dist is not None):
        data = {"mtx": mtx.tolist(), "dist": dist.tolist()}

        with open(OUTPUT_JSON, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f'Data has been saved to {OUTPUT_JSON}')


def undistort_image(image, mtx, dst):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
    undist = cv2.undistort(image, mtx, dst, None, newcameramtx)
    return (newcameramtx, undist)