import cv2
import numpy as np
import json
import glob

# TODO: (before project end)
# Improve the reprojection error. 1.23 currently.
# Calibration images are pretty good currently, but some images have soft spots.
# See charuco_images folder/*dump.

# TODO: measure the length of a square

'''
to calibrate, call:

mtx, dst = write_calibration_parameters('./charuco_images/745dump/charuco*.png', './calibration_data/cam745_calibration.json')
images = glob.glob('./charuco_images/745dump/charuco*.png')
for image_file in images:
        image = cv2.imread(image_file)
        if (image is not None):
            new_K, _ = undistort_image(image, mtx, dst)
'''

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # Dictionary ID
SQUARES_VERTICALLY = 5               # Number of squares vertically
SQUARES_HORIZONTALLY = 7             # Number of squares horizontally
SQUARE_LENGTH = 1                    # Square side length (in arbitrary unit u)
MARKER_LENGTH = 0.7                  # ArUco marker side length (in u)
MARGIN_PX = 0                        # Margins size (in u)

# TODO: try with newer API calls? (commented out)

def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementMinAccuracy = 0.01
    params.cornerRefinementMaxIterations = 100

    detector = cv2.aruco.ArucoDetector(dictionary, params)
    # detector = cv2.aruco.CharucoDetector(board)
    
    # Load images from directory
    images = glob.glob(img_dir)
    all_charuco_ids = []
    all_charuco_corners = []

    imgOne = cv2.imread(images[0])
    shape = np.empty((1, 2))
    if (imgOne is not None):
        h, w = imgOne.shape[:2]
        shape = (w, h)
    else:
        return (None, None, None, None)
    
    mtx = np.zeros((3, 3))
    dst = np.zeros((4, 1))
    
    # Loop over images and extraction of corners
    for image_file in images:
        image = cv2.imread(image_file)
        if (image is None):
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_copy = image.copy()
        # charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
        marker_corners, marker_ids, _ = detector.detectMarkers(image)
        
        if marker_ids is not None and len(marker_ids) > 1: # If at least two markers are detected
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board) 
            
            if charuco_ids is not None and len(charuco_corners) > 10:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
    
    # Calibrate camera with extracted information
    flags = cv2.CALIB_RATIONAL_MODEL
    result, mtx, dst, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners, 
        charucoIds=all_charuco_ids, 
        board=board, 
        imageSize=shape, 
        cameraMatrix=mtx, 
        distCoeffs=dst,
        flags=flags)
    newcam_mtx, _=cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
    
    print(f"Error: {result}")
    return (mtx, dst, rvecs, tvecs, newcam_mtx)


def write_calibration_parameters(img_dir, OUTPUT_JSON):
    (mtx, dist, rvecs, tvecs, newcam_mtx) = get_calibration_parameters(img_dir) # type: ignore
    if (mtx is not None and dist is not None):
        data = {"mtx": mtx.tolist(), 
                "dist": dist.tolist(),
                "newcam_mtx":newcam_mtx.tolist()}

        with open(OUTPUT_JSON, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f'Data has been saved to {OUTPUT_JSON}')
    return mtx, dist


def undistort_image(img, mtx, dst):
    h,  w = img.shape[:2]
    newcam_mtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
    undist = cv2.undistort(img, mtx, dst, None, newcam_mtx)

    # crop the image
    x, y, w, h = roi
    undist = undist[y:y+h, x:x+w]
    '''
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.imshow('window', undist)
    cv2.waitKey(0)
    '''
    return newcam_mtx, undist