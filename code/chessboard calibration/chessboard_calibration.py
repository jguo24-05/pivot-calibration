import numpy as np
import cv2 as cv
import glob
import json

'''
Test Code:

'''

def returnCameraCoeffsNonFisheye(imgfolder):
    # termination criteria
    CHECKERBOARD = (7,10)
    subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    counter = 1

    images = glob.glob(imgfolder)
    imgOne = cv.imread(images[0])
    shape = np.empty((1, 2))
    if (imgOne is not None):
        shape = imgOne.shape[:2]
    else:
        return (None, None)

    for fname in images:
        img = cv.imread(fname)
        
        if (img is None):
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, flags = cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (3,3), (-1,-1), subpix_criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # cv.imshow('window', img)
            # cv.waitKey(500)
            counter += 1
        else:
            print(f"Corners not found on {fname}")
    
    # For non-fisheye calibration: 
    cameraMatrix = np.zeros((3, 3))
    dist_coeffs = np.zeros((4, 1))
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, cameraMatrix, dist_coeffs)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )
    # cv.destroyAllWindows()
    return (cameraMatrix, dist_coeffs)
    

def returnCameraCoeffsFisheye(imgfolder, json_name):
    # termination criteria
    CHECKERBOARD = (7,10)
    subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_CHECK_COND + cv.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # prepare object points, (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    square_size = 6.0 / 8.0     # square side length in millimeters
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    counter = 1

    images = glob.glob(imgfolder)
    imgOne = cv.imread(images[0])
    shape = np.empty((1, 2))
    if (imgOne is not None):
        shape = imgOne.shape[:2]
    else:
        return (None, None)

    for fname in images:
        img = cv.imread(fname)
        
        if (img is None):
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);

        # Find the chess board corners
        calib_flags = cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE+cv.CALIB_CB_FAST_CHECK
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, flags = calib_flags)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (3,3), (-1,-1), subpix_criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            
            key = cv.waitKey(1) & 0xFF
            if (key == ord('q')):
                break
            
            counter += 1
        else:
            print(f"Corners not found on {fname}")

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(counter)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(counter)]
    rms, K, D, rvecs, tvecs = cv.fisheye.calibrate(
        objpoints,
        imgpoints,
        shape,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    
    # Save intrinsic parameters to a json file
    OUTPUT_JSON = json_name
    data = {"mtx": K.tolist(), 
            "dist": D.tolist(), 
            "objpoints": [o.tolist() for o in objpoints], 
            "imgpoints": [i.tolist() for i in imgpoints]}
    with open(OUTPUT_JSON, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f'Data has been saved to {OUTPUT_JSON}')
    print( f"total error: {rms}" )
    return (K, D)


def undistortedNonFisheye(K, D, img):
    h,  w = img.shape[:2]
    newcam_mtx, roi=cv.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    dst = cv.undistort(img, K, D, None, newcam_mtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    '''
    cv.imshow('window', img)
    cv.waitKey(0)
    cv.imshow('window', dst)
    cv.waitKey(0)
    '''
    return dst


def undistortFisheye(K, D, img):
    new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, img.shape[:2], np.eye(3), balance=1.0)
    undistorted_img = cv.fisheye.undistortImage(img, K, D, Knew = new_K)
    '''
    cv.imshow('window', img)
    cv.waitKey(0)
    cv.imshow('window', undistorted_img)
    cv.waitKey(0)
    '''
    return (new_K, undistorted_img)


# Functions for solving for the 3D coordinates - not sure of these
def findRTvecs(objpoints, imgpoints, new_K, D):
    _, rvec, tvec=cv.solvePnP(objpoints,imgpoints,new_K,D)    # TODO: understand these parameters
    R_mtx, _=cv.Rodrigues(rvec)
    return (R_mtx, tvec)


def calcWorldCoordinates(u, v, new_K, R_mtx, tvec):
    cameraVec = np.array([[u,v,1]], dtype=np.float32).T
    R_inv = np.linalg.inv(R_mtx)
    K_inv = np.linalg.inv(new_K)
    return np.dot(R_inv, (np.dot(K_inv, cameraVec)-tvec))


# For accessing the preexisting calibration parameters
json_file_path = './adjacent_camera_calibration.json'
with open(json_file_path, 'r') as file: # Read the JSON file
    json_data = json.load(file)

cameraMatrix = np.array(json_data['mtx'])
distCoeffs = np.array(json_data['dist'])


# TODO: grab better coverage for calibration images
# K,  D = returnCameraCoeffsFisheye('./mostRecent_opposite/chessboard*.png', './opposite_cam_calibration.json')
# returnCameraCoeffsFisheye('./mostRecent_opposite/chessboard*.png', './opposite_cam_calibration.json')
# returnCameraCoeffsFisheye('./adjacent_camera/chessboard*.png', './adjacent_cam_calibration.json')

images = glob.glob('./charuco_images/adjacent_camera/charuco*.png')

for i in range(len(images)):
    img = cv.imread(images[i])
    newCamMat = np.array((3,3))
    if (img is not None):
        image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        h,  w = image.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))
        undist = cv.undistort(image, cameraMatrix, distCoeffs, None, newcameramtx)

        cv.imshow('window', undist)
        cv.waitKey(500)

    key = cv.waitKey(1) & 0xFF
    if (key == ord('q')):
        cv.destroyWindow('window')
        break 