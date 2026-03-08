import numpy as np
import cv2 as cv
import glob

'''
Test Code:
cameraMatrix, dist = returnCameraCoeffs()
imgPath1 = f"C:\\Users\\ninig\\JHU_Work\\lcsr\\calibration_current_iteration\\chessboard10x7\\chessboard28.png"
imgPath2 = f"C:\\Users\\ninig\\JHU_Work\\lcsr\\calibration_current_iteration\\chessboard10x7\\chessboard44.png"
imgPath3 = f"C:\\Users\\ninig\\JHU_Work\\lcsr\\calibration_current_iteration\\chessboard10x7\\chessboard84.png"

img1 = cv.imread(imgPath1)
img2 = cv.imread(imgPath2)
img3 = cv.imread(imgPath3)

showUndistorted(cameraMatrix, dist, img1, "img1")
showUndistorted(cameraMatrix, dist, img2, "img2")
showUndistorted(cameraMatrix, dist, img3, "img3")
'''

def returnCameraCoeffs():
    # termination criteria
    CHECKERBOARD = (7,10)
    subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_CHECK_COND + cv.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    counter = 1

    images = glob.glob('./chessboard10x7/chessboard*.png')
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
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (3,3), (-1,-1), subpix_criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # cv.imshow('window', img)
            counter += 1
        else:
            print(f"Corners not found on {fname}")
    
    '''
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, cameraMatrix, dist_coeffs)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )
    cv.destroyAllWindows()
    return (cameraMatrix, dist_coeffs)
    '''
    
    print(objpoints)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(counter)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(counter)]
    rms, _, _, _, _ = cv.fisheye.calibrate(
        objpoints,
        imgpoints,
        shape,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    
    mean_error = 0
    for i in range(len(objpoints)): # TODO: figure out what reprojection error means / how it can possibly be below 1
        imgpoints2, _ = cv.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        print(f"imgpoints2.shape: {np.reshape(imgpoints[i], (imgpoints2.shape[1], 1, 2)).shape}")
        print(f"impoints[i].shape: {imgpoints[i].shape}")
        error = cv.norm(np.reshape(imgpoints[i], (imgpoints2.shape[1], 1, 2)), cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )
    return (K, D)


def showUndistorted(K, D, img):
    map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img.shape[:2], cv.CV_16SC2)
    undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    cv.imshow('window', img)
    cv.waitKey(0)
    cv.imshow('window', undistorted_img)
    cv.waitKey(0)

cameraMatrix, dist = returnCameraCoeffs()
print(cameraMatrix, dist)

images = glob.glob('./chessboard10x7/chessboard*.png')

for fname in images:
    img = cv.imread(fname)
    showUndistorted(cameraMatrix, dist, img)