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
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessW = 10
    chessH = 7

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessW*chessH,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessH,0:chessW].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cameraMatrix = np.empty((3,3))
    dist_coeffs = np.empty((1,5))

    images = glob.glob('C:/Users/ninig/JHU_Work/lcsr/calibration_current_iteration/chessboard10x7/chessboard*.png')
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
        ret, corners = cv.findChessboardCorners(gray, (chessW,chessH), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (chessW,chessH), corners2, ret)
            # cv.imshow('window', img)
            # cv.waitKey(500)
        else:
            print(f"Corners not found on {fname}")
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, cameraMatrix, dist_coeffs)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )
    cv.destroyAllWindows()
    return (cameraMatrix, dist_coeffs)

def showUndistorted(cameraMatrix, dist_coeffs, img, testName):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist_coeffs, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, cameraMatrix, dist_coeffs, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f"C:\\Users\\ninig\\JHU_Work\\lcsr\\calibration_current_iteration\\{testName}orig.png", img)
    cv.imwrite(f"C:\\Users\\ninig\\JHU_Work\\lcsr\\calibration_current_iteration\\{testName}test.png", dst)

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