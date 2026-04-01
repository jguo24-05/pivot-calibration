import numpy as np
import cv2
import glob
import json

'''
Calibration Call:
with open(f"./calibration_data/cam745_calibration.json", 'r') as file:
    json_data = json.load(file)

cameraMatrix745 = np.array(json_data['mtx'])
distCoeffs745 = np.array(json_data['dist'])
newmtx745 = np.array(json_data['newcam_mtx'])

with open(f"./calibration_data/cam746_calibration.json", 'r') as file:
    json_data = json.load(file)

cameraMatrix746 = np.array(json_data['mtx'])
distCoeffs746 = np.array(json_data['dist'])
newmtx746 = np.array(json_data['newcam_mtx'])

stereo_calibrate(cameraMatrix745, distCoeffs745, newmtx745, cameraMatrix746, distCoeffs746, newmtx746,
                 "./charuco_images/external_745/charuco*.png", "./charuco_images/external_746/charuco*.png",
                 "./calibration_data/external_parameters.json")
    
'''

# References: 
# https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
# https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/

# criteria used by chessboard pattern detector
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

rows = 7 #number of checkerboard rows.
columns = 10 #number of checkerboard columns.
world_scaling = 1.2 #mm

def stereo_calibrate(mtx1, dist1, newmtx1, mtx2, dist2, newmtx2, filepath1, filepath2, OUTPUT_JSON):
    c1_images_names = sorted(glob.glob(filepath1))
    c2_images_names = sorted(glob.glob(filepath2))
    
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2)
        c2_images.append(_im)
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            # cv2.drawChessboardCorners(frame1, (columns, rows), corners1, c_ret1)
            # cv2.drawChessboardCorners(frame2, (columns, rows), corners2, c_ret2)
            # cv2.waitKey(0)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = 0
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, 
                                                      imgpoints_left, imgpoints_right, 
                                                      mtx1, dist1,
                                                      mtx2, dist2, 
                                                      (width, height), 
                                                      criteria = criteria,
                                                      flags = stereocalibration_flags)
    
    rectL, rectR, projMtxL, projMtxR, Q, roiL, roiR = cv2.stereoRectify(mtx1, dist1,
                                                                        mtx2, dist2,
                                                                        (width, height),
                                                                        R, T)
    
    data = {"R":R.tolist(),
            "T":T.tolist(),
            "745_projection_mtx":projMtxL.tolist(),
            "746_projection_mtx":projMtxR.tolist()}

    with open(OUTPUT_JSON, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print( f"Reprojection error: {ret}" )
    return (projMtxL, projMtxR)


# Calculate world coordinates based on projection matrices 
def grab3DPoints(json_path, projPoints1, projPoints2):
     with open(json_path, 'r') as file:
        json_data = json.load(file)
        projMatr1 = np.array(json_data["745_projection_mtx"])
        projMatr2 = np.array(json_data["746_projection_mtx"])
        homogeneous = cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2)
        return (homogeneous[0], homogeneous[1], homogeneous[2]) / (homogeneous[3])
     

#### Calibration Call ####
with open(f"./calibration_data/cam745_calibration.json", 'r') as file:
    json_data = json.load(file)

cameraMatrix745 = np.array(json_data['mtx'])
distCoeffs745 = np.array(json_data['dist'])
newmtx745 = np.array(json_data['newcam_mtx'])

with open(f"./calibration_data/cam746_calibration.json", 'r') as file:
    json_data = json.load(file)

cameraMatrix746 = np.array(json_data['mtx'])
distCoeffs746 = np.array(json_data['dist'])
newmtx746 = np.array(json_data['newcam_mtx'])


stereo_calibrate(cameraMatrix746, distCoeffs746, newmtx746, cameraMatrix745, distCoeffs745, newmtx745,
                 "./charuco_images/external_746/charuco*.png", "./charuco_images/external_745/charuco*.png",
                 "./calibration_data/external_parameters.json")
    
