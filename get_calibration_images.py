from pypylon import pylon
import numpy as np
import cv2

def main():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessW = 10
    chessH = 7

    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # 2. Initialize UI Window and Trackbars
    win_name = 'Drill 3D Pose Estimation'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    numImages = 0
    objp = np.zeros((chessW*chessH,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessH,0:chessW].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    while camera.IsGrabbing():
        camera.ExposureTime.SetValue(100000)
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            color_image = image.GetArray()
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            cv2.imshow(win_name, gray)
            ret, corners = cv2.findChessboardCorners(gray, (chessW,chessH), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(gray, (chessW,chessH), corners2, ret)
                cv2.imshow(win_name, gray)

            key = cv2.waitKey(1) & 0xFF
            if (key == ord('s')):
                cv2.imwrite(f"C:/Users/ninig/JHU_Work/lcsr/calibration_current_iteration/chessboard{numImages}.png", color_image)
                numImages += 1;
            if (key == ord('q')):
                grabResult.Release()
        
                # Releasing the resource    
                camera.StopGrabbing()
                cv2.destroyAllWindows()

main()