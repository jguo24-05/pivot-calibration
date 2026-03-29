from pypylon import pylon
import numpy as np
import cv2


# TODO: improve image quality and coverage, speed this process up if possible 

def main():
    chessW = 10
    chessH = 7
    # Important for these flags to match the ones in chessboard_calibration.py
    calib_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE   

    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabbing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # 2. Initialize UI Window and Trackbars
    win_name = 'Drill 3D Pose Estimation'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    numImages = 0

    while camera.IsGrabbing():
        camera.ExposureTime.SetValue(5000)
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            color_image = image.GetArray()
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            
            ret, corners = cv2.findChessboardCorners(gray, (chessW,chessH), flags = calib_flags)

            # If found, add object points, image points (after refining them)
            if ret == True:
                cv2.imwrite(f'./chessboard{numImages}.png', color_image)
                numImages += 1

                # Draw and display the corners
                cv2.drawChessboardCorners(color_image, (chessW,chessH), corners, ret)
                cv2.imshow(win_name, color_image)
            else:
                cv2.imshow(win_name, color_image)
                

            key = cv2.waitKey(1) & 0xFF
            if (key == ord('q')):
                grabResult.Release()
        
                # Releasing the resource    
                camera.StopGrabbing()
                cv2.destroyAllWindows()

main()