from pypylon import pylon
import cv2

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # Dictionary ID
SQUARES_VERTICALLY = 5               # Number of squares vertically
SQUARES_HORIZONTALLY = 7             # Number of squares horizontally
SQUARE_LENGTH = 30                   # Square side length (in pixels)
MARKER_LENGTH = 24                   # ArUco marker side length (in pixels)
MARGIN_PX = 0                       # Margins size (in pixels)


def getCalibrationImages(filepath):
    CHECKERBOARD = (4,5)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params) 

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
        camera.ExposureTime.SetValue(30000) # Set exposure here
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            color_image = image.GetArray()
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        
            if marker_ids is not None and len(marker_ids) > 1: # If at least two markers are detected
                _, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, color_image, board)

                if charucoIds is not None and len(charucoCorners) > 10:
                    img_copy = color_image.copy()
                    cv2.aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)
                    cv2.drawChessboardCorners(color_image, CHECKERBOARD, charucoCorners, True)
                    key = cv2.waitKey(1) & 0xFF
                    if (key == ord('s')):
                        cv2.imwrite(f'{filepath}/charuco{numImages}.png', img_copy)
                        numImages += 1
                       
            cv2.imshow(win_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if (key == ord('q')):
                grabResult.Release()
        
                # Releasing the resource    
                camera.StopGrabbing()
                cv2.destroyAllWindows()

getCalibrationImages(f'./charuco_images/746dump')