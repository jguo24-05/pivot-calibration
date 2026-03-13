'''
A simple Program for grabbing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import cv2
import numpy as np
from charuco_calibration import *
import math

def nothing(x):
    pass

# Checks if two lines are parallel with the given tolerance
def checkParallel(line1, line2, tolerance):
    x1, y1, x2, y2 = line1[0]
    u1, v1, u2, v2 = line2[0]
    if (x2 - x1 == 0 and u2 - u1 == 0): return True
    elif (x2 - x1 == 0 or u2 - u1 == 0): return False
    return abs(((y2 - y1) / (x2 - x1)) - ((v2 - v1) / (u2 - u1))) < tolerance
    
   
# Checks if point is on the line defined by (endpoint1, endpoint2) with the given tolerance
def pointOnLine(point, endpoint1, endpoint2, tolerance):
    if ((endpoint2[0] - endpoint1[0]) == 0):    # vertical line
        return point[0] == endpoint2[0]
    
    m = (endpoint2[1] - endpoint1[1]) / (endpoint2[0] - endpoint1[0])
    c = endpoint1[1] - m * endpoint1[0]
    expectedY = point[0] * m + c
    return abs(expectedY - point[1]) < tolerance


# Returns the distance between two lines
def distBetweenLines(line1, line2): # assumes parallel lines
    ax1, ax2, ay1, ay2 = line1[0]
    bx1, bx2, by1, by2 = line2[0]
    if (ax2-ax1 == 0):              # vertical line
        return bx1-ax1
    elif (ay2-ay1 == 0):            # horizontal line
        return by1-ay1
    else:
        m = (ay2-ay1) / float((ax2-ax1))
        c1 = ay1 - m*ax1
        c2 = by1 - m*bx1
        return (abs(c2 - c1)) / ((1 + pow(m, 2))**0.5)


# Detect lines using Probabilistic Hough Transform
def detectLines(edges, line_thresh, minLineLength, maxLineGap, minDistBtwnEdges, maxDistBtwnEdges, parallelTolerance):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=line_thresh, minLineLength=minLineLength, maxLineGap=maxLineGap) 

    if lines is not None:
        if (len(lines) < 2):
            return None    # skip this frame if fewer than two lines were found 
    
        for i in range(len(lines)-1):  
            # Debugging 
            # ax1, ay1, ax2, ay2 = lines[i][0]
            # cv2.line(edges, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
                
            for j in range(i, len(lines)):
                line1 = lines[i]
                line2 = lines[j]
                isParallel = checkParallel(line1, line2, parallelTolerance)
                distBtwnEdges = distBetweenLines(line1, line2)

                # Debugging:
                print(f"is parallel: {isParallel}")
                print(f"distBtwnEdges: {distBtwnEdges}")

                if (isParallel and distBtwnEdges > minDistBtwnEdges and distBtwnEdges < maxDistBtwnEdges):
                    return (line1, line2)
        
        return None     # no parallel lines were found this frame


# Detect circles that lie on the given centralAxis
def detectCircles(grayFrame, accumulatorRes, minDist, cannyThreshold, circAccThreshold, minRadius, maxRadius, centralAxis, dispTolerance):
    # Detect circles with the Hough transform
    circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 
                            dp = accumulatorRes, minDist = minDist, param1 = cannyThreshold, 
                            param2 = circAccThreshold, minRadius = minRadius, 
                            maxRadius = maxRadius);

    if not (circles is None):
        circles = np.round(circles).astype("uint16")
        for i in circles[0,:]: 
            center = (i[0], i[1])
            radius = i[2]

            if (pointOnLine(center, centralAxis[0], centralAxis[1], dispTolerance)):
                return (center, radius)

def detectTip(calibration_filepath):
    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabbing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # 1. Calibrate the Camera
    json_file_path = calibration_filepath
    with open(json_file_path, 'r') as file: # Read the JSON file
        json_data = json.load(file)

    cameraMatrix = np.array(json_data['mtx'])
    distCoeffs = np.array(json_data['dist'])
    objpoints = np.array(json_data['objpoints'], dtype=np.float32).reshape(-1, 1, 3)
    imgpoints = np.array(json_data['imgpoints'], dtype=np.float32).reshape(-1, 1, 2)

    # 2. Initialize UI Window and Trackbars
    win_name = 'Drill 3D Pose Estimation'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Initialize Trackbars for Parameter Tuning
    # Canny Threshold for Line Detection
    cv2.createTrackbar('Canny Threshold', win_name, 100, 300, nothing);
    # Accumulator Threshold for Lines
    cv2.createTrackbar('Line Accumulator Threshold', win_name, 100, 300, nothing);
    # Circle Radius Accumulator Threshold
    cv2.createTrackbar('Circle Accumulator Threshold', win_name, 100, 300, nothing)
    # Min Radius: Minimum circle radius to detect
    cv2.createTrackbar('Minimum Radius', win_name, 10, 100, nothing)
    # Max Radius: Maximum circle radius to detect
    cv2.createTrackbar('Maximum Radius', win_name, 200, 500, nothing)
    # Minimum Line Length: Minimum line length to detect
    cv2.createTrackbar('Minimum Line Length', win_name, 10, 100, nothing)
    # Maximum Line Gap: Maximum allowed gap in a line
    cv2.createTrackbar('Maximum Line Gap', win_name, 50, 250, nothing)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Minimum Distance Between Edges', win_name, 5, 100, nothing)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Maximum Distance Between Edges', win_name, 10, 100, nothing)
    # Tolerance for how far the radius can be from the detected central axis
    cv2.createTrackbar('Maximum Error for Tip and Axis Alignment', win_name, 25, 50, nothing)

    while camera.IsGrabbing():
        camera.ExposureTime.SetValue(5000)
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            color_image = image.GetArray()

            # undistort the image
            new_K, color_image = undistort_image(color_image, cameraMatrix, distCoeffs)
            
            # 2. Preprocessing (Grayscale + Gaussian Blur)
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # Blur is crucial to reduce noise/specular highlights on metal surfaces
            blurred = cv2.GaussianBlur(gray, (9, 9), 5)

            # Get current trackbar positions
            cannyThreshold = cv2.getTrackbarPos('Canny Threshold', win_name)    
            line_thresh = cv2.getTrackbarPos('Line Accumulator Threshold', win_name)
            circ_thresh = cv2.getTrackbarPos('Circle Accumulator Threshold', win_name)
            minLineLength = cv2.getTrackbarPos('Minimum Line Length', win_name)
            maxLineGap = cv2.getTrackbarPos('Maximum Line Gap', win_name)
            minDistBtwnEdges = cv2.getTrackbarPos('Minimum Distance Between Edges', win_name)
            maxDistBtwnEdges = cv2.getTrackbarPos('Maximum Distance Between Edges', win_name)
            minRadius = cv2.getTrackbarPos('Minimum Radius', win_name)
            maxRadius = cv2.getTrackbarPos('Maximum Radius', win_name)
            dispTolerance = cv2.getTrackbarPos('Maximum Error for Tip and Axis Alignment', win_name)

            # 3. Detect Drill Tip Handle Edges with Hough Probabilistic Transform 
            cannyMinThreshold = 80;
            edges = cv2.Canny(blurred, cannyMinThreshold, cannyThreshold);
            
            lineTuple = detectLines(edges, line_thresh, minLineLength, maxLineGap, minDistBtwnEdges, maxDistBtwnEdges, 10)    
            if (lineTuple is not None):
                line1 = lineTuple[0]
                line2 = lineTuple[1]
                ax1, ay1, ax2, ay2 = line1[0]
                cx1, cy1, cx2, cy2 = line2[0]
                
                bx1 = int((ax1+cx1)/2.0)
                by1 = int((ay1+cy1)/2.0)
                bx2 = int((ax2+cx2)/2.0)
                by2 = int((ay2+cy2)/2.0)
                centralAxis = ((bx1, by1), (bx2, by2))

                cv2.line(color_image, (ax1, ay1), (ax2, ay2), (255, 0, 0), 5)    
                cv2.line(color_image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 5)
                cv2.line(color_image, centralAxis[0], centralAxis[1], (125, 125, 0), 5)

                # 4. Detect Drill Tip (Ball/Sphere) Using Hough Circles if Edges Were Detected
                accumulatorRes = 1     
                minDist = 2000;             

                circle = detectCircles(blurred, accumulatorRes, minDist, cannyThreshold, circ_thresh, minRadius, maxRadius, centralAxis, dispTolerance)
                if not (circle is None):
                    center = circle[0]
                    radius = circle[1]
                    center_x = center[0]
                    center_y = center[1]

                    micronsOverPixels = 1000/radius
                    centralAxisSlope = 0
                    if (bx2==bx1):
                        centralAxisSlope = math.inf
                    else:
                        centralAxisSlope = (by2-by1) / (bx2-bx1)
                    
                    cv2.circle(color_image, (center_x, center_y), radius, (0, 255, 0), 2)
                    cv2.circle(color_image, (center_x, center_y), 2, (0, 0, 255), 3)
                    cv2.putText(color_image, f"Microns per pixel: {micronsOverPixels: .2f}", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
                    cv2.putText(color_image, f"Slope of tool: {centralAxisSlope: .2f}", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)

                    # 5. Calculate the world coordinates using the camera's intrinsic and extrinsic parameters
                    # R_mtx, tvec = findRTvecs(objpoints, imgpoints, new_K, distCoeffs)
                    # worldCoords = calcWorldCoordinates(center[0], center[1], new_K, R_mtx, tvec)

                    # cv2.putText(color_image, f"Center (mm): ({worldCoords[0][0] :.2f}, {worldCoords[1][0]: .2f})", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
            
            # Show Result
            cv2.imshow(win_name, color_image)
            
            k = cv2.waitKey(1)
            if k == 27:
                grabResult.Release()
        
                # Releasing the resource    
                camera.StopGrabbing()
                cv2.destroyAllWindows()

detectTip("adjacent_camera_calibration.json")