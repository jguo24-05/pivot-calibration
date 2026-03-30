import pypylon.pylon as py
from charuco_calibration import *
from stereo_calibration import *
import numpy as np
import cv2
import math
import json


# TODO: refine hough circles to be more accomodating (currently trying hough_circles_alt)
# Note: top-down lighting from a little far away worked (on 3/29)


######################### GEOMETRIC HELPERS #########################
# Checks if two lines are parallel with the given tolerance
def checkParallel(line1, line2, tolerance):
    eps = 10**-6
    x1, y1, x2, y2 = line1[0]
    u1, v1, u2, v2 = line2[0]
    if (abs(x2 - x1) < eps and abs(u2 - u1) < eps): return True
    elif (abs(x2 - x1) < eps or abs(u2 - u1) < eps): return False
    m1 = (y2 - y1) / (x2 - x1)
    m2 = (v2 - v1) / (u2 - u1)
    return abs(m1-m2) < tolerance
    
   
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
    eps = 10**-3
    ax1, ax2, ay1, ay2 = line1[0]
    bx1, bx2, by1, by2 = line2[0]
    if (abs(ax2-ax1) < eps):              # vertical line
        return abs(bx1-ax1)
    elif (abs(ay2-ay1) < eps):            # horizontal line
        return abs(by1-ay1)
    else:
        m = (ay2-ay1) / float((ax2-ax1))
        c1 = ay1 - m*ax1
        c2 = by1 - m*bx1
        return (abs(c2 - c1)) / ((1 + pow(m, 2))**0.5)
    
    
def angleBetweenLines(line1, line2):
    eps = 10**-3
    ax1, ax2, ay1, ay2 = line1[0]
    bx1, bx2, by1, by2 = line2[0]

    if (abs(ax2-ax1) < eps and abs(bx2-bx1) < eps): return 0
    elif (abs(ax2-ax1) < eps or abs(bx2-bx1) < eps): return math.pi/2   # arbitrary large value

    m1 = (ay2-ay1) / (ax2-ax1)
    m2 = (by2-by1) / (bx2-bx1)
    norm1 = (1 + pow(m1, 2))**0.5
    norm2 = (1 + pow(m2, 2))**0.5

    cosTheta = (1 + m1*m2) / (norm1*norm2)
    return math.acos(cosTheta)


# Detect lines using Probabilistic Hough Transform
def detectLines(color_image, edges, line_thresh, minLineLength, maxLineGap, minDistBtwnEdges, maxDistBtwnEdges, parallelTolerance):
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
                
                if (isParallel):
                    
                    # Debugging - pale blue lines mean parallel but not within distance range
                    ax1, ay1, ax2, ay2 = line1[0]
                    cv2.line(color_image, (ax1, ay1), (ax2, ay2), (255, 255, 120), 5)    
                    bx1, by1, bx2, by2 = line2[0]
                    cv2.line(edges, (bx1, by1), (bx2, by2), (255, 120, 255), 5) 

                    if (distBtwnEdges > minDistBtwnEdges and distBtwnEdges < maxDistBtwnEdges):
                        return (line1, line2)
        
        return None     # no parallel lines were found this frame


# Detect circles that lie on the given centralAxis
def detectCircles(grayFrame, accumulatorRes, minDist, cannyThreshold, circAccThreshold, minRadius, maxRadius, centralAxis, dispTolerance):
    # Detect circles with the Hough transform
    circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT_ALT, 
                            dp = accumulatorRes, minDist = minDist, param1 = 300,   # for hough_gradient_alt 
                            param2 = 0.5, minRadius = minRadius, 
                            maxRadius = maxRadius);

    if not (circles is None):
        circles = np.round(circles).astype("uint16")
        for i in circles[0,:]: 
            center = (i[0], i[1])
            radius = i[2]

            if (pointOnLine(center, centralAxis[0], centralAxis[1], dispTolerance)):
                return (center, radius)


######################### DETECTION LOOP #########################
def findTCP():
    NUM_CAMERAS = 2
    tlf = py.TlFactory.GetInstance()
    
    devs = tlf.EnumerateDevices()
    cam_array = py.InstantCameraArray(NUM_CAMERAS)
    
    for idx, cam in enumerate(cam_array):
        cam.Attach(tlf.CreateDevice(devs[idx]))
    cam_array.Open()

    # Set the exposure time for each camera and store a unique 
    # number for each camera to identify the incoming images
    cam1 = cam_array[0]
    cam1.ExposureTime.SetValue(300000)
    cam1.SetCameraContext(0)
    cam2 = cam_array[1]
    cam2.ExposureTime.SetValue(300000)
    cam2.SetCameraContext(1)

    # Camera Calibration
    with open(f"./calibration_data/cam745_calibration.json", 'r') as file:
        json_data = json.load(file)

    cameraMatrix745 = np.array(json_data['mtx'])
    distCoeffs745 = np.array(json_data['dist'])

    with open(f"./calibration_data/cam746_calibration.json", 'r') as file:
        json_data = json.load(file)

    cameraMatrix746 = np.array(json_data['mtx'])
    distCoeffs746 = np.array(json_data['dist'])
    
    # store last framecount in array
    frame_counts = [0]*NUM_CAMERAS

    converter = py.ImageFormatConverter()
    # converting to opencv bgr format
    converter.OutputPixelFormat = py.PixelType_BGR8packed
    converter.OutputBitAlignment = py.OutputBitAlignment_MsbAligned

    cv2.namedWindow("Window 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 1", 800, 650)
    win_name = "Window 1"

    # Canny Threshold
    cannyThreshMin = 80
    cannyThreshMax = 300
    # Line Accumulator Matrix
    lineAccMin = 100
    lineAccMax = 300
    # Circle Accumulator Matrix (superfluous if hough_gradient_alt)
    circleAccMin = 10
    circleAccMax = 300
    # Minimum circle radius
    minRadiusMin = 50
    minRadiusMax = 100
    # Maximum circle radius
    maxRadiusMin = 200
    maxRadiusMax = 300
    # Minimum line length
    minLineMin = 10
    minLineMax = 100
    # Maximum line gap
    maxLineMin = 50
    maxLineMax = 250
    # Minimum distance between edges
    minDistMin = 30
    minDistMax = 75
    # Maximum distance between edges
    maxDistMin = 100
    maxDistMax = 200
    # Error for tip and axis alignment
    errorMin = 25
    errorMax = 50

    cv2.createTrackbar('Canny Threshold', win_name, cannyThreshMin, cannyThreshMax, lambda a: None);
    # Accumulator Threshold for Lines
    cv2.createTrackbar('Line Accumulator Threshold', win_name, lineAccMin, lineAccMax, lambda a: None);
    # Circle Radius Accumulator Threshold
    cv2.createTrackbar('Circle Accumulator Threshold', win_name, circleAccMin, circleAccMax, lambda a: None)
    # Min Radius: Minimum circle radius to detect
    cv2.createTrackbar('Minimum Radius', win_name, minRadiusMin, minRadiusMax, lambda a: None)
    # Max Radius: Maximum circle radius to detect
    cv2.createTrackbar('Maximum Radius', win_name, maxRadiusMin, maxRadiusMax, lambda a: None)
    # Minimum Line Length: Minimum line length to detect
    cv2.createTrackbar('Minimum Line Length', win_name, minLineMin, minLineMax, lambda a: None)
    # Maximum Line Gap: Maximum allowed gap in a line
    cv2.createTrackbar('Maximum Line Gap', win_name, maxLineMin, maxLineMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Minimum Distance Between Edges', win_name, minDistMin, minDistMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Maximum Distance Between Edges', win_name, maxDistMin, maxDistMax, lambda a: None)
    # Tolerance for how far the radius can be from the detected central axis
    cv2.createTrackbar('Maximum Error for Tip and Axis Alignment', win_name, errorMin, errorMax, lambda a: None)

    cv2.namedWindow("Window 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window 2", 800, 650)
    win_name = "Window 2"

    cv2.createTrackbar('Canny Threshold', win_name, cannyThreshMin, cannyThreshMax, lambda a: None);
    # Accumulator Threshold for Lines
    cv2.createTrackbar('Line Accumulator Threshold', win_name, lineAccMin, lineAccMax, lambda a: None);
    # Circle Radius Accumulator Threshold
    cv2.createTrackbar('Circle Accumulator Threshold', win_name, circleAccMin, circleAccMax, lambda a: None)
    # Min Radius: Minimum circle radius to detect
    cv2.createTrackbar('Minimum Radius', win_name, minRadiusMin, minRadiusMax, lambda a: None)
    # Max Radius: Maximum circle radius to detect
    cv2.createTrackbar('Maximum Radius', win_name, maxRadiusMin, maxRadiusMax, lambda a: None)
    # Minimum Line Length: Minimum line length to detect
    cv2.createTrackbar('Minimum Line Length', win_name, minLineMin, minLineMax, lambda a: None)
    # Maximum Line Gap: Maximum allowed gap in a line
    cv2.createTrackbar('Maximum Line Gap', win_name, maxLineMin, maxLineMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Minimum Distance Between Edges', win_name, minDistMin, minDistMax, lambda a: None)
    # Minimum Distance Between Edge Lines
    cv2.createTrackbar('Maximum Distance Between Edges', win_name, maxDistMin, maxDistMax, lambda a: None)
    # Tolerance for how far the radius can be from the detected central axis
    cv2.createTrackbar('Maximum Error for Tip and Axis Alignment', win_name, errorMin, errorMax, lambda a: None)

    # Detect the tool center point
    (points745, points746) = runDetection(win_name, 
                                          cam_array, frame_counts, converter, 
                                          cameraMatrix745, distCoeffs745, 
                                          cameraMatrix746, distCoeffs746)
    
    # Calculate its 3D position relative to the camera
    worldCoordinates = []
    if (points745 is not None):
        worldPoint = grab3DPoints("./calibration_data/external_parameters.json", points745, points746)
        worldCoordinates.append(worldPoint)

    return worldCoordinates


# Runs the main loop until the TCP is detected by both cameras (loop broken with 'q' key)
def runDetection(win_name, cam_array, frame_counts, converter, cameraMatrix745, distCoeffs745, cameraMatrix746, distCoeffs746):
    currCam_id = 0
    points745 = np.array
    points746 = np.array

    cam_array.StartGrabbing()
    while True:
        with cam_array.RetrieveResult(5000) as res:
            if res.GrabSucceeded():
                img_nr = res.ImageNumber
                cam_id = res.GetCameraContext()
                frame_counts[cam_id] = img_nr

                # Access the image data
                image = converter.Convert(res)
                color_image = image.GetArray()
                
                if (cam_id == 1):
                    new_K, color_image = undistort_image(color_image, cameraMatrix745, distCoeffs745)
                else:
                    new_K, color_image = undistort_image(color_image, cameraMatrix746, distCoeffs746)

                # 2. Preprocessing (Grayscale + Gaussian Blur)
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                # Blur is crucial to reduce noise/specular highlights on metal surfaces
                blurred = cv2.GaussianBlur(gray, (9, 9), 5)

                win_name = 'Window 1'
                if (cam_id == 1):
                    win_name = 'Window 2'

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
                cannyMinThreshold = 50;
                edges = cv2.Canny(blurred, cannyMinThreshold, cannyThreshold);
                
                lineTuple = detectLines(color_image, 
                                        edges, 
                                        line_thresh, 
                                        minLineLength, 
                                        maxLineGap, 
                                        minDistBtwnEdges, 
                                        maxDistBtwnEdges, 
                                        4)    # Allowed deviation between slopes
                
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

                    circle = detectCircles(blurred, 
                                           accumulatorRes, 
                                           minDist, 
                                           cannyThreshold, 
                                           circ_thresh, 
                                           minRadius, 
                                           maxRadius, 
                                           centralAxis, 
                                           dispTolerance)
                    
                    if not (circle is None):
                        center = circle[0]
                        radius = circle[1]
                        center_x = center[0]
                        center_y = center[1]

                        if (currCam_id == 0):
                            points745 = np.transpose(np.array(center_x, center_y))
                            currCam_id = 1
                        else:
                            points746 = np.transpose(np.array(center_x, center_y))

                            cam_array.StopGrabbing()
                            cam_array.Close()
                            return (points745, points746)

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

                if (cam_id == 0):
                    cv2.imshow('Window 1', color_image)
                    
                elif (cam_id == 1):
                    cv2.imshow('Window 2', color_image)
                
                # check if q button has been pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    cam_array.StopGrabbing()
    cam_array.Close()
    return (None, None)


######################### Driver #########################
worldCoords = findTCP()
print(worldCoords)