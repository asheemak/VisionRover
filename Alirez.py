import cv2
import numpy as np


def contourSolidity(contour):
	
	contour_area = cv2.contourArea(contour)  
	convex_hull = cv2.convexHull(contour)  
	convex_area = cv2.contourArea(convex_hull)  
	
	solidity = contour_area / convex_area if convex_area != 0 else 0
	return solidity


def contourFactorStruct(contour):
    
    contour_area = cv2.contourArea(contour)  
    perimeter = cv2.arcLength(contour, True)  
    structure = (perimeter ** 2) / contour_area if contour_area != 0 else 0
    return structure


def contourAnisometry(contour):
    
    if len(contour) >= 5: 
        ellipse = cv2.fitEllipse(contour)
        MA, ma = ellipse[1]  
        aspect_ratio = MA / ma if ma != 0 else 0

        anisometry = aspect_ratio
    else:
        anisometry = 0

    return anisometry



def contourAxisMinorLength(contour):

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        _, (MA, ma), _ = ellipse
        length = min(MA, ma)
    else:
        length = 0

    return length


def contourAxisMajorLength(contour):

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        _, (MA, ma), _ = ellipse
        length = max(MA, ma)
    else:
        length = 0
    
    return length


def contourOuterRadius(contour):

    _, radius = cv2.minEnclosingCircle(contour)
    return radius



def minMaxLoc(input):
    if len(input.shape) == 3:  # RGB or RGBA image
        minVal_per_channel = []
        maxVal_per_channel = []
        minLoc_per_channel = []
        maxLoc_per_channel = []

        for i in range(input.shape[2]):

            minVal = np.min(input[:,:,i])
            maxVal = np.max(input[:,:,i])
            minLoc = np.where(input[:,:,i] == minVal)
            maxLoc = np.where(input[:,:,i] == maxVal)

            minVal_per_channel.append(minVal)
            maxVal_per_channel.append(maxVal)
            minLoc_per_channel.append((minLoc[0][0], minLoc[1][0]))
            maxLoc_per_channel.append((maxLoc[0][0], maxLoc[1][0]))

        return minVal_per_channel, maxVal_per_channel, minLoc_per_channel, maxLoc_per_channel
    
    else:  # Grayscale image

        minVal = np.min(input)
        maxVal = np.max(input)
        minLoc = np.where(input == minVal)
        maxLoc = np.where(input == maxVal)

        return minVal, maxVal, (minLoc[0][0], minLoc[1][0]), (maxLoc[0][0], maxLoc[1][0])


def panoramaStichting(leftView, rightView):

    if len(leftView.shape) == 3:
        if leftView.shape[2] == 3:
            gray1 = cv2.cvtColor(leftView,cv2.COLOR_BGR2GRAY)
        elif leftView.shape[2] == 4:
            gray1 = cv2.cvtColor(leftView,cv2.COLOR_BGRA2GRAY)

    if len(rightView.shape) == 3:
        if rightView.shape[2] == 3:
            gray2 = cv2.cvtColor(rightView, cv2.COLOR_BGR2GRAY)
        elif rightView.shape[2] == 4:
            gray2 = cv2.cvtColor(rightView,cv2.COLOR_BGRA2GRAY)
    

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)


    matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    H_inv = np.linalg.inv(H)

    stitchedImage = cv2.warpPerspective(rightView, H_inv, (leftView.shape[1] + rightView.shape[1], leftView.shape[0]))

    stitchedImage[0:leftView.shape[0], 0:leftView.shape[1]] = leftView

    return stitchedImage


def imageNumOfHoles(image):
    all_contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    holes = sum(1 for i in range(len(all_contours)) if hierarchy[0][i][3] != -1)
    return holes
