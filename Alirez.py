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