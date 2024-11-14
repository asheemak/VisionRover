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