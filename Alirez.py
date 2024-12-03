import cv2
import numpy as np
import math

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

######################################################
#   it will add as 78 to 81 in future
######################################################
# def minMaxLoc(input):
#     if len(input.shape) == 3:  # RGB or RGBA image
#         minVal_per_channel = []
#         maxVal_per_channel = []
#         minLoc_per_channel = []
#         maxLoc_per_channel = []

#         for i in range(input.shape[2]):

#             minVal = np.min(input[:,:,i])
#             maxVal = np.max(input[:,:,i])
#             minLoc = np.where(input[:,:,i] == minVal)
#             maxLoc = np.where(input[:,:,i] == maxVal)

#             minVal_per_channel.append(minVal)
#             maxVal_per_channel.append(maxVal)
#             minLoc_per_channel.append((minLoc[0][0], minLoc[1][0]))
#             maxLoc_per_channel.append((maxLoc[0][0], maxLoc[1][0]))

#         return minVal_per_channel, maxVal_per_channel, minLoc_per_channel, maxLoc_per_channel
    
#     else:  # Grayscale image

#         minVal = np.min(input)
#         maxVal = np.max(input)
#         minLoc = np.where(input == minVal)
#         maxLoc = np.where(input == maxVal)

#         return minVal, maxVal, (minLoc[0][0], minLoc[1][0]), (maxLoc[0][0], maxLoc[1][0])
#########################################################

def panoramaStitching(leftView, rightView):

    def __drawInliersOutliers(left_image, right_image, src_point, dst_point, mask):

        rows1, cols1, _ = left_image.shape
        rows2, cols2, _ = right_image.shape

        matchImage = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
        matchImage[:rows1, :cols1, :] = np.dstack([left_image])
        matchImage[:rows2, cols1:cols1 + cols2, :] = np.dstack([right_image])

        matchImage2 = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
        matchImage2[:rows1, :cols1, :] = np.dstack([left_image])
        matchImage2[:rows2, cols1:cols1 + cols2, :] = np.dstack([right_image])

        # draw lines
        for i in range(src_point.shape[0]):
            x1, y1 = src_point[i][0]
            x2, y2 = dst_point[i][0]

            point1 = (int(x1), int(y1))
            point2 = (int(x2 + left_image.shape[1]), int(y2))

            if mask[i][0] == 1:
                cv2.line(matchImage, point1, point2, (0, 255, 0), 1)
            else :
                cv2.line(matchImage2, point1, point2, (255, 0, 0), 1)

        return matchImage, matchImage2
    


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

    Overlay_inlier_image, Overlay_outlier_image = __drawInliersOutliers(leftView, rightView, src_pts, dst_pts, mask)

    H_inv = np.linalg.inv(H)

    stitchedImage = cv2.warpPerspective(rightView, H_inv, (leftView.shape[1] + rightView.shape[1], leftView.shape[0]))

    stitchedImage[0:leftView.shape[0], 0:leftView.shape[1]] = leftView

    return Overlay_inlier_image, Overlay_outlier_image, stitchedImage


def imageNumOfHoles(image):
    all_contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    holes = sum(1 for i in range(len(all_contours)) if hierarchy[0][i][3] != -1)
    return holes

def imageHolesArea(image):
    all_contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    area = sum(cv2.contourArea(all_contours[i]) for i in range(len(all_contours)) if hierarchy[0][i][3] != -1)
    return area

def minAreaRect(points):

    center, size, angle = cv2.minAreaRect(points)
    
    radians = np.radians(angle)

    halfWidth = size[0] / 2.0
    halfHeight = size[1] / 2.0

    relativePoints = [[-halfWidth, -halfHeight],[halfWidth, -halfHeight],[halfWidth, halfHeight],[-halfWidth, halfHeight]]

    rectCorners = []
    for point in relativePoints:
        x = point[0]
        y = point[1]
        rectCorners.append(( int(center[0] + x * np.cos(radians) - y * np.sin(radians)), int(center[1] + x * np.sin(radians) + y * np.cos(radians))))


    def angle_from_centroid(point):
        return math.atan2(point[1] - center[1], point[0] - center[0])
    

    sorted_corners = sorted(rectCorners, key=angle_from_centroid)

    return center, size, angle, sorted_corners


def contourMoments(contour):
    
    M = cv2.moments(contour)

    moments = []

    for key in M:
        moments.append(M[key]) 

    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = 0, 0

    moments.append(cX)
    moments.append(cY)

    huMoments = cv2.HuMoments(M).flatten()

    for i in range(7):
        moments.append(huMoments[i]) 

    return moments


def imageMoments(image, binaryImage):
    
    M = cv2.moments(image, binaryImage)

    moments = []

    for key in M:
        moments.append(M[key]) 

    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = 0, 0

    moments.append(cX)
    moments.append(cY)

    huMoments = cv2.HuMoments(M).flatten()

    for i in range(7):
        moments.append(huMoments[i]) 

    return moments




def splitData(features, labels, testRatio, shuffle=True):

    if isinstance(features, np.ndarray):
        if len(features.shape) > 2:
            raise ValueError("feature should be 2d")
        
    if isinstance(labels, np.ndarray):
        if len(labels.shape) > 2:
            raise ValueError("labels should be 2d")
        
        if len(labels.shape) == 2:
            if labels.shape[0] != 1 and labels.shape[1] != 1:
                raise ValueError("labels should 1xn or nx1")

            if labels.shape[0] == 1:
                labels = labels.flatten()
     
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    if labels.flatten().shape[0] == features.shape[0]:
        indices = np.arange(features.shape[0])
        if shuffle:
            np.random.shuffle(indices)

        features, labels = features[indices], labels[indices]
        
        split_idx = round(features.shape[0] * (1 - testRatio))
        x_train, x_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

    elif labels.flatten().shape[0] == features.shape[1]:
        indices = np.arange(features.shape[1])
        if shuffle:
            np.random.shuffle(indices)

        features, labels = features[:, indices], labels[indices]
        
        split_idx = round(features.shape[1] * (1 - testRatio))
        x_train, x_test = features[:, :split_idx], features[:, split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

    else:
        raise ValueError("Each feature must have a corresponding label")
    
    return x_train, x_test, y_train, y_test


def trainModel(model, XTrain, yTrain, sampleType=cv2.ml.ROW_SAMPLE):
    
    model.train(XTrain, sampleType, yTrain)
    
    return model



def predict(model, features):
    features = np.array(features, dtype=np.float32)
    
    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    _, preds = model.predict(features)
    preds = preds.flatten()
    return preds


def evaluateModel(model, features, labels):
    
    _, preds = model.predict(features)

    labels = labels.flatten()
    preds = preds.flatten()
    accuracy = np.mean((preds == labels).astype(np.float32)) * 100 
  
    num_classes = len(np.unique(labels))
    confusion_mx = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(labels.flatten(), preds.flatten()):
        confusion_mx[true_label, int(pred_label)] += 1
    
    return accuracy, confusion_mx



def SVM(C=1.0, kernelType=cv2.ml.SVM_LINEAR, degree=0, gamma=0, classWeights=None):
  
    svm_model = cv2.ml.SVM_create()

    # Set the parameters
    svm_model.setC(C)
    svm_model.setKernel(kernelType)
    svm_model.setDegree(degree)
    svm_model.setGamma(gamma)
    svm_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
    svm_model.setType(cv2.ml.SVM_C_SVC)
    
    if classWeights is not None:
        svm_model.setClassWeights(classWeights)

    return svm_model


def randomForest(maxDepth=10, minSampleCount=2, maxCategories=10):

    # Create an RTrees instance
    rf_model = cv2.ml.RTrees_create()

    # Set the parameters
    rf_model.setMaxDepth(maxDepth)
    rf_model.setMinSampleCount(minSampleCount)
    rf_model.setMaxCategories(maxCategories)
    rf_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    return rf_model

