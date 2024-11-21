import cv2
import numpy as np
import math
import pywt

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


def contourBulkiness(contour):
    contour_area = cv2.contourArea(contour)  
    convex_hull = cv2.convexHull(contour)    
    convex_area = cv2.contourArea(convex_hull)  
    feret_diameter = np.sqrt(4 * convex_area / np.pi) if convex_area > 0 else 0
    bulkiness = contour_area / (np.pi * (feret_diameter / 2) ** 2) if feret_diameter > 0 else 0
    return bulkiness

def contourRoundness(contour):
    contour_area = cv2.contourArea(contour)  
    if contour_area > 0:
        perimeter = cv2.arcLength(contour, True)  
        roundness = (4 * contour_area) / (np.pi * (perimeter ** 2)) if perimeter != 0 else 0
    else:
         roundness = 0  
    return roundness

def contourAspectRatio(contour):
    _, _, w, h = cv2.boundingRect(contour)
    aspectRatio = (w / h if h != 0 else 0) 
    return aspectRatio

def contourRectangularity(contour):
    bounding_box = cv2.boundingRect(contour)  
    bounding_box_area = bounding_box[2] * bounding_box[3]  
    contour_area = cv2.contourArea(contour)  
    rectangularity = contour_area / bounding_box_area if bounding_box_area != 0 else 0
    return rectangularity

def contourNumOfSides(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)  
    approx = cv2.approxPolyDP(contour, epsilon, True)  
    numSides = len(approx) 
    return numSides

def contourCircularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    return(circularity)

def contourCentroid(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = 0, 0       
    centroid = [cX, cY]
    return centroid

def prewitt(image):
    prewitt_x = cv2.filter2D(image, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
    prewitt_y = cv2.filter2D(image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    return prewitt_x,prewitt_y

def roberts(image):
    roberts_x = cv2.filter2D(image, -1, np.array([[1, 0], [0, -1]]))
    roberts_y = cv2.filter2D(image, -1, np.array([[0, 1], [-1, 0]]))
    roberts_edges = np.sqrt(roberts_x**2 + roberts_y**2)
    return roberts_x, roberts_y, roberts_edges

def laplacianOfGaussian(image, sigma):

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    log_edges = cv2.Laplacian(blurred, cv2.CV_64F)
    log_edges = np.abs(log_edges)
    
    return log_edges

def differenceOfGaussian(image, sigma1, sigma2):
    blurred1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    blurred2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    dog_edges = blurred1 - blurred2
    return dog_edges

def contourEccentricity(contour):
    if len(contour) >= 5:  
        ellipse = cv2.fitEllipse(contour)
        _, (MA, ma), _ = ellipse  
        eccentricity = MA / ma
    else:
        eccentricity = 0
    return eccentricity

def contourInnerRectArea(contour):
    inner_rect = cv2.minAreaRect(contour)
    innerRectArea = inner_rect[1][0]* inner_rect[1][1]
    return innerRectArea

def lineProfile(image, startPoint, endPoint, lineColor):
    def get_line_coordinates(startPoint, endPoint):
        x1, y1 = startPoint
        x2, y2 = endPoint
        coordinates = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        for _ in range(max(dx, dy) + 1):  
            coordinates.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return coordinates

    def get_line_profile(image, startPoint, endPoint):

        coordinates = get_line_coordinates(startPoint, endPoint)
        profile = [image[y, x] for x, y in coordinates if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]]
        return profile, coordinates

    line_thickness = 1
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.line(img, startPoint, endPoint, lineColor, line_thickness)
    profile, line_coordinates = get_line_profile(image, startPoint, endPoint)
    lineLength = np.hypot(endPoint[0] - startPoint[0], endPoint[1] - startPoint[1])
    lineAngle = np.arctan2(endPoint[1] - startPoint[1], endPoint[0] - startPoint[0]) * 180 / np.pi
    deriv2 = np.gradient(np.gradient(profile))
    deriv1 = np.gradient(profile)

    return img, profile, deriv1.tolist(), deriv2.tolist(), lineLength, lineAngle


def randomForest(features, labels, maxDepth, testRatio):
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32).reshape(-1, 1)
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    features, labels = features[indices], labels[indices]
    
    splitIndex = int(features.shape[0] * (1 - testRatio))
    XTrain, XTest = features[:splitIndex], features[splitIndex:]
    yTrain, yTest = labels[:splitIndex], labels[splitIndex:]
    
    rfModel = cv2.ml.RTrees_create()
    rfModel.setMaxDepth(maxDepth)
    rfModel.setMinSampleCount(2)
    rfModel.setRegressionAccuracy(0)
    rfModel.setUseSurrogates(False)
    rfModel.setMaxCategories(len(np.unique(labels)))
    rfModel.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6))
    rfModel.train(XTrain, cv2.ml.ROW_SAMPLE, yTrain)
    
    _, trainPreds = rfModel.predict(XTrain)
    trainAccuracy = np.mean((trainPreds == yTrain).astype(np.float32)) * 100
    
    numClasses = len(np.unique(labels))
    trainConfusionMx = np.zeros((numClasses, numClasses), dtype=np.int32)
    for trueLabel, predLabel in zip(yTrain.flatten(), trainPreds.flatten()):
        trainConfusionMx[trueLabel, int(predLabel)] += 1
    
    _, testPreds = rfModel.predict(XTest)
    testAccuracy = np.mean((testPreds == yTest).astype(np.float32)) * 100
    
    testConfusionMx = np.zeros((numClasses, numClasses), dtype=np.int32)
    for trueLabel, predLabel in zip(yTest.flatten(), testPreds.flatten()):
        testConfusionMx[trueLabel, int(predLabel)] += 1
    
    return rfModel, trainAccuracy, trainConfusionMx, testAccuracy, testConfusionMx


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

"""
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
"""


def imageNumOfHoles(image):
    all_contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = sum(1 for i in range(len(all_contours)) if hierarchy[0][i][3] != -1)
    return holes

def svm(features, labels, kernelType, C, gamma, testRatio):
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32).reshape(-1, 1)
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    features, labels = features[indices], labels[indices]

    splitIndex = int(features.shape[0] * (1 - testRatio))
    XTrain, XTest = features[:splitIndex], features[splitIndex:]
    yTrain, yTest = labels[:splitIndex], labels[splitIndex:]

    svmModel = cv2.ml.SVM_create()
    svmModel.setKernel(kernelType)  
    svmModel.setC(C)  
    svmModel.setGamma(gamma)  
    svmModel.setType(cv2.ml.SVM_C_SVC)  
    svmModel.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6))
    svmModel.train(XTrain, cv2.ml.ROW_SAMPLE, yTrain)

    _, trainPreds = svmModel.predict(XTrain)
    trainAccuracy = np.mean((trainPreds == yTrain).astype(np.float32)) * 100

    numClasses = len(np.unique(labels))
    trainConfusionMx = np.zeros((numClasses, numClasses), dtype=np.int32)
    for trueLabel, predLabel in zip(yTrain.flatten(), trainPreds.flatten()):
        trainConfusionMx[trueLabel, int(predLabel)] += 1

    _, testPreds = svmModel.predict(XTest)
    testAccuracy = np.mean((testPreds == yTest).astype(np.float32)) * 100

    testConfusionMx = np.zeros((numClasses, numClasses), dtype=np.int32)
    for trueLabel, predLabel in zip(yTest.flatten(), testPreds.flatten()):
        testConfusionMx[trueLabel, int(predLabel)] += 1

    return svmModel, trainAccuracy, trainConfusionMx, testAccuracy, testConfusionMx

def imageEdgeAngle(image):
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edge_angles = np.arctan2(sobelY, sobelX) * 180 / np.pi
    meanEdgeAngle = np.mean(edge_angles)
    return meanEdgeAngle


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


def imageEdgeDensity(image, firstThresh, secondThresh):
    edges = cv2.Canny(image, firstThresh, secondThresh)
    edgeDensity = np.count_nonzero(edges) / edges.size
    return edgeDensity



def contrastAdjustment(image, in_ranges, out_ranges, gammas, alphas, betas):
    adjustedImage = np.zeros_like(image, dtype=np.float32)
    for i, (in_range, out_range) in enumerate(zip(in_ranges, out_ranges)):
        in_min, in_max = in_range
        out_min, out_max = out_range
        gamma = gammas[i]
        alpha = alphas[i]
        beta = betas[i]
        mask = (image >= in_min) & (image <= in_max)
        normalized = (image.astype(np.float32) - in_min) / (in_max - in_min)
        normalized = np.clip(normalized, 0, 1)  
        gamma_corrected = np.power(normalized, gamma) * mask
        scaled = gamma_corrected * (out_max - out_min) + out_min
        transformed = alpha * scaled + beta
        adjustedImage[mask] = transformed[mask]

    return adjustedImage


def pcaFusion(firstImage, secondImage):
    stacked_images = np.dstack((firstImage, secondImage))
    reshaped_images = stacked_images.reshape(-1, 2)
    reshaped_images = np.float32(reshaped_images)
    mean, eigenvectors = cv2.PCACompute(reshaped_images, mean=None, maxComponents=1)
    fusedImage = cv2.PCAProject(reshaped_images, mean, eigenvectors)
    fusedImage = fusedImage.reshape(firstImage.shape)
    return fusedImage.astype(np.float32)

def waveletFusion(firstImage, secondImage):
    coeffs1 = pywt.dwt2(firstImage, 'haar')
    coeffs2 = pywt.dwt2(secondImage, 'haar')
    cA1, (cH1, cV1, cD1) = coeffs1
    cA2, (cH2, cV2, cD2) = coeffs2
    cA_fused = (cA1 + cA2) / 2
    cH_fused = (cH1 + cH2) / 2
    cV_fused = (cV1 + cV2) / 2
    cD_fused = (cD1 + cD2) / 2

    fusedImage = pywt.idwt2((cA_fused, (cH_fused, cV_fused, cD_fused)), 'haar')

    return fusedImage.astype(np.float32)



def fftFusion(firstImage, secondImage):
    def fft_decompose(image):
        fft_image = np.fft.fft2(image)
        fft_image_shifted = np.fft.fftshift(fft_image)

        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2

        low_pass = np.zeros((rows, cols))
        low_pass[center_row-rows//4:center_row+rows//4, center_col-cols//4:center_col+cols//4] = 1
        low_pass_fft = fft_image_shifted * low_pass

        high_pass = 1 - low_pass
        high_pass_fft = fft_image_shifted * high_pass

        low_pass_image = np.fft.ifft2(np.fft.ifftshift(low_pass_fft))
        high_pass_image = np.fft.ifft2(np.fft.ifftshift(high_pass_fft))
        return low_pass_image, high_pass_image

    
    low1, high1 = fft_decompose(firstImage)
    low2, high2 = fft_decompose(secondImage)

    fused_low = (np.abs(low1) + np.abs(low2)) / 2
    fused_high = (np.abs(high1) + np.abs(high2)) / 2

    fused_fft = np.fft.fft2(fused_low) + np.fft.fft2(fused_high)
    fused_fft_shifted = np.fft.fftshift(fused_fft)
    fusedImage = np.fft.ifft2(np.fft.ifftshift(fused_fft_shifted))

    fusedImage = np.real(fusedImage)
    return fusedImage.astype(np.float32)

def floodFill(image, seed_point, new_color, lo_diff, up_diff, floodFillFlags):
    if isinstance(new_color, int):
        new_color = (new_color, new_color, new_color)
    filledImage = image.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask must be 2 pixels larger than the image
    retval, filledImage, mask, rect = cv2.floodFill(
        filledImage, mask, seedPoint=seed_point, newVal=new_color, loDiff=(lo_diff, lo_diff, lo_diff),
        upDiff=(up_diff, up_diff, up_diff), flags=floodFillFlags
    )

    return retval, filledImage, mask, rect
