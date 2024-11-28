import cv2
import numpy as np
import pywt


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
    rectang = contour_area / bounding_box_area if bounding_box_area != 0 else 0
    return rectang

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

def laplacianOfGaussian(image, sigma=1):

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    log_edges = cv2.Laplacian(blurred, cv2.CV_32F)
    log_edges = np.abs(log_edges)
    
    return log_edges

def differenceOfGaussian(image, sigma1=1, sigma2=2):
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



def lineProfile(image, points, lineColor):
    # Extract startPoint and endPoint from points
    startPoint = points[0]
    endPoint = points[1]

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
        profile = [
            image[y, x] for x, y in coordinates
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]
        ]
        return profile, coordinates

    line_thickness = 1
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.line(img, startPoint, endPoint, lineColor, line_thickness)
    profile, line_coordinates = get_line_profile(image, startPoint, endPoint)
    Length = np.hypot(endPoint[0] - startPoint[0], endPoint[1] - startPoint[1])

    deriv2 = np.gradient(np.gradient(profile))
    deriv1 = np.gradient(profile)

    return img, profile, deriv1.tolist(), deriv2.tolist(), Length


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
    sobelX = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
    sobelY = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
    edge_angles = np.arctan2(sobelY, sobelX) * 180 / np.pi
    meanEdgeAngle = np.mean(edge_angles)
    return meanEdgeAngle

def imageEdgeDensity(image, firstThresh, secondThresh):
    edges = cv2.Canny(image, firstThresh, secondThresh)
    edgeDensity = np.count_nonzero(edges) / edges.size
    return edgeDensity



def adjustContrastExperimental(image, in_ranges, out_ranges, gammas, alphas, betas):
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


def floodFill(image, seedPoint, newVal, loDiff, upDiff, floodFillFlags):
    if isinstance(newVal, int):
        newVal = (newVal, newVal, newVal)
    filledImage = image.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask must be 2 pixels larger than the image
    retval, filledImage, mask, rect = cv2.floodFill(
        filledImage, mask, seedPoint=seedPoint, newVal=newVal, loDiff=(loDiff, loDiff, loDiff),
        upDiff=(upDiff, upDiff, upDiff), flags=floodFillFlags
    )

    return retval, filledImage, mask, rect


def fitLine(image, points, dist_type, param, reps, aeps,
                               color_line, thickness):
    points = np.array(points, dtype=np.float32)
    line = cv2.fitLine(points, dist_type, param, reps, aeps)
    vx, vy, x, y = line.flatten()
    height, width = image.shape[:2]
    t = max(width, height)  # Extend line beyond the image size
    pt1 = (int(x - vx * t), int(y - vy * t))
    pt2 = (int(x + vx * t), int(y + vy * t))
    cv2.line(image, pt1, pt2, color_line, thickness)

    return image, vx, vy, x, y