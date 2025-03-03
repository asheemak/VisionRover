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
    dog_edges = cv2.subtract(blurred1 - blurred2)
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

def kmeansSegmentation(image, k, criteria_eps, criteria_max_iter, attempts):

    data = np.float32(image.reshape(-1, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, criteria_max_iter, criteria_eps)
    flags=cv2.KMEANS_RANDOM_CENTERS
    ret, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, flags)
    
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(image.shape)
    
    return segmented_img

def logisticRegression(learningRate, iterations, regularization, trainMethod, miniBatchSize):
    model = cv2.ml.LogisticRegression_create()
    model.setLearningRate(learningRate)
    model.setIterations(iterations)
    model.setRegularization(regularization)
    model.setTrainMethod(trainMethod)
    model.setMiniBatchSize(miniBatchSize)
    return model

def zscore(features):
    features = np.asarray(features)
    reshaped = features.reshape(features.shape[0], 1, features.shape[1])
    mean, std = cv2.meanStdDev(reshaped)
    mean = mean.ravel()
    std = std.ravel()
    Mat = (features - mean) / (std + 1e-8)
    return Mat, mean, std 

def normMinMax(features):
    features = np.asarray(features)
    reshaped = features.reshape(features.shape[0], 1, features.shape[1])
    channels = cv2.split(reshaped)
    min_vals = []
    max_vals = []
    for ch in channels:
        min_val, max_val, _, _ = cv2.minMaxLoc(ch)
        min_vals.append(min_val)
        max_vals.append(max_val)
    
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    
    # Perform min-max normalization: (x - min) / (max - min)
    mat = (features - min_vals) / ((max_vals - min_vals) + 1e-8)
    return mat, min_vals, max_vals

def anovaFeatureRank(X, y):
    
    #X: shape (n_samples, n_features)
    #y: binary labels (0 or 1)
    
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]

    n0 = X_class0.shape[0]
    n1 = X_class1.shape[0]
    n = n0 + n1

    mean0 = np.mean(X_class0, axis=0)
    mean1 = np.mean(X_class1, axis=0)
    mean_all = np.mean(X, axis=0)

    # Between-group sum of squares
    SSB = n0 * (mean0 - mean_all) ** 2 + n1 * (mean1 - mean_all) ** 2

    # Within-group sum of squares
    SSW0 = np.sum((X_class0 - mean0) ** 2, axis=0)
    SSW1 = np.sum((X_class1 - mean1) ** 2, axis=0)
    SSW = SSW0 + SSW1

    df_between = 1      # for 2 classes
    df_within = n - 2

    F = (SSB / df_between) / (SSW / df_within)
    feature_ranks  = np.argsort(F)[::-1]
    return feature_ranks 

def PCA(data, n_components):

    mean, eigenvectors = cv2.PCACompute(data, mean=None, maxComponents=n_components)
    transformed_data = cv2.PCAProject(data, mean, eigenvectors)
    
    return transformed_data, mean, eigenvectors

def boost(weakCount=100, weightTrimRate=0.95, maxDepth=1):
    useSurrogates=False
    model = cv2.ml.Boost_create()
    model.setBoostType(cv2.ml.BOOST_DISCRETE)
    model.setWeakCount(weakCount)
    model.setWeightTrimRate(weightTrimRate)
    model.setMaxDepth(maxDepth)
    model.setUseSurrogates(useSurrogates)
    return model

def siftDetectCompute(image, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma):

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )

    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def orbDetectCompute(image, nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, patchSize=31):
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=scaleFactor,
        nlevels=nlevels,
        edgeThreshold=edgeThreshold,
        firstLevel=firstLevel,
        WTA_K=WTA_K,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=patchSize
    )
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def gLCM(image, d, angle):
    max_gray_level = image.max() + 1
    rows, cols = image.shape
    glcm = np.zeros((max_gray_level, max_gray_level), dtype=np.float64)
    
    # Compute the pixel offset for the given distance and angle.
    dx, dy = int(np.cos(angle) * distance), int(np.sin(angle) * distance)
    
    # Compute shifted indices.
    row_indices, col_indices = np.indices((rows, cols))
    shifted_row = row_indices + dy
    shifted_col = col_indices + dx
    
    # Create a mask for valid indices.
    valid = (
        (shifted_row >= 0) & (shifted_row < rows) &
        (shifted_col >= 0) & (shifted_col < cols)
    )
    
    # Select current and neighbor pixels.
    current_pixels = image[valid]
    neighbor_pixels = image[shifted_row[valid], shifted_col[valid]]
    
    # Compute the 2D histogram.
    glcm, _, _ = np.histogram2d(
        current_pixels,
        neighbor_pixels,
        bins=[max_gray_level, max_gray_level],
        range=[[0, max_gray_level], [0, max_gray_level]]
    )
    
    # Normalize the GLCM.
    total = glcm.sum()
    if total > 0:
        glcm /= total
        
    return glcm    

def compute_homogeneity(glcm):
    i, j = np.indices(glcm.shape)
    return np.sum(glcm / (1.0 + (i - j) ** 2))

def gLCMDissimilarity(glcm):
    i, j = np.indices(glcm.shape)
    return np.sum(glcm * np.abs(i - j))

def gLCMCorrelation(glcm):
    i, j = np.indices(glcm.shape)
    mean_i = np.sum(i * glcm)
    mean_j = np.sum(j * glcm)
    std_i = np.sqrt(np.sum(((i - mean_i) ** 2) * glcm))
    std_j = np.sqrt(np.sum(((j - mean_j) ** 2) * glcm))
    return np.sum((i - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)

def gLCMContrast(glcm):
    i, j = np.indices(glcm.shape)
    return np.sum(glcm * (i - j) ** 2)

def magnitudeFeatures(f_transform):

    # Shift zero frequency component to the center
    f_transform_centered = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magn_spectrum = cv2.magnitude(f_transform_centered[:, :, 0],
                                       f_transform_centered[:, :, 1])

    # Calculate individual features
    mean_magn = np.mean(magnitude_spectrum)
    var_magn = np.var(magnitude_spectrum)
    max_magn = np.max(magnitude_spectrum)
    sum_magn = np.sum(magnitude_spectrum)

    return mean_magn, var_magn, max_magn, sum_magn, magn_spectrum

def furierEnergyFeatures(f_transform):
    f_transform_centered = np.fft.fftshift(f_transform)
    magnitude_spectrum = cv2.magnitude(f_transform_centered[:, :, 0],
                                       f_transform_centered[:, :, 1])
    power_spect = np.abs(magnitude_spectrum) ** 2
    mean_pow = np.mean(power_spect)
    variance_pow = np.var(power_spect)
    max_pow = np.max(power_spect)
    sum_pow = np.sum(power_spect)
    height, width = power_spect.shape
    low_freq_band = power_spect[:height // 4, :width // 4]
    mid_freq_band = power_spect[height // 4:3 * height // 4, width // 4:3 * width // 4]
    high_freq_band = power_spect[3 * height // 4:, 3 * width // 4:]
    energyL = np.sum(low_freq_band)
    energyM = np.sum(mid_freq_band)
    energyH = np.sum(high_freq_band)



    return (mean_pow, variance_pow, max_pow, sum_pow, 
            energyL, energyM, energyH, 
            power_spect)    


def LBP(image):
    height, width = image.shape
    lbp = np.zeros((height, width), dtype=np.uint8)

    
    cond = (image[:-1, :-1] >= image[1:, 1:])
    lbp[1:, 1:] += 128 * cond.astype(np.uint8)

    
    cond = (image[:-1, :] >= image[1:, :])
    lbp[1:, :] += 64 * cond.astype(np.uint8)

    
    cond = (image[:-1, 1:] >= image[1:, :width-1])
    lbp[1:, :width-1] += 32 * cond.astype(np.uint8)

    
    cond = (image[:, 1:] >= image[:, :width-1])
    lbp[:, :width-1] += 16 * cond.astype(np.uint8)

    
    cond = (image[1:, 1:] >= image[:-1, :-1])
    lbp[:-1, :-1] += 8 * cond.astype(np.uint8)

    
    cond = (image[1:, :] >= image[:-1, :])
    lbp[:-1, :] += 4 * cond.astype(np.uint8)

    
    cond = (image[1:, :-1] >= image[:-1, 1:])
    lbp[:-1, 1:] += 2 * cond.astype(np.uint8)

    
    cond = (image[:, :-1] >= image[:, 1:])
    lbp[:, 1:] += 1 * cond.astype(np.uint8)

    return lbp


def grabcut(input_image, mode, rect, mask, iter):

    # Initialize the mask and models
    grabcut_mask = np.zeros(input_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Handle modes
    if mode in ['RECT', 'MASK', 'MASK+RECT', 'EVAL']:
        if mask is None and mode in ['MASK', 'MASK+RECT']:
            raise ValueError("Mask must be provided for MASK or MASK+RECT mode")

        if mask is not None:
            grabcut_mask[mask == 1] = cv2.GC_FGD  # Sure foreground
            grabcut_mask[mask == 3] = cv2.GC_PR_FGD  # Probable foreground
            grabcut_mask[mask == 0] = cv2.GC_BGD  # Sure background
            grabcut_mask[mask == 2] = cv2.GC_PR_BGD  # Probable background

    if mode == 'RECT':
        if rect is None:
            raise ValueError("Rectangle coordinates must be provided for RECT mode")
        cv2.grabCut(input_image, grabcut_mask, rect, bgdModel, fgdModel, iter, cv2.GC_INIT_WITH_RECT)

    elif mode == 'MASK':
        cv2.grabCut(input_image, grabcut_mask, None, bgdModel, fgdModel, iter, cv2.GC_INIT_WITH_MASK)

    elif mode == 'MASK+RECT':
        if rect is None:
            raise ValueError("Rectangle coordinates must be provided for MASK+RECT mode")
        cv2.grabCut(input_image, grabcut_mask, rect, bgdModel, fgdModel, iter, cv2.GC_INIT_WITH_MASK + cv2.GC_INIT_WITH_RECT)

    elif mode == 'EVAL':
        cv2.grabCut(input_image, grabcut_mask, None, bgdModel, fgdModel, iter, cv2.GC_EVAL)

    else:
        raise ValueError("Invalid mode selected. Choose from 'RECT', 'MASK', 'MASK+RECT', 'EVAL'.")

    _, mask_image = cv2.threshold(grabcut_mask, 2, 255, cv2.THRESH_BINARY)
    mask_expanded = cv2.merge((mask_thresh, mask_thresh, mask_thresh))
    maskedImage = cv2.bitwise_and(input_image, mask_expanded)

    return maskedImage,mask_image  