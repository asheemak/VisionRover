import cv2
import numpy as np
import glob
import os
import sys
import re

def loadMlModel(filePath):
    filePath = __normalizeFilePath(filePath)
    loader_mapping = {
        "opencv_ml_svm": cv2.ml.SVM.load,
        "opencv_ml_rtrees": cv2.ml.RTrees.load,
        "opencv_ml_knn": cv2.ml.KNearest.load,
        "opencv_ml_ann_mlp": cv2.ml.ANN_MLP.load,
        "opencv_ml_boost": cv2.ml.Boost.load,
        "opencv_ml_logistic_regression": cv2.ml.LogisticRegression.load,
        "opencv_ml_normal_bayes_classifier": cv2.ml.NormalBayesClassifier.load,
        "opencv_ml_em": cv2.ml.EM.load,
    }

    fs = cv2.FileStorage(filePath, cv2.FILE_STORAGE_READ)

    if not fs.isOpened():
        raise ValueError("Failed to open pretrained model")
    
    try:
        model_type_key = None
        for key in loader_mapping:
            node = fs.getNode(key)
            if not node.empty():
                model_type_key = key
                break
    finally:
        fs.release()

    if not model_type_key:
        raise ValueError("Pretrained model not detected")
    
    model = loader_mapping[model_type_key](filePath)

    return model

def loadCsv(filePath):
    filePath = __normalizeFilePath(filePath)
    import pandas as pd
    df = pd.read_csv(filePath)
    return df
	
def __normalizeFilePath(path: str):
    script_file = sys.modules['__main__'].__file__
    script_dir = os.path.dirname(os.path.abspath(script_file))

    if not os.path.isabs(path):
        path = os.path.join(script_dir, path)

    return path

def loadDirectoryEntriesInfo(directoryPath):

    if directoryPath.endswith(os.path.sep) or directoryPath.endswith("/"):
        directoryPath = directoryPath + "*"

    directoryPath = re.sub(r'(?<!\*)\*(?!\*)', '**', directoryPath)

    import pandas as pd
    from datetime import datetime

    directoryPath = __normalizeFilePath(directoryPath)
    paths = glob.glob(directoryPath, recursive=True)
    paths = [p for p in paths if os.path.isfile(p)]
    files_infos = []
    paths = sorted(paths)
    for path in paths:
        filename = os.path.basename(path)
        file_size = os.path.getsize(path)
        last_modified = os.path.getmtime(path)
        last_modified_date = datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')
        folder_name = os.path.dirname(path)
        _, file_extension = os.path.splitext(path)
        
        files_infos.append({
            'fileName' :  filename,
            'folderName' : folder_name,
            'size' : file_size,
            'lastModifiedDate': last_modified_date,
            'path' : path,
            'extension' : file_extension
        })

    return pd.DataFrame(files_infos)
	
def loadImage(imagePath, colorConversion=-1):

    imagePath = __normalizeFilePath(imagePath)

    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Failed to load and decode Image")
    
    if len(image.shape) > 2 and image.shape[2] == 3: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    elif len(image.shape) > 2 and image.shape[2] == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    

    if colorConversion != -1:
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            if colorConversion not in [cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2RGB, cv2.COLOR_GRAY2BGRA, cv2.COLOR_GRAY2RGBA]:
                raise ValueError("This color convertion is not supported for 1ch image")
        else:
            if colorConversion in [cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2RGB, cv2.COLOR_GRAY2BGRA, cv2.COLOR_GRAY2RGBA]:
                raise ValueError(f"This color convertion is not supported for {image.shape[2]}ch image")
                
        image = cv2.cvtColor(image, colorConversion)

    return image


def loadImages(imagePath: str, colorConversion=-1):

    if imagePath.endswith(os.path.sep) or imagePath.endswith("/"):
        imagePath = imagePath + "*"

    imagePath = re.sub(r'(?<!\*)\*(?!\*)', '**', imagePath)
    imagePath = __normalizeFilePath(imagePath)
    files = glob.glob(imagePath, recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files = sorted(files)
    return [loadImage(file, colorConversion=colorConversion) for file in files]


def loadDicom(file_path):
    import SimpleITK as sitk
    file_path = __normalizeFilePath(file_path)
    def remove_sensitive_data(metadata):
        """Remove personal data from DICOM metadata dictionary."""
        sensitive_keys = [
            "0010|0010",  # Patient's Name
            "0010|0020",  # Patient ID
            "0010|0030",  # Patient's Birth Date
            "0010|0040",  # Patient's Sex
            "0010|1000",  # Other Patient IDs
            "0010|2160",  # Ethnic Group
            "0010|4000"   # Patient Comments
        ]
        for key in sensitive_keys:
            if key in metadata:
                metadata[key] = "Anonymous"  # Replace sensitive data with generic values

    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        image = reader.Execute()
        metadata = {key: image.GetMetaData(key) for key in image.GetMetaDataKeys()}
        remove_sensitive_data(metadata)
        image_array = sitk.GetArrayFromImage(image)
        return image_array[0], metadata
    except Exception as e:
        raise ValueError(f"Could not read DICOM file: {file_path}")

def onnx_model_loader(modelPath):
    import onnxruntime
    options = onnxruntime.SessionOptions()
    
    # Enable all graph optimizations
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Set the number of threads used within a parallel operator
    # sess_options.intra_op_num_threads = 2

    # Set the execution mode to sequential
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    session = onnxruntime.InferenceSession(modelPath, sess_options=options)

    return session

def sam(encoder_session, decoder_session, image, input_point, input_label):
    # Convert input_point (tuple) to NumPy array
    np_input_point = np.array(input_point, dtype=np.float32)[None, :]  # Shape (1, 2)
    np_input_label =  np.array([input_label])
    orig_height, orig_width = image.shape[:2]

    def prepare_inputs_encoder(image, ort_session):
        # Preprocess the image and convert it into a blob
        image = cv2.resize(image, (1024, 1024))
        blob = cv2.dnn.blobFromImage(image, 1/256)
        # Prepare the inputs for the encoder model
        inputs_encoder = {ort_session.get_inputs()[0].name: blob}
        return inputs_encoder

    def prepare_decoder_inputs(image_embedding, input_point, input_label):
        # Prepare the coordinates for the input points, adding a dummy point
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
        # Prepare the labels for the input points, adding a dummy label
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
        # Prepare the inputs for the decoder model
        ort_inputs_decoder = {
            "image_embeddings": image_embedding.astype(np.float32),
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
        }
        return ort_inputs_decoder

    # Prepare the inputs for the encoder
    inputs_encoder = prepare_inputs_encoder(image, encoder_session)
    result_encoder = encoder_session.run(None, inputs_encoder)
    image_embedding = np.array(result_encoder[0])

    # Adjust input_point according to image resizing
    scale_x = 1024 / orig_width
    scale_y = 1024 / orig_height
    np_input_point_scaled = np_input_point * np.array([scale_x, scale_y])

    # Decoder inference
    ort_inputs_decoder = prepare_decoder_inputs(image_embedding, np_input_point_scaled, np_input_label)
    result_decoder = decoder_session.run(None, ort_inputs_decoder)
    low_res_logits, maskss = result_decoder

    # Apply a binary threshold to convert the mask probabilities to a binary mask
    _, binaryMasks = cv2.threshold(maskss[0][2], 0, 255, cv2.THRESH_BINARY)
    # Resize the mask to match the original image dimensions
    binaryMasks = cv2.resize(binaryMasks, (image.shape[1], image.shape[0]))

    return binaryMasks.astype(np.uint8)


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

def laplacianOfGaussian(image, sigma):

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    log_edges = cv2.Laplacian(blurred, cv2.CV_64F)
    log_edges = np.abs(log_edges)
    
    return log_edges

def differenceOfGaussian(image, sigma1, sigma2):
    blurred1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    blurred2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    dog_edges = cv2.absdiff(blurred1 , blurred2)
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


def panoramaStitching(leftView, rightView):
    def __drawInliersOutliers(leftImage, rightImage, srcPoint, dstPoint, mask):

        rows1, cols1, _ = leftImage.shape
        rows2, cols2, _ = rightImage.shape

        match_image = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
        match_image[:rows1, :cols1, :] = np.dstack([leftImage])
        match_image[:rows2, cols1:cols1 + cols2, :] = np.dstack([rightImage])

        match_image2 = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
        match_image2[:rows1, :cols1, :] = np.dstack([leftImage])
        match_image2[:rows2, cols1:cols1 + cols2, :] = np.dstack([rightImage])

        # draw lines
        for i in range(srcPoint.shape[0]):
            x1, y1 = srcPoint[i][0]
            x2, y2 = dstPoint[i][0]

            point1 = (int(x1), int(y1))
            point2 = (int(x2 + leftImage.shape[1]), int(y2))

            if mask[i][0] == 1:
                cv2.line(match_image, point1, point2, (0, 255, 0), 1)
            else :
                cv2.line(match_image2, point1, point2, (255, 0, 0), 1)

        return match_image, match_image2
    


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

    overlay_inlier_img, overlay_outlier_img = __drawInliersOutliers(leftView, rightView, src_pts, dst_pts, mask)

    H_inv = np.linalg.inv(H)

    img = cv2.warpPerspective(rightView, H_inv, (leftView.shape[1] + rightView.shape[1], leftView.shape[0]))

    img[0:leftView.shape[0], 0:leftView.shape[1]] = leftView

    # Remove blank spaces
    gray_stitched = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_stitched, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        img = img[y:y + h, x:x + w-1]
        
    return img, overlay_inlier_img, overlay_outlier_img


def imageNumOfHoles(image):
    all_contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = sum(1 for i in range(len(all_contours)) if hierarchy[0][i][3] != -1)
    return holes

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
    import math
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
    import pywt
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
        raise ValueError("Each sample must have a corresponding label")
    
    return x_train, x_test, y_train, y_test


def trainModel(model, XTrain, yTrain, sampleType=cv2.ml.ROW_SAMPLE):
    XTrain = XTrain.astype(np.float32)
    yTrain = yTrain.astype(np.int32)

    if (sampleType == cv2.ml.ROW_SAMPLE and yTrain.flatten().shape[0] != XTrain.shape[0]) or (sampleType == cv2.ml.COL_SAMPLE and yTrain.flatten().shape[0] != XTrain.shape[1]):
        raise ValueError("Each sample must have a corresponding label")    

    model.train(XTrain, sampleType, yTrain)
    return model



def predict(model, features):
    features = np.array(features, dtype=np.float32)
    
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    elif model.getVarCount() == features.shape[0]:
        features = features.T

    elif model.getVarCount() != features.shape[1]:
        raise ValueError(f"This model expects {model.getVarCount()} features, but the provided data contains only {features.shape[1]} features.")
    
    _, preds = model.predict(features)
    preds = preds.flatten()
    return preds


def evaluateModel(model, features, labels):
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    if labels.flatten().shape[0] == features.shape[0]:
        if model.getVarCount() != features.shape[1]:
            raise ValueError(f"Samples are row-wise but input data does not contain the required number of features. This model expects {model.getVarCount()} features, but the provided data contains only {features.shape[1]} features.")
    
    elif labels.flatten().shape[0] == features.shape[1]:
        if model.getVarCount() != features.shape[0]:
            raise ValueError(f"Samples are column-wise but input data does not contain the required number of features. This model expects {model.getVarCount()} features, but the provided data contains only {features.shape[0]} features.")
        
        features = features.T
        
    else:
        raise ValueError("Each sample must have a corresponding label")  
    
    _, preds = model.predict(features)

    labels = labels.flatten()
    preds = preds.flatten()
    accuracy = np.mean((preds == labels).astype(np.float32)) * 100 
    num_classes = len(np.unique(labels))
    confusion_mx = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for true_label, pred_label in zip(labels.flatten(), preds.flatten()):
        confusion_mx[int(true_label), int(pred_label)] += 1

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


def fitLine(image, points, dist_type, param, reps, aeps, color_line, thickness):
    points = np.array(points, dtype=np.float32)
    line = cv2.fitLine(points, dist_type, param, reps, aeps)
    vx, vy, x, y = line.flatten()
    height, width = image.shape[:2]
    t = max(width, height)  # Extend line beyond the image size
    pt1 = (int(x - vx * t), int(y - vy * t))
    pt2 = (int(x + vx * t), int(y + vy * t))
    cv2.line(image, pt1, pt2, color_line, thickness)

    return image, vx, vy, x, y


def featureSelector(features, indexes):
    if not indexes or len(indexes) == 0:
        return []
    return features[:, indexes]
