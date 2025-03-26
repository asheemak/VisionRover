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
	labels = np.array(labels)
	
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



def SVM(modelType=0, type=cv2.ml.SVM_C_SVC, kernelType=cv2.ml.SVM_RBF, classWeights=None, C=1, coef=0, degree=1, gamma=1, nu=0.1, p=0.1):
  
	svm_model = cv2.ml.SVM_create()

	if modelType == 0: # Classfiers
		if type not in [cv2.ml.SVM_C_SVC, cv2.ml.SVM_NU_SVC, cv2.ml.SVM_ONE_CLASS]:
			raise ValueError("This type is not related to Classification.")
		
	elif modelType == 1: # Regresors		
		if type not in [cv2.ml.SVM_NU_SVR, cv2.ml.SVM_EPS_SVR]:
			raise ValueError("This type is not related to Regression.")

	svm_model.setType(type)
	svm_model.setKernel(kernelType)
	
	if type == cv2.ml.SVM_C_SVC:
		if classWeights is not None:
			svm_model.setClassWeights(classWeights)
		
	if type in [cv2.ml.SVM_C_SVC, cv2.ml.SVM_EPS_SVR, cv2.ml.SVM_NU_SVR]:
		svm_model.setC(C)

	if type in [cv2.ml.SVM_NU_SVR, cv2.ml.SVM_NU_SVC, cv2.ml.SVM_ONE_CLASS]:
		svm_model.setNu(nu)

	elif type == cv2.ml.SVM_EPS_SVR:
		svm_model.setP(p)

	if kernelType in [cv2.ml.SVM_POLY, cv2.ml.SVM_SIGMOID]:
		svm_model.setCoef0(coef)

	if kernelType == cv2.ml.SVM_POLY:
		svm_model.setDegree(degree)

	if kernelType in [cv2.ml.SVM_POLY, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_RBF, cv2.ml.SVM_CHI2]:
		svm_model.setGamma(gamma)
	
	svm_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
	 
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


def loadImage(imagePath, colorConversion=-1):
	image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)

	if image is None:
		raise ValueError("Failed to load and decode Image")
	
	if colorConversion != -1:
		image = cv2.cvtColor(image, colorConversion)
	
	return image



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
	script_file = sys.modules['__main__'].__file__
	script_dir = os.path.dirname(os.path.abspath(script_file))
	files = glob.glob(os.path.join(script_dir, imagePath), recursive=True)
	return [loadImage(file, colorConversion=colorConversion) for file in files]


def fastFourierTransform(src, flags=cv2.DFT_COMPLEX_OUTPUT):
	dft = cv2.dft(src.astype(np.float32), flags=flags)
	dft_shift = np.fft.fftshift(dft)

	if len(dft_shift.shape) == 3:
		spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
	elif len(dft_shift.shape) == 2:
		spectrum = 20 * np.log(dft_shift)
	else:
		spectrum = dft_shift

	return dft, dft_shift, spectrum


def filterFft(src, customFilter=False, filter=None, filterType=0, kernelType=0, lowerBound=30, upperBound=60):
	rows, cols, _ = src.shape
	crow, ccol = rows // 2 , cols // 2  

	mask = np.ones((rows, cols, 2), np.float32)

	if customFilter:
		mask[:, :, 0] = filter
		mask[:, :, 1] = filter
	
	else:
		if kernelType == 0: # gaussian
			lp_kernel = cv2.getGaussianKernel(upperBound, -1)
			lp_kernel = lp_kernel * lp_kernel.T
			hp_kernel = cv2.getGaussianKernel(lowerBound, -1)
			hp_kernel = hp_kernel * hp_kernel.T

		elif kernelType == 1: # box
			lp_kernel = np.ones((upperBound, upperBound), np.float32) / (upperBound * upperBound)
			hp_kernel = np.ones((lowerBound, lowerBound), np.float32) / (lowerBound * lowerBound)

		elif kernelType == 2: # hamming
			lp_kernel = np.hamming(upperBound)[:, None] * np.hamming(upperBound)
			hp_kernel = np.hamming(lowerBound)[:, None] * np.hamming(lowerBound)

		elif kernelType == 3: # hanning
			lp_kernel = np.hanning(upperBound)[:, None] * np.hanning(upperBound)
			hp_kernel = np.hanning(lowerBound)[:, None] * np.hanning(lowerBound)

		elif kernelType == 4: # circle
			lp_kernel = np.zeros((upperBound, upperBound), np.float32)
			hp_kernel = np.zeros((lowerBound, lowerBound), np.float32)
			cv2.circle(lp_kernel, (upperBound//2, upperBound//2), upperBound//2, 1, -1)
			cv2.circle(hp_kernel, (lowerBound//2, lowerBound//2), lowerBound//2, 1, -1)
		

		lp_kernel = lp_kernel / np.max(lp_kernel)  
		hp_kernel = hp_kernel / np.max(hp_kernel)

		if filterType == 0: # low pass
			lp_mask_full = np.zeros((rows, cols), np.float32)
			lp_mask_full[crow-upperBound//2:crow+upperBound//2, ccol-upperBound//2:ccol+upperBound//2] = lp_kernel
			mask[:, :, 0] = lp_mask_full
			mask[:, :, 1] = lp_mask_full

		elif filterType == 1: # high pass
			hp_mask_full = np.ones((rows, cols), np.float32)
			hp_mask_full[crow-lowerBound//2:crow+lowerBound//2, ccol-lowerBound//2:ccol+lowerBound//2] = 1 - hp_kernel
			mask[:, :, 0] = hp_mask_full
			mask[:, :, 1] = hp_mask_full

		elif filterType == 2: # band pass
			lp_mask_full = np.zeros((rows, cols), np.float32)
			hp_mask_full = np.ones((rows, cols), np.float32)
			lp_mask_full[crow-upperBound//2:crow+upperBound//2, ccol-upperBound//2:ccol+upperBound//2] = lp_kernel
			hp_mask_full[crow-lowerBound//2:crow+lowerBound//2, ccol-lowerBound//2:ccol+lowerBound//2] = 1 - hp_kernel

			band_pass_mask = lp_mask_full * hp_mask_full
			mask[:, :, 0] = band_pass_mask
			mask[:, :, 1] = band_pass_mask
		
		elif filterType == 3: # stop pass
			lp_mask_full = np.zeros((rows, cols), np.float32)
			hp_mask_full = np.ones((rows, cols), np.float32)
			lp_mask_full[crow-upperBound//2:crow+upperBound//2, ccol-upperBound//2:ccol+upperBound//2] = lp_kernel
			hp_mask_full[crow-lowerBound//2:crow+lowerBound//2, ccol-lowerBound//2:ccol+lowerBound//2] = 1 - hp_kernel

			stop_band_mask = 1 - (lp_mask_full * hp_mask_full)
			mask[:, :, 0] = stop_band_mask
			mask[:, :, 1] = stop_band_mask

	dst = src * mask

	return dst, mask[:, :, 0]


def inverseFft(src):
	f_ishift = np.fft.ifftshift(src)
	dst = cv2.idft(f_ishift)
	dst = cv2.magnitude(dst[:, :, 0], dst[:, :, 1])

	dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
	dst = np.uint8(dst)

	return dst


def scanBarcode(image):
	instance = cv2.barcode.BarcodeDetector()
	retval, points = instance.detectMulti(image)

	if retval:
		ret, decoded_info, _ = instance.decodeMulti(image, points)

		if len(image.shape) == 2 or image.shape[2] == 1:
			img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		else:
			img = image.copy()
			
		img = cv2.polylines(img, points.astype(int), True, (0, 255, 0), 2)
		for s, p in zip(decoded_info, points):
				img = cv2.putText(img, s, p[1].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
		
		return points, decoded_info, img
	
	return [], (), image


def scanQRCode(image):
	instance = cv2.QRCodeDetector()
	retval, points = instance.detectMulti(image)

	if retval:

		ret, decoded_info, _ = instance.decodeMulti(image, points)

		if len(image.shape) == 2 or image.shape[2] == 1:
			img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		else:
			img = image.copy()
				  
		img = cv2.polylines(img, points.astype(int), True, (0, 255, 0), 2)
		for s, p in zip(decoded_info, points):
			img = cv2.putText(img, s, p[0].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
		
		return points, decoded_info, img
	
	return [], (), image


def calibrateCameraChessboard(images,
 								patternSize=(6, 9),
								criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
								winSize=(5, 5),
								zeroZone=(-1, -1)):

	threedpoints = [] 
	twodpoints = [] 

	objectp3d = np.zeros((1, patternSize[0] * patternSize[1], 3), np.float32) 
	objectp3d[0, :, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2) 

	overlay = []

	for image in images: 
		grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
		ret, corners = cv2.findChessboardCorners(grayColor, patternSize, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) 

		if ret == True: 
			threedpoints.append(objectp3d) 
			corners2 = cv2.cornerSubPix(grayColor, corners, winSize, zeroZone, criteria) 
			twodpoints.append(corners2) 
			output = cv2.drawChessboardCorners(image.copy(), patternSize, corners2, ret) 
			
		overlay.append(output)
		  
	ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None) 
	return ret, camera_matrix, dist_coeffs, rvecs, tvecs, overlay


def pruning(binaryImage, pruneLength=2):
	thinned_image = cv2.ximgproc.thinning(binaryImage)
	pruned_image = thinned_image.copy()
	kernel = np.ones((3, 3), dtype=np.uint8)
	
	while True:
		neighbor_count = cv2.filter2D((pruned_image > 0).astype(np.uint8), -1, kernel)
		prune_mask = (pruned_image == 255) & (neighbor_count <= pruneLength)
		
		if not np.any(prune_mask):
			break
		
		pruned_image[prune_mask] = 0

	return pruned_image, thinned_image


def mser(image, delta=5, minArea=60, maxArea=14400, maxVariation=0.25, minDiversity=0.2, maxEvolution=200, areaThresh=1.01, minMargin=0.003, edgeBlurSize=5):

	mser = cv2.MSER_create(delta=delta,
						   min_area=minArea,
						   max_area=maxArea,
						   max_variation=maxVariation,
						   min_diversity=minDiversity,
						   max_evolution=maxEvolution,
						   area_threshold=areaThresh,
						   min_margin=minMargin,
						   edge_blur_size=edgeBlurSize)

	msers, bboxes = mser.detectRegions(image)
	return msers, bboxes



def emCluster(nClusters=2, covarianceMatrixType=cv2.ml.EM_COV_MAT_SPHERICAL, termCriteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6)):
	em_model = cv2.ml.EM_create()

	em_model.setClustersNumber(nClusters)
	em_model.setCovarianceMatrixType(covarianceMatrixType)
	em_model.setTermCriteria(termCriteria)

	return em_model


def mlp(layerSizes=[5, 10, 1], activation=cv2.ml.ANN_MLP_SIGMOID_SYM, alpha=0.1, maxIter=10000):
	
	mlp_model = cv2.ml.ANN_MLP_create()
	mlp_model.setLayerSizes(np.array(layerSizes, dtype=np.int32))
	mlp_model.setActivationFunction(activation)
	mlp_model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, alpha)
	mlp_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, maxIter, 1e-6))
	
	return mlp_model


def copyImage(image):
	if isinstance(image, np.ndarray) and image.size > 0 and (len(image.shape) == 2 or (len(image.shape) == 3 and (image.shape[2] == 1 or image.shape[2] == 3 or image.shape[2] == 4))) and not np.issubdtype(image.dtype, np.str_):
		img = image.copy()
		return img
	else:
		raise ValueError("the input parameter is not an image")

def simpleBlobDetector(image, mask=None, threshold=(50.0, 220.0), thresholdStep=10.0, minDistBetweenBlobs=10.0, minRepeatability=2, filterByColor=False, blobColor=0, filterByArea=False, area=(25.0, 5000.0), filterByCircularity=False, circularity=(0.800000011920929, 3.4028234663852886e+38), filterByConvexity=False, convexity=(0.949999988079071, 3.4028234663852886e+38), filterByInertia=False, inertiaRatio=(0.10000000149011612, 3.4028234663852886e+38)):
	
	params = cv2.SimpleBlobDetector_Params()
	
	params.minThreshold = threshold[0]
	params.maxThreshold = threshold[1]
	params.thresholdStep = thresholdStep
	params.minDistBetweenBlobs = minDistBetweenBlobs
	params.minRepeatability = minRepeatability
	params.collectContours = True

	params.filterByColor = filterByColor
	params.blobColor = blobColor

	params.filterByArea = filterByArea
	params.minArea = area[0]
	params.maxArea = area[1]
	
	params.filterByCircularity = filterByCircularity
	params.minCircularity = circularity[0]
	params.maxCircularity = circularity[1]

	params.filterByConvexity = filterByConvexity
	params.minConvexity = convexity[0]
	params.maxConvexity = convexity[1]

	params.filterByInertia = filterByInertia
	params.minInertiaRatio = inertiaRatio[0]
	params.maxInertiaRatio = inertiaRatio[1]

	detector = cv2.SimpleBlobDetector_create(params)

	keypoints = detector.detect(image, mask=mask)

	contours = detector.getBlobContours()

	return keypoints, contours


def loadVideo(videoPath, startFrame=0, endFrame=1, numberOfFrames=2, colorConversion=-1):
	if numberOfFrames < 2:
		raise ValueError("numberOfFrames must be at least 2.")
	if endFrame <= startFrame:
		raise ValueError("endFrame must be greater than startFrame.")
	
	videoPath = __normalizeFilePath(videoPath)
	cap = cv2.VideoCapture(videoPath)
	if not cap.isOpened():
		raise ValueError("Unable to open video.")
	
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if endFrame >= total_frames:
		cap.release()
		raise ValueError(f"endFrame is greater than the max frame ({total_frames-1}) in the video.")

	available_frames = endFrame - startFrame + 1
	if available_frames < numberOfFrames:
		cap.release()
		raise ValueError(f"Not enough frames between startFrame ({startFrame}) and endFrame ({endFrame}). Requested {numberOfFrames} frames, but only {available_frames} frames are available.")

	indices = np.linspace(startFrame, endFrame, numberOfFrames, dtype=int)
	
	frames = []
	for idx in indices:
		cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
		ret, frame = cap.read()
		if not ret or frame is None:
			cap.release()
			raise ValueError(f"Failed to load and decode frame at index {idx}")

		if len(frame.shape) > 2:
			if frame.shape[2] == 3:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			elif frame.shape[2] == 4:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
		
		if colorConversion != -1:
			if (len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1)):
				if colorConversion not in [cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2RGB, cv2.COLOR_GRAY2BGRA, cv2.COLOR_GRAY2RGBA]:
					cap.release()
					raise ValueError("This color conversion is not supported for 1-channel images")
			else:
				if colorConversion in [cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2RGB, cv2.COLOR_GRAY2BGRA, cv2.COLOR_GRAY2RGBA]:
					cap.release()
					raise ValueError(f"This color conversion is not supported for {frame.shape[2]}-channel images")
				
			frame = cv2.cvtColor(frame, colorConversion)
		
		frames.append(frame)

	cap.release()
	
	return frames


def knn(modelType=0, k=10):
	knn_model = cv2.ml.KNearest_create()
	
	knn_model.setDefaultK(k)
	
	if modelType == 0:
		isClassifier = True
	elif modelType == 1:
		isClassifier = False
		
	knn_model.setIsClassifier(isClassifier)
	knn_model.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE)
	return knn_model