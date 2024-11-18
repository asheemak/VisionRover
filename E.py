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

def laplacianOfGaussian(image, sigma=1):

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    log_edges = cv2.Laplacian(blurred, cv2.CV_64F)
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
    innerRect = inner_rect[1][0]* inner_rect[1][1]
    return innerRectArea



def lineProfile(image, start_point, end_point, lineColor=(0, 0, 255)):
    def get_line_coordinates(start_point, end_point):

        x1, y1 = start_point
        x2, y2 = end_point
        coordinates = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        while True:
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

    def get_line_profile(image, start_point, end_point):

        coordinates = get_line_coordinates(start_point, end_point)
        profile = [image[y, x] for x, y in coordinates if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]]
        return profile, coordinates

    line_thickness = 1
    overlayImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.line(overlayImage, start_point, end_point, lineColor, line_thickness)
    profile, line_coordinates = get_line_profile(image, start_point, end_point)
    lineLength = np.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    lineAngle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0]) * 180 / np.pi
    secondDerivative = np.gradient(np.gradient(profile))

    return overlayImage, profile, secondDerivative, lineLength, lineAngle