from scipy.spatial import distance as dist

def detect_shape(contour):
    shape = "unidentified"
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
    
    if len(approx) == 3:
        shape = 'triangle'
        
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ratio = w / float(h)
        
        if ratio >= 0.95 and ratio <= 1.05:
            shape = 'square'
        else:
            shape = 'rectangle'
        
    elif len(approx) == 5:
        shape = 'pentagon'
            
    else:
        shape = 'circle'
        
    return shape

def detect_color(image, contour):
    
    colors = np.array([
        [[255, 0, 0]],
        [[0, 255, 0]],
        [[0, 0, 255]],
    ], dtype="uint8")
    
    color_names = ['red', 'green', 'blue']
    
    lab = cv2.cvtColor(colors, cv2.COLOR_RGB2LAB)
        
    mask = mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.erode(mask, None, iterations=1)
        
    mean = cv2.mean(image, mask=mask)[:3]
    
    min_dist = (np.inf, None)
    
    for (i, row) in enumerate(lab):
        
        d = dist.euclidean(row[0], mean)
        
        if d < min_dist[0]:
            min_dist = (d, i)
            
    return color_names[min_dist[1]]

def detect_shape_color(image):
    '''
    Detects shapes and their color in the input image.
    
    Parameters:
        image (numpy.ndarray): The input image.
    
    Returns:
        numpy.ndarray: The result image.
    '''
    
    result = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

    for contour in contours:
        if cv2.contourArea(contour) > 0:

            M = cv2.moments(contour)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            shape = detect_shape(contour)

            color = detect_color(image, contour)

            text = color + " " + shape
            cv2.putText(image, text, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

