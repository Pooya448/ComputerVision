def harris_points(image):
    '''
    Gets corner points by applying the harris detection algorithm.
    
    Parameters:
        image (numpy.ndarray): The input image.
    
    Returns:
        numpy.ndarray: The result image.
    '''    
    out_image = image.copy()
    
    threshold = 0.5
    K = 0.05
    window_size = 3
    
    sum_kernel = np.ones((window_size, window_size))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    Ix, Iy = np.gradient(gray)
    
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = Ix * Iy
    
    SIxx = cv2.filter2D(Ixx, -1, sum_kernel)
    SIyy = cv2.filter2D(Iyy, -1, sum_kernel)
    SIxy = cv2.filter2D(Ixy, -1, sum_kernel)
    
    result = (SIxx * SIyy) - np.square(SIxy) - (K * np.square(SIxx + SIyy))
    
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX)
    
    rows, cols = np.where(result >= threshold)
    points = list(zip(rows, cols))

    for point in points:
        cv2.circle(out_image, point, 5, (255, 0, 0), -1)
    
    return out_image

