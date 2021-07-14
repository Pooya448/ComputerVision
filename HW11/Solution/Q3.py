def compute_k(image, element):
    thres = image.copy()
    k = 1
    while cv2.erode(image, element, iterations = k).any():
        k += 1
    return k

def get_skeleton(image):
    """
    Finds the skeleton of the input image.
    
    Parameters:
        image (numpy.ndarray): The input image.
    
    Returns:
        numpy.ndarray: The skeleton image.  
    """
    blur = cv2.GaussianBlur(image, (3, 3) ,0)
    ret, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)
    
    element = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype = np.uint8)
    
    K = compute_k(binary, element)    
    union = np.zeros_like(binary)
    
    for i in range(K):
        
        erosion = cv2.erode(binary, element, iterations = i)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, element)
        partial_result = cv2.subtract(erosion, opening)
        union = cv2.add(union, partial_result)

    return union

