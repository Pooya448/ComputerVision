def make_slice(image, offset, i, j):
    return image[i - offset : i + offset + 1, j - offset : j + offset + 1]

def get_indices(shape, offset):
    indices = []
    for val in range(shape):
        if val - offset >= 0 and val + offset + 1 < shape:
            indices.append(val)
    return indices

structuring_element = np.ones((25, 25))

def your_dilate(image, structuring_element):
    '''
    Applies your dilation.
    
    Parameters:
        image (numpy.ndarray): The input image.
        structuring_element (numpy.ndarray): The structuring element must be square.
    
    Returns:
        numpy.ndarray: The result image.
    '''

    
    offset = structuring_element.shape[0] // 2
    
    rows = get_indices(image.shape[0], offset)
    cols = get_indices(image.shape[1], offset)
    
    dilated_image = np.zeros_like(image)
    
    for i in rows:
        for j in cols:
            image_slice = make_slice(image, offset, i, j)
            if np.logical_and(structuring_element, image_slice).any():
                dilated_image[i, j] = 1

    return dilated_image

def your_erode(image, structuring_element):
    '''
    Applies your erosion.
    
    Parameters:
        image (numpy.ndarray): The input image.
        structuring_element (numpy.ndarray): The structuring element must be square.
    
    Returns:
        numpy.ndarray: The result image.
    '''
    offset = structuring_element.shape[0] // 2
    
    rows = get_indices(image.shape[0], offset)
    cols = get_indices(image.shape[1], offset)
    
    eroded_image = np.zeros_like(image)
    
    for i in rows:
        for j in cols:
            image_slice = make_slice(image, offset, i, j)
            logical_and = np.logical_and(structuring_element, image_slice)

            if np.array_equal(logical_and, structuring_element):
                eroded_image[i, j] = 1
                        
    return eroded_image

def cv_dilate(image, structuring_element):
    '''
    Applies OpenCV dilation.
    
    Parameters:
        image (numpy.ndarray): The input image.
        structuring_element (numpy.ndarray): The structuring element must be square.
    
    Returns:
        numpy.ndarray: The result image.
    '''
    
    return cv2.dilate(image, structuring_element, iterations=1)  

def cv_erode(image, structuring_element):
    '''
    Applys OpenCV erosion.
    
    Parameters:
        image (numpy.ndarray): The input image.
        structuring_element (numpy.ndarray): The structuring element must be square.
    
    Returns:
        numpy.ndarray: The result image.
    '''
    
    return cv2.erode(image, structuring_element, iterations=1)  

