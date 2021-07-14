import numpy.ma as ma

def make_slice(image, offset, i, j):
    return image[i - offset : i + offset + 1, j - offset : j + offset + 1]

def get_indices(shape, offset):
    indices = []
    for val in range(shape):
        if val - offset >= 0 and val + offset + 1 <= shape:
            indices.append(val)
    return indices

def crop_padding(image, offset):
    return image[offset : image.shape[0] - offset, offset : image.shape[1] - offset].astype(np.uint8)

def gray_morphology(image, element, method):
    
    offset = element.shape[0] // 2
    
    rows = get_indices(image.shape[0], offset)
    cols = get_indices(image.shape[1], offset)
    
    result_image = np.zeros(image.shape[:2])
    
    for i in rows:
        for j in cols:

            image_slice = ma.masked_array(make_slice(image, offset, i, j), mask = np.logical_not(element))
            
            if method == 'ERODE':
                result_image[i, j] = image_slice.min()
            elif method == 'DILATE':
                result_image[i, j] = image_slice.max()
                
    return result_image

structuring_element = np.ones((3, 3))

def RGB_dilate(image, structuring_element):
    '''
    Applies dilation in RGB space.
    
    Parameters:
        image (numpy.ndarray): The input image.
        structuring_element (numpy.ndarray): The structuring element must be square.
    
    Returns:
        dilated_image (numpy.ndarray): The dilated result image.   
    '''
    offset = structuring_element.shape[0] // 2
    
    padded = cv2.copyMakeBorder(image, offset, offset, offset, offset, cv2.BORDER_REFLECT)
    
    channels = cv2.split(padded)

    result = []
    for ch in channels:
        result.append(gray_morphology(ch, structuring_element, 'DILATE'))
    
    dilated_image = crop_padding(cv2.merge(tuple(result)), offset)
    return dilated_image

def RGB_erode(image, structuring_element):
    '''
    Applies erosion in RGB space.
    
    Parameters:
        image (numpy.ndarray): The input image.
        structuring_element (numpy.ndarray): The structuring element must be square.
    
    Returns:
        eroded_image (numpy.ndarray): The eroded result image.   
    '''
    offset = structuring_element.shape[0] // 2
    
    padded = cv2.copyMakeBorder(image, offset, offset, offset, offset, cv2.BORDER_REFLECT)
    
    channels = cv2.split(padded)

    result = []
    for ch in channels:
        result.append(gray_morphology(ch, structuring_element, 'ERODE'))
    
    eroded_image = crop_padding(cv2.merge(tuple(result)), offset)
    return eroded_image

