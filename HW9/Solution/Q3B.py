def local_otsu(image):
    '''
    Applys local otsu on the input image.
    
    Parameters:
        image (numpy.ndarray): The input image.
    
    Returns:
        numpy.ndarray: The result panorama image.
    '''
    
    result = np.zeros_like(image)
    rows, cols = image.shape
    
    x_mid = rows // 2
    y_mid = cols // 2

    result[:x_mid, :y_mid] = global_otsu(image[:x_mid, :y_mid])
    result[:x_mid, y_mid:] = global_otsu(image[:x_mid, y_mid:])
    result[x_mid:, :y_mid] = global_otsu(image[x_mid:, :y_mid])
    result[x_mid:, y_mid:] = global_otsu(image[x_mid:, y_mid:])
    
    return result

