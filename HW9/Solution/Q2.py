def AR(background, image):
    '''
    Adds the input image to the background image properly.
    
    Parameters:
        background (numpy.ndarray) : background image
        image (numpy.ndarray): input image
    
    Returns:
        numpy.ndarray: The result image.
    '''
    result = background.copy()
    
    src_points = np.float32([
        [0, 0],
        [0, 1799],
        [1199, 1799],
        [1199, 0]
    ])
    dst_points = np.float32([
        [148, 105],
        [143, 225],
        [236, 228],
        [237, 105]
    ])
    
    h, status = cv2.findHomography(src_points, dst_points)

    image_perspective = cv2.warpPerspective(image, h, (background.shape[1], background.shape[0]))

    for i in range(background.shape[0]):
        for j in range(background.shape[1]):
            if image_perspective[i, j].all() > 0:
                result[i, j] = image_perspective[i, j]    
    
    return result

