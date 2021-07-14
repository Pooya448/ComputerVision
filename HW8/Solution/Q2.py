def stitch(image1, image2):
    '''
    Creates panorama image of two inputs.
    
    Parameters:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
    
    Returns:
        numpy.ndarray: The result panorama image.
    '''

    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch([image1, image2])

    
    return stitched

