def create_kernels():
    kernel_45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    kernel_135 = np.rot90(kernel_45)
    return kernel_45, kernel_135

def get_45_edges(image):
    '''
    Returns the image which shows the 45-degree edges.
    
    Parameters:
        image (numpy.ndarray): The input image.
    
    Returns:
        edges_45 (numpy.ndarray): The 45-degree edges of input image.
    '''
    kernel, _ = create_kernels()
    
    edges_45 = cv2.filter2D(image, -1, kernel)

    return edges_45

def get_135_edges(image):
    '''
    Returns the image which shows the 135-degree edges.
    
    Parameters:
        image (numpy.ndarray): The input image.
    
    Returns:
        edges_135 (numpy.ndarray): The 135-degree edges of input image.
    '''
    _ , kernel = create_kernels()
    
    edges_135 = cv2.filter2D(image, -1, kernel)
    
    return edges_135

