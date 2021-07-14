 def compute_histogram(image):
        
    histogram = np.zeros((256), np.int)
    np.add.at(histogram, image, 1)
    return histogram

def global_otsu(image):
    '''
    Applys global otsu on the input image.
    
    Parameters:
        image (numpy.ndarray): The input image.
    
    Returns:
        numpy.ndarray: The result panorama image.
    '''
    
    sigma_optim = 100000000
    threshold = 0
    
    histogram = compute_histogram(image)
    
    total_count = np.sum(histogram)

    for t in range(1, 255):
        P1 = histogram[:t]
        P2 = histogram[t:]
        
        Q1 = np.sum(P1)
        Q2 = np.sum(P2)
        
        W1 = Q1 / total_count
        W2 = Q2 / total_count
                
        V1 = np.var(image[image < t])
        V2 = np.var(image[image >= t])
        
        Sigma = W1 * V1 + W2 * V2
        
        if Sigma < sigma_optim:
            sigma_optim = Sigma
            threshold = t
    
    image = np.where(image < threshold, 0, 255)
    
    return image

