def process_frame(frame):
    '''
    Converts red circles in the input image to white circles.
    
    Parameters:
        frame (numpy.ndarray): The input frame.
    
    Returns:
        numpy.ndarray: The result output frame.
    '''
    
    result = frame.copy()
    blue, green, red = cv2.split(frame)
    
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    rows, cols = np.where(cr > 150)
    result[rows, cols, :] = 255
    
    return result

