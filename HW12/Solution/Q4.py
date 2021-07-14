from sklearn.metrics.pairwise import euclidean_distances

def calculate_eccentricity(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    cnt = contours[0]

    for c in contours:
        area = cv2.contourArea(c)
        if area > cv2.contourArea(cnt):
            cnt = c

    leftmost = cnt[cnt[:,:,0].argmin()][0].reshape((1, -1))
    rightmost = cnt[cnt[:,:,0].argmax()][0].reshape((1, -1))
    topmost = cnt[cnt[:,:,1].argmin()][0].reshape((1, -1))
    bottommost = cnt[cnt[:,:,1].argmax()][0].reshape((1, -1))

    h_axis = euclidean_distances(leftmost, rightmost)
    v_axis = euclidean_distances(topmost, bottommost)

    major_axis = max(h_axis, v_axis)
    minor_axis = min(h_axis, v_axis)

    if major_axis == 0:
        eccentricity = 1
    else:
        eccentricity = np.sqrt(1 - np.power( np.divide(minor_axis, major_axis) ,2) )

    return eccentricity

def classify_leaf(image):
    '''
    Classifies the input image to only two classes of leaves.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        int: The class of the image. 1 == apple, 0 == linden
    '''
    eccentricity = calculate_eccentricity(image)

    if eccentricity < 0.7:
        leaf_type = 1
    else:
        leaf_type = 0

    return leaf_type
