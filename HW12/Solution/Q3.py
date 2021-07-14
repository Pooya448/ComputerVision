from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import euclidean_distances

np.seterr(divide='ignore', invalid='ignore')

def extract_shape_desc_vec(image):

    features = []

    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    for c in contours:
        area = cv2.contourArea(c)
        if area > cv2.contourArea(cnt):
            cnt = c

    area = cv2.contourArea(cnt)
    primeter = cv2.arcLength(cnt, True)
    p2 = np.square(primeter)

    if not np.isfinite(p2):
        compactness = 0
    else:
        compactness = np.divide((4 * np.pi * area), (np.square(primeter)))

    features.append(np.nan_to_num(compactness))

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        solidity = 0
    else:
        solidity = float(area)/hull_area
    features.append(solidity)

    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if not np.isfinite(aspect_ratio):
        aspect_ratio = 0
    features.append(aspect_ratio)

    leftmost = cnt[cnt[:,:,0].argmin()][0].reshape((1, -1))
    rightmost = cnt[cnt[:,:,0].argmax()][0].reshape((1, -1))
    topmost = cnt[cnt[:,:,1].argmin()][0].reshape((1, -1))
    bottommost = cnt[cnt[:,:,1].argmax()][0].reshape((1, -1))

    features.append(leftmost[0, 0])
    features.append(leftmost[0, 1])


    features.append(rightmost[0, 0])
    features.append(rightmost[0, 1])


    features.append(topmost[0, 0])
    features.append(topmost[0, 1])


    features.append(bottommost[0, 0])
    features.append(bottommost[0, 1])

    h_axis = euclidean_distances(leftmost, rightmost)
    v_axis = euclidean_distances(topmost, bottommost)

    eccentricity = np.sqrt(1 - np.power( np.divide(min(h_axis, v_axis), max(h_axis, v_axis)) ,2) )
    features.append(np.nan_to_num(eccentricity))

    return features

def get_dataset_shape_descs(dataset):

    out = []

    for i in range(dataset.shape[0]):
        slice = dataset[i, :, :]
        vec = extract_shape_desc_vec(slice)
        out.append(vec)

    final_vecs = np.array(out, dtype=np.float64)

    return final_vecs

def lbp_dataset(dataset, skimage):
    result = []

    for i in range(dataset.shape[0]):
        if skimage:
            hist, _ = np.histogram(local_binary_pattern(dataset[i, :, :], 8, 1, 'default'), bins=256, density=True)
            result.append(hist)
        else:
            result.append(LBP(dataset[i, :, :]))

        if i%10000 == 0 and (not skimage):
            print("Sample No: ", i)
    return np.array(result)

def get_dataset_hogs(dataset):
    out = []

    for i in range(dataset.shape[0]):
        slice = dataset[i, :, :]
        vec = hog(slice, orientations=8, pixels_per_cell=(3, 3), cells_per_block=(8, 8), visualize=False, feature_vector=True, multichannel=False)
        out.append(vec)

    final_vecs = np.array(out, dtype=np.float64)

    return final_vecs

def classify_hog(x_train, y_train, x_test):
    '''
    Classifies images with HOG.

    Parameters:
        x_train(numpy.ndarray) : train data
        y_train(numpy.ndarray) : train labels
        x_test(numpy.ndarray) : test data

    outputs:
        predicted_labels(numpy.ndarray): labels that predicted
    '''

    x_train_vecs = get_dataset_hogs(x_train)

    x_test_vecs = get_dataset_hogs(x_test)

    classifier = LinearSVC(multi_class='ovr', max_iter=1000)
    classifier.fit(x_train_vecs, y_train)

    prediction = classifier.predict(x_test_vecs)

    return prediction

def classify_shape_desc(x_train, y_train, x_test):
    '''
    Classifies images by using shape descriptors.

    Parameters:
        x_train(numpy.ndarray) : train data
        y_train(numpy.ndarray) : train labels
        x_test(numpy.ndarray) : test data

    outputs:
        predicted_labels(numpy.ndarray): labels that predicted
    '''

    x_train_vecs = get_dataset_shape_descs(x_train)
    x_test_vecs = get_dataset_shape_descs(x_test)

    classifier = LinearSVC(max_iter=1000)
    classifier.fit(x_train_vecs, y_train)

    prediction = classifier.predict(x_test_vecs)

    return prediction

def LBP(img):
    '''
    Extracts LBP features from the input image.

    Parameters:
        img(numpy.ndarray) : image data
    outputs:
        output: LBP features
    '''
    LBP_coded = np.zeros_like(img)

    window = 3

    for i in range(img.shape[0] - window):
        for j in range(img.shape[1] - window):

            win = img[i : i + window, j : j + window]

            win_center = win[1, 1]
            win_coded = (win >= win_center) * 1

            ### Flattening array and removing the center
            win_binary = np.delete(win_coded.T.flatten(), 4)

            ### Getting coressponding indices of 1 values within the array
            indices = np.where(win_binary)[0]
            if indices.any():
                lbp_val = np.sum(np.power(2, indices))
            else:
                lbp_val = 0

            LBP_coded[i + 1, j + 1] = lbp_val

    histogram, _ = np.histogram(LBP_coded, bins=256, density=True)
    return histogram

def classify_your_lbp(x_train, y_train, x_test):
    '''
    Classifies images by using your LBP.

    Parameters:
        x_train(numpy.ndarray) : train data
        y_train(numpy.ndarray) : train labels
        x_test(numpy.ndarray) : test data

    outputs:
        predicted_labels(numpy.ndarray): labels that predicted
    '''

    x_train_vecs = lbp_dataset(x_train, False)
    x_test_vecs = lbp_dataset(x_test, False)

    classifier = LinearSVC(max_iter=1000)
    classifier.fit(x_train_vecs, y_train)

    prediction = classifier.predict(x_test_vecs)

    return prediction

def classify_skimage_lbp(x_train, y_train, x_test):
    '''
    Classifies images by using Scikit-Image LBP.

    Parameters:
        x_train(numpy.ndarray) : train data
        y_train(numpy.ndarray) : train labels
        x_test(numpy.ndarray) : test data

    outputs:
        predicted_labels(numpy.ndarray): labels that predicted
    '''

    x_train_vecs = lbp_dataset(x_train, True)
    x_test_vecs = lbp_dataset(x_test, True)

    classifier = LinearSVC(max_iter=2000)
    classifier.fit(x_train_vecs, y_train)

    prediction = classifier.predict(x_test_vecs)

    return prediction    
