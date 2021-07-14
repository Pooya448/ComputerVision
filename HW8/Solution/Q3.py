import dlib

def put_mask(face, mask):
    '''
    Adds mask image on face image.

    Parameters:
        face (numpy.ndarray): face image.
        mask (numpy.ndarray): mask image.

    Returns:
        numpy.ndarray: The result image.
    '''
    image_size = (700, 700)
    result = face.copy()

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    face_detected = detector(gray, 1)

    print("Number of faces detected: ", len(face_detected))

    path = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(path)

    for f in face_detected:
        landmarks = predictor(gray, f)

    face_landmark_indices = [30, 9, 3, 15]
    mask_landmark_coordinates = np.array([
        [620, 102],
        [625, 700],
        [200, 220],
        [986, 209]
    ],np.float32)

    temp_list = []
    for n in face_landmark_indices:
        x = landmarks.part(n - 1).x
        y = landmarks.part(n - 1).y
        temp = np.array([x, y],np.float32)
        temp_list.append(temp)

    face_landmark_coordinates = np.array(temp_list,np.float32)

    mat = cv2.getPerspectiveTransform(mask_landmark_coordinates, face_landmark_coordinates)
    mask_transformed = cv2.warpPerspective(mask, mat, image_size)

    for i in range(face.shape[0]):
        for j in range(face.shape[1]):
            if mask_transformed[i, j].all() > 0:
                result[i, j] = mask_transformed[i, j]

    return result
