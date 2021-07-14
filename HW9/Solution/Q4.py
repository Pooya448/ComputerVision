def get_keypoints(image):
    
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_harris = cv2.cornerHarris(image_gray, 5, 3, 0.05)
    image_key = cv2.dilate(image_harris, None) 
    key_points = image_key > 0.05 * image_key.max()
    
    return key_points


def clean_keypoints(keypoints, dist_offset, disc_offset):
    
    rows, cols = keypoints.shape
    
    for i in range(rows):
        for j in range(cols):
            if keypoints[i, j]:
                if i + disc_offset >= rows or j + disc_offset >= cols or i - disc_offset < 0 or j - disc_offset < 0:
                    keypoints[i, j] = False
                else:
                    keypoints[i - dist_offset:i + dist_offset + 1, j - dist_offset:j + dist_offset + 1] = False
                    keypoints[i, j] = True
                    
    return keypoints

def ncc(slice1, slice2):
    return cv2.matchTemplate(slice1, slice2, cv2.TM_CCORR_NORMED)

def make_slice(image, i, j, offset):
    return image[i - offset:i + offset + 1, j - offset:j + offset + 1, :]

def ncc_loop(image1, src_keypoints, image2, tar_keypoints, disc_size):
    
    offset = disc_size // 2
    
    correspondence = []

    for src_point in src_keypoints:
        
        current_score = -1
        most_scored = None
        
        for tar_point in tar_keypoints:     
            
            slice1 = make_slice(image1, src_point[0], src_point[1], offset)
            slice2 = make_slice(image2, tar_point[0], tar_point[1], offset)

            ncc_score = ncc(slice1, slice2)
            
            if ncc_score > current_score and ncc_score > 0.85:
                current_score = ncc_score
                most_scored = tar_point
                
        correspondence.append(most_scored)
        
    return src_keypoints, correspondence

def zip_coordinates(keypoints):
    
    rows, cols = np.where(keypoints)
    coords = list(zip(rows, cols))
    
    return coords

def draw_combined_lines(srcs, dsts, image, width_to_add):
    for src_point, tar_point in zip(srcs, dsts):
        
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)

        src_inverted = (src_point[1], src_point[0])
        
        tar_inverted = (tar_point[1] + width_to_add, tar_point[0])
        
        cv2.line(image, src_inverted, tar_inverted, (r, g, b), 2)
    return image

def find_match(image1, image2):
    '''
    Finds match points between two input images.
    
    Parameters:
        image1 (numpy.ndarray): input image.
        image2 (numpy.ndarray): second input image.
    
    Returns:
        numpy.ndarray: The result image.
    '''
    
    distance_threshold = 15
    dist_offset = distance_threshold // 2
    
    discriptor_size = 17
    disc_offset = discriptor_size // 2
    
    keypoints1 = get_keypoints(image1)
    keypoints2 = get_keypoints(image2)
    
    keypoints1 = clean_keypoints(keypoints1, dist_offset, disc_offset)
    keypoints2 = clean_keypoints(keypoints2, dist_offset, disc_offset)
    
    src_coordinates = zip_coordinates(keypoints1)
    tar_coordinates = zip_coordinates(keypoints2)
    
    srcs, dsts = ncc_loop(image1, src_coordinates, image2, tar_coordinates, discriptor_size)

    combined_image = np.concatenate([image1, image2], axis = -2)
    
    final_image = draw_combined_lines(srcs, dsts, combined_image, image2.shape[1])
    
    return final_image

