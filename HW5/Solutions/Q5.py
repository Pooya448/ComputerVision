def gaussian(x, y, std):
    return (1 / (2 * np.pi * np.square(std))) * (np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(std))))

def gaussian_kernel(size, sigma=1):
    '''
    Calculates and Returns Gaussian kernel.

    Parameters:
        size (int): size of kernel.
        sigma(float): standard deviation of gaussian kernel

    Returns:
        gaussian: A 2d array shows gaussian kernel
    '''
    kernel = np.zeros((size,size), np.float)

    zero_based_gap = np.floor(size / 2)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = i - zero_based_gap
            y = j - zero_based_gap
            kernel[i, j] = gaussian(x, y, sigma)

    return kernel

def sobel_filters(image):
    '''
    finds the magnitude and orientation of the image using Sobel kernels.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        (magnitude, theta): A tuple consists of magnitude and orientation of the image gradients.
    '''
    #Writer your code here

    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = Kx.T

#     out_y_imp = cv2.filter2D(img, -1, kernel)


    Ix = cv2.filter2D(image, -1, Kx)
    Iy = cv2.filter2D(image, -1, Ky)

    magnitude = np.sqrt(np.square(Ix) + np.square(Iy))
    magnitude *= 255.0 / magnitude.max()
    
    arctan = np.arctan2(Iy, Ix)
    theta = np.array([(np.rad2deg(x) + 360) if x < 0 else np.rad2deg(x) for x in arctan.flatten()]).reshape((image.shape))

    return (magnitude, theta)

def non_max_suppression(magnitude, theta):
    '''
    Applies Non-Maximum Suppression.

    Parameters:
        image (numpy.ndarray): The input image.
        theta (numpy.ndarray): The orientation of the image gradients.

    Returns:
        Z(numpy.ndarray): Output of Non-Maximum Suppression algorithm.
    '''
    #Writer your code here
    PI = 180

    M, N = image.shape
    Z = np.zeros((M,N), dtype=np.int32)


    for i in range(1, M - 1):
        for j in range(1, N - 1):

            orientation = theta[i, j]

            if (0 <= orientation < PI / 8) or (15 * PI / 8 <= orientation <= 2 * PI) or (7 * PI / 8 <= orientation < 9 * PI / 8):

                previous_pixel = magnitude[i, j - 1]
                next_pixel = magnitude[i, j + 1]

            elif (PI / 8 <= orientation < 3 * PI / 8) or (9 * PI / 8 <= orientation < 11 * PI / 8):

                previous_pixel = magnitude[i + 1, j - 1]
                next_pixel = magnitude[i - 1, j + 1]

            elif (3 * PI / 8 <= orientation < 5 * PI / 8) or (11 * PI / 8 <= orientation < 13 * PI / 8):

                previous_pixel = magnitude[i - 1, j]
                next_pixel = magnitude[i + 1, j]

            elif (5 * PI / 8 <= orientation < 7 * PI / 8) or (13 * PI / 8 <= orientation < 15 * PI / 8):

                previous_pixel = magnitude[i - 1, j - 1]
                next_pixel = magnitude[i + 1, j + 1]

            if magnitude[i, j] > previous_pixel and magnitude[i, j] > next_pixel:
                Z[i, j] = magnitude[i, j]
    return Z

def hysteresis_threshold(image, lowThreshold, highThreshold):
    '''
    Finds strong, weak, and non-relevant pixels.

    Parameters:
        image (numpy.ndarray): The input image.
        lowThreshold(int): Low Threshold.
        highThreshold(int): High Threshold.

    Returns:
        result(numpy.ndarray): Output of applying hysteresis threshold.
    '''
    #Writer your code here
    M, N = image.shape
    result = np.zeros((M,N), dtype=np.int32)

    strong = 255
    weak = 40

    image[np.where(image <= lowThreshold)] = 0
    image[np.where((image < highThreshold) & (image > lowThreshold))] = weak
    image[np.where(image >= highThreshold)] = strong

    # top to down
    top_down = image.copy()
    for i in range(M):
        for j in range(N):
            if top_down[i, j] == weak:
                neighbors = top_down[i - 1:i + 1, j - 1:j + 1].flatten()
                if strong in neighbors:
                    top_down[i, j] = strong
                else:
                    top_down[i, j] = 0

    # bottom to up
    bottom_up = image.copy()
    for i in range(M - 1, 0, -1):
        for j in range(N - 1, 0, -1):
            if bottom_up[i, j] == weak:
                neighbors = bottom_up[i - 1:i + 1, j - 1:j + 1].flatten()
                if strong in neighbors:
                    bottom_up[i, j] = strong
                else:
                    bottom_up[i, j] = 0

    # left to right
    left_to_right = image.copy()
    for i in range(M - 1, 0, -1):
        for j in range(N):
            if left_to_right[i, j] == weak:
                neighbors = left_to_right[i - 1:i + 1, j - 1:j + 1].flatten()
                if strong in neighbors:
                    left_to_right[i, j] = strong
                else:
                    left_to_right[i, j] = 0

    #right to left
    right_to_left = image.copy()
    for i in range(M):
        for j in range(N - 1, 0, -1):
            if right_to_left[i, j] == weak:
                neighbors = right_to_left[i - 1:i + 1, j - 1:j + 1].flatten()
                if strong in neighbors:
                    right_to_left[i, j] = strong
                else:
                    right_to_left[i, j] = 0

    result = top_down + bottom_up + left_to_right + right_to_left
    result[result > 255] = 255

    return result

def canny(image, kernel_size = 5, sigma = 3.5, lowtreshold = 30, hightreshold = 180):
    '''
    Applys Canny edge detector on the input image.

    Parameters:
        image (numpy.ndarray): The input image.
        size (int): size of kernel.
        sigma(float): standard deviation of gaussian kernel.
        lowThreshold(int): Low Threshold.
        highThreshold(int): High Threshold.

    Returns:
        img_smoothed(numpy.ndarray): Result of applying the Gaussian kernel on the input image.
        gradient(numpy.ndarray): The image of the gradients.
        nonMaxImg(numpy.ndarray): Output of Non-Maximum Suppression algorithm.
        thresholdImg(numpy.ndarray): Output of applying hysteresis threshold.
        img_final(numpy.ndarray): Result of canny edge detector. The image of detected edges.
    '''
    img_smoothed = cv2.filter2D(image, -1, gaussian_kernel(kernel_size, sigma))
    magnitude, theta = sobel_filters(img_smoothed)

    nonMaxImg = non_max_suppression(magnitude, theta)
    thresholdImg = hysteresis_threshold(nonMaxImg, lowtreshold, hightreshold)
    img_final = thresholdImg

    return img_smoothed, magnitude, nonMaxImg, thresholdImg, img_final
