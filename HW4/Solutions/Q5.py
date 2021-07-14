# Copied from HW3 with slight modifications

def gaussian(x, y, std):
    return np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(std)))


def gaussian_filter(shape, std):
    '''
    Creates the Guassian kernel with given size and std.

    Parameters:
        size (int): The size of the kernel. It must be odd.
        std (float): The standard deviation of the kernel.

    Returns:
        numpy.ndarray: The Guassina kernel.
    '''

    kernel = np.zeros((shape[0],shape[1]), np.float)

    gap_x = np.floor(shape[0] / 2)
    gap_y = np.floor(shape[1] / 2)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = i - gap_x
            y = j - gap_y
            kernel[i, j] = gaussian(x, y, std)

    return kernel

def enhance_image(image):
    '''
    Enhances the input image by applying a filter in the frequency domain.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The result enhanced image.
    '''

    ###  Inverting the gaussian kernel to become a high pass filter.
    inv_kernel = 1 - gaussian_filter(image.shape, 15)

    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    edges = dft_shift * inv_kernel

    idft_shift = np.fft.ifftshift(edges)
    idft = np.fft.ifft2(idft_shift)

    enhanced = np.abs(idft) + image
    return enhanced
