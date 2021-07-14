def compute_phase_magnitude(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    phase_spectrum = np.angle(dft_shift)
    magnitude_spectrum = np.abs(dft_shift)

    return phase_spectrum, magnitude_spectrum

def compute_ifft(phase, magnitude):

    complex_spectrum = magnitude * np.exp(phase * 1j)
    shift = np.fft.ifftshift(complex_spectrum)
    ifft = np.fft.ifft2(shift)

    return np.abs(ifft)

def draw_phase_amplitude(image):
    '''
    Returns the phase image and the amplitude image from the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        tuple of numpy.ndarray: The tuple of the phase image and the amplitude image.
    '''

    phase, amp = compute_phase_magnitude(image)

    ### Used np.log so that the amplitude can be shown like a image and values don't be too large or too small
    amp = np.log(amp)

    return phase, amp

def change_phase_domain(image1, image2):
    '''
    Substitutes the phase of image1 by the phase of image2 and returns two new images.

    Parameters:
        image1 (numpy.ndarray): The input image1.
        image2 (numpy.ndarray): The input image2.

    Returns:
        tuple of numpy.ndarray: The tuple of result images.
    '''

    phase_1, mag_1 = compute_phase_magnitude(image1)
    phase_2, mag_2 = compute_phase_magnitude(image2)

    img1 = compute_ifft(phase_1, mag_2)
    img2 = compute_ifft(phase_2, mag_1)

    return img1, img2s
