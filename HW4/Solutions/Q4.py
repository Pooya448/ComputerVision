def denoise_image(image):
    '''
    Denoises the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The result denoised image.
    '''

    denoised = image.copy()

    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    ### These indices are calculated using photoshop and hard-coded for this problem.
    dft_shift[:,:190] = 0.000001
    dft_shift[:,320:] = 0.000001

    dft_shift[:190,:] = 0.000001
    dft_shift[320:,:] = 0.000001

    idft_shift = np.fft.ifftshift(dft_shift)
    idft = np.fft.ifft2(idft_shift)

    denoised = np.abs(idft)
    return denoised
