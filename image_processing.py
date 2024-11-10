import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def resize_image(image, new_size):
    return image.resize(new_size)

def flip_image(image, axis):
    if axis == 'horizontal':
        return ImageOps.mirror(image)
    elif  axis == 'vertical':
        return ImageOps.flip(image)
    return image

def translate_and_rotate(image, tx, ty, angle):
    rows, cols = image.size
    M_translate = np.float32([[1, 0, tx],
                              [0, 1, ty]])
    translated = cv2.warpAffine(np.array(image), M_translate, (cols, rows))

    M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(translated, M_rotate, (cols, rows))

    return Image.fromarray(rotated)

def convert_to_grayscale(image):
    return image.convert('L')

def histogram_equalization(image):
    img_array = np.array(image)
    channels = cv2.split(img_array)
    equalized_channels = []
    
    for channel in channels:
        equalized_channel = cv2.equalizeHist(channel)
        equalized_channels.append(equalized_channel)

    equalized_image = cv2.merge(equalized_channels) 
    return Image.fromarray(equalized_image)


def histogram_matching(source_image, reference_image):
    source_array = np.array(source_image)
    reference_array = np.array(reference_image)
    source_channels = cv2.split(source_array)
    reference_channels = cv2.split(reference_array)
    matched_channels = []
    for i in range(3):
        src_hist, bins = np.histogram(source_channels[i].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference_channels[i].flatten(), 256, [0, 256])
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        src_cdf_normalized = src_cdf / src_cdf.max()
        ref_cdf_normalized = ref_cdf / ref_cdf.max()
        lookup_table = np.interp(src_cdf_normalized, ref_cdf_normalized, bins[:-1])
        matched_channel = np.interp(source_channels[i], bins[:-1], lookup_table).astype(np.uint8)
        matched_channels.append(matched_channel)
    matched_image = cv2.merge(matched_channels)
    return Image.fromarray(matched_image)

def binarize_image_manual(image, threshold):
    img_array = np.array(image.convert('L'))
    _, binary_image = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary_image)    

def binarize_image_adaptive_mean(image, kernel_size):
    img_array = np.array(image.convert('L'))
    binary_image = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel_size, 10)
    return Image.fromarray(binary_image)

def binarize_image_adaptive_gaussian(image, kernel_size):
    img_array = np.array(image.convert('L'))
    binary_image = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel_size, 10)
    return Image.fromarray(binary_image)

def denoise_mean_filter(image, kernel_size):
    img_array = np.array(image)
    denoised_image = cv2.blur(img_array, (kernel_size, kernel_size))
    return Image.fromarray(denoised_image)

def denoise_median_filter(image, kernel_size):
    img_array = np.array(image)
    denoised_image = cv2.medianBlur(img_array, kernel_size)
    return Image.fromarray(denoised_image)

def denoise_gaussian_filter(image, kernel_size):
    kernel_size = int(kernel_size)
    img_array = np.array(image)
    denoised_image = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    return Image.fromarray(denoised_image)

def canny_edge_detection(image, low_threshold, high_threshold, aperture_size=3, l2_gradient=False):
    img_array = np.array(image.convert('L'))
    edges = cv2.Canny(img_array, low_threshold, high_threshold, apertureSize=aperture_size, L2gradient=l2_gradient)
    return Image.fromarray(edges)

def sobel_edge_detection(image, direction='Both'):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3) 
    if direction == 'X':
        return Image.fromarray(np.uint8(np.abs(sobel_x)))
    elif direction == 'Y':
        return Image.fromarray(np.uint8(np.abs(sobel_y)))
    else:
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        return Image.fromarray(np.uint8(np.clip(sobel_combined, 0, 255)))

def roberts_edge_detection(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    roberts_x = cv2.filter2D(gray_image, -1, kernel_x)
    roberts_y = cv2.filter2D(gray_image, -1, kernel_y)
    roberts_combined = np.sqrt(roberts_x**2 + roberts_y**2)
    return Image.fromarray(np.uint8(np.clip(roberts_combined, 0, 255)))


def kmeans_segmentation(image, n_clusters):
    img_array = np.array(image)
    Z = img_array.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(Z, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape((img_array.shape))
    return Image.fromarray(segmented_image)

def dft_rgb_image(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted

def dct_rgb_image(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    dct = cv2.dct(np.float32(gray_image))
    dct_shifted = np.fft.fftshift(dct)
    return dct_shifted