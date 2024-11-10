import streamlit as st
import numpy as np
from PIL import Image
from image_processing import *

st.title("Image Processing Application")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

# Horizontal radio buttons for algorithm selection
algorithm = st.radio("Select an Algorithm", [
    "Image Resizing and Flipping",
    "Translate and Rotate Image",
    "Grayscale Conversion",
    "Histogram Equalization",
    "Histogram Matching",
    "Image Binarization",
    "Image Denoising",
    "Edge Detection",
    "Image Segmentation",
    "Image Domain Transformation"
], horizontal=True)

# Algorithm-specific input fields
if uploaded_file is not None:
    if algorithm == "Image Resizing and Flipping":
        st.header("Image Resizing and Flipping")
        new_width = st.number_input("New Width", min_value=1)
        new_height = st.number_input("New Height", min_value=1)
        flip_axis = st.selectbox("Flip Axis", ['None', 'Horizontal', 'Vertical'])
        if st.button("Resize and Flip"):
            resized_img = resize_image(image, (new_width, new_height))
            flipped_img = flip_image(resized_img, flip_axis.lower())
            st.image(flipped_img, caption="Resized and Flipped Image")

    elif algorithm == "Translate and Rotate Image":
        st.header("Translate and Rotate Image")
        tx = st.number_input("Translation X", min_value=-100.0, max_value=100.0, value=0.0)
        ty = st.number_input("Translation Y", min_value=-100.0, max_value=100.0, value=0.0)
        angle = st.number_input("Rotation Angle (degrees)", min_value=0.0, max_value=360.0, value=0.0)
        if st.button("Translate and Rotate"):
            transformed_img = translate_and_rotate(image, tx, ty, angle)
            st.image(transformed_img, caption="Translated and Rotated Image")

    elif algorithm == "Grayscale Conversion":
        st.header("Grayscale Conversion")
        if st.button("Convert to Grayscale"):
            gray_img = convert_to_grayscale(image)
            st.image(gray_img, caption="Grayscale Image")

    elif algorithm == "Histogram Equalization":
        st.header("Histogram Equalization")
        if st.button("Equalize Histogram"):
            equalized_img = histogram_equalization(image)
            st.image(equalized_img, caption="Equalized Histogram")
    
    elif algorithm == "Histogram Matching":
        st.header("Histogram Matching")
        reference_file = st.file_uploader("Choose a reference image...", type=["jpg", "png", "jpeg"])
        if reference_file is not None:
            reference_image = Image.open(reference_file)
            if st.button("Match Histograms"):
                matched_image = histogram_matching(image, reference_image)
                st.image(matched_image, caption="Histogram Matched Image")
            else:
                st.warning("Please upload a reference image.")

    elif algorithm == "Image Binarization":
        st.header("Image Binarization")
        binarization_method = st.radio("Choose Binarization Method", ["Manual Thresholding", "Adaptive Mean Thresholding", "Adaptive Gaussian Thresholding"])
        
        if binarization_method == "Manual Thresholding":
            threshold_value = st.slider("Select Threshold Value", min_value=0, max_value=255, value=128)
            if st.button("Binarize Manual"):
                binarized_img_manual = binarize_image_manual(image, threshold_value)
                st.image(binarized_img_manual, caption="Binarized Image (Manual)")

        elif binarization_method == "Adaptive Mean Thresholding":
            kernel_size_mean = st.number_input("Kernel Size for Mean", min_value=3, max_value=21, value=11)  # Must be odd number.
            if st.button("Binarize Adaptive Mean"):
                binarized_img_adaptive_mean = binarize_image_adaptive_mean(image, kernel_size_mean)
                st.image(binarized_img_adaptive_mean, caption="Binarized Image (Adaptive Mean)")

        elif binarization_method == "Adaptive Gaussian Thresholding":
            kernel_size_gaussian = st.number_input("Kernel Size for Gaussian", min_value=3, max_value=21, value=11)  # Must be odd number.
            if st.button("Binarize Adaptive Gaussian"):
                binarized_img_adaptive_gaussian = binarize_image_adaptive_gaussian(image, kernel_size_gaussian)
                st.image(binarized_img_adaptive_gaussian, caption="Binarized Image (Adaptive Gaussian)")

    elif algorithm == "Image Denoising":
        st.header("Image Denoising")
        denoising_method = st.radio("Choose Denoising Method", ["Mean Filter", "Median Filter", "Gaussian Filter"])
        
        kernel_size_denoise = st.number_input("Kernel Size (must be odd)", min_value=3, max_value=21, value=3)  # Must be odd number.
        
        if denoising_method == "Mean Filter":
            if st.button("Denoise Mean Filter"):
                denoised_mean_img = denoise_mean_filter(image, kernel_size_denoise)
                st.image(denoised_mean_img , caption="Denoised Image (Mean Filter)")

        elif denoising_method == "Median Filter":
            if kernel_size_denoise % 2 == 0:
                kernel_size_denoise += 1
            
            if st.button("Denoise Median Filter"):
                denoised_median_img = denoise_median_filter(image,kernel_size_denoise)
                st.image(denoised_median_img , caption="Denoised Image (Median Filter)")

        elif denoising_method == "Gaussian Filter":
            if kernel_size_denoise % 2 == 0:
                kernel_size_denoise += 1
            
            if st.button("Denoise Gaussian Filter"):
                denoised_gaussian_img = denoise_gaussian_filter(image,kernel_size_denoise )
                st.image(denoised_gaussian_img , caption="Denoised Image (Gaussian Filter)")

    elif algorithm == "Edge Detection":
        st.header("Edge Detection")
        st.header("Edge Detection")
        edge_method = st.radio("Choose Edge Detection Method", ["Canny", "Sobel", "Roberts"])
        
        if edge_method == "Canny":
            low_threshold = st.number_input("Low Threshold", min_value=0, max_value=255, value=100)
            high_threshold = st.number_input("High Threshold", min_value=0, max_value=255, value=200)
            aperture_size = st.selectbox("Aperture Size", [3, 5, 7])
            L2_gradient = st.checkbox("Use L2 Gradient", value=False)
            
            if st.button("Detect Edges"):
                edges_img_canny = canny_edge_detection(image, low_threshold, high_threshold, aperture_size, L2_gradient)
                st.image(edges_img_canny, caption="Edges Detected with Canny")

        elif edge_method == "Sobel":
            sobel_direction = st.selectbox("Sobel Direction", ["Both", "X", "Y"])
            if st.button("Detect Edges with Sobel"):
                edges_img_sobel = sobel_edge_detection(image, direction=sobel_direction)
                st.image(edges_img_sobel, caption="Edges Detected with Sobel")

        elif edge_method == "Roberts":
            if st.button("Detect Edges with Roberts"):
                edges_img_roberts = roberts_edge_detection(image)
                st.image(edges_img_roberts, caption="Edges Detected with Roberts")


    elif algorithm == "Image Segmentation":
        st.header("Image Segmentation using K-Means Clustering")
        n_clusters_kmeans = st.number_input("Number of Clusters", min_value=1,max_value=10,value=3 )
        if st.button("Segment with K-Means"):
            segmented_kmeans_img = kmeans_segmentation(image,n_clusters_kmeans )
            st.image(segmented_kmeans_img, caption="Segmented with K-Means")

    elif algorithm == "Image Domain Transformation":
        st.header("Image Domain Transformation")
        transform_method = st.radio("Choose Transformation Method", ["Discrete Fourier Transform (DFT)", "Discrete Cosine Transform (DCT)"])
        
        if transform_method == "Discrete Fourier Transform (DFT)":
            if st.button("Calculate DFT"):
                dft_result = dft_rgb_image(image)
                # Calculate magnitude spectrum and shift it to center
                magnitude_spectrum = cv2.magnitude(dft_result[:, :, 0], dft_result[:, :, 1])
                magnitude_spectrum += 1  # Avoid log(0)
                magnitude_spectrum = np.log(magnitude_spectrum)  # Log scale
                magnitude_spectrum -= np.min(magnitude_spectrum)  # Normalize to [0, max]
                magnitude_spectrum /= np.max(magnitude_spectrum)  # Normalize to [0, 1]
                st.image(magnitude_spectrum, caption="DFT Magnitude Spectrum", use_column_width=True)

        elif transform_method == "Discrete Cosine Transform (DCT)":
            if st.button("Calculate DCT"):
                dct_result = dct_rgb_image(image)
                # Display log-scaled DCT coefficients directly from updated function
                dct_result_normalized = np.log(np.abs(dct_result) + 1)
                dct_result_normalized -= np.min(dct_result_normalized)  # Normalize to [0, max]
                dct_result_normalized /= np.max(dct_result_normalized)  # Normalize to [0, 1]
                st.image(dct_result_normalized, caption="DCT Coefficients (Log Scale)", use_column_width=True)
