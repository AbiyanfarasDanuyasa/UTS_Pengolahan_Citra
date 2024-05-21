import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def calculate_histogram(image):
    color = ('b', 'g', 'r')
    histogram = {}
    for i, col in enumerate(color):
        histogram[col] = cv2.calcHist([image], [i], None, [256], [0, 256])
    return histogram

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    brightness = int((brightness - 50) * 2.55)
    contrast = int((contrast - 50) * 2.55)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
    return contour_image

def segment_image_kmeans(image, k):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image

def main():
    st.title('Image Manipulation with Streamlit')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.subheader('Convert RGB to HSV')
        if st.button('Convert to HSV'):
            hsv_image = convert_rgb_to_hsv(image)
            st.image(hsv_image, caption='HSV Image', use_column_width=True)

        st.subheader('Calculate Histogram')
        if st.button('Show Histogram'):
            histogram = calculate_histogram(image)
            fig, ax = plt.subplots()
            for col in histogram:
                ax.plot(histogram[col], color=col)
            st.pyplot(fig)

        st.subheader('Adjust Brightness and Contrast')
        brightness = st.slider('Brightness', 0, 100, 50)
        contrast = st.slider('Contrast', 0, 100, 50)
        adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
        st.image(adjusted_image, caption='Adjusted Image', use_column_width=True)

        st.subheader('Detect Contours')
        if st.button('Show Contours'):
            contour_image = detect_contours(image)
            st.image(contour_image, caption='Contour Image', use_column_width=True)

        st.subheader('Segment Image using K-Means Clustering')
        k = st.slider('Number of clusters (k)', 2, 10, 4)
        if st.button('Segment Image'):
            segmented_image = segment_image_kmeans(image, k)
            st.image(segmented_image, caption='Segmented Image', use_column_width=True)

if __name__ == '__main__':
    main()
