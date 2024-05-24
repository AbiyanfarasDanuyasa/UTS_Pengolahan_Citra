# UTS PENGOLAHAN CITRA

```
NAMA KELOMPOK
Abiyanfaras Danuyasa    | 312210103
Birrham Efendi Lubis    | 312210272
Muhammad Zidan Fadillah | 312210277

Kelas : TI.22.A.2
Mata Kuliah : Pengolahan Citra
```

# LINK PDF TUTORIAL PENGGUNAAN APLIKASI
[Tutorial penggunaan Aplikasi Manipulasi Proses Gambar Citra1,.pdf](https://github.com/AbiyanfarasDanuyasa/UTS_pengolahan_citra/files/15436556/Tutorial.penggunaan.Aplikasi.Manipulasi.Proses.Gambar.Citra1.pdf)



# Code nya 

```
import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

def set_theme():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to bottom, #2e2e2e, #101010) !important;
            color: #EAEAEA !important;
        }
        .css-2trqyj {
            border-radius: 12px !important;
            background-color: #1f77b4 !important;
            color: white !important;
        }
        .css-2trqyj:hover {
            background-color: #105399 !important;
        }
        .title-wrapper {
            perspective: 1000px;
            perspective-origin: center;
            overflow: hidden;
        }
        .title-wrapper h1 {
            position: relative;
            transition: all 0.3s;
            transform-origin: center;
            transform-style: preserve-3d;
            backface-visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def convert_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def compute_histogram(image):
    colors = ('b', 'g', 'r')
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, col in enumerate(colors):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(histogram, color=col)
        ax.set_xlim([0, 256])
    ax.set_title('Histogram')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def adjust_brightness_contrast(image, brightness, contrast):
    adjusted = cv2.convertScaleAbs(image, alpha=contrast/127.0, beta=brightness)
    return adjusted

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

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
    set_theme()

    st.markdown(
        """
        <div class="title-wrapper">
            <h1 style="font-size: 48px; text-align: center; color: #F08080;
                text-shadow: 2px 2px 4px #000000;">Aplikasi Manipulasi Gambar Citra</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.subheader('Original Image')
        st.image(image, channels="BGR", use_column_width=True)

        if st.button('Convert to HSV'):
            hsv_image = convert_to_hsv(image)
            st.subheader('HSV Image')
            st.image(hsv_image, channels="HSV", use_column_width=True)

        if st.button('Compute Histogram'):
            st.subheader('Histogram')
            compute_histogram(image)

        st.subheader('Adjust Brightness and Contrast')
        brightness = st.slider('Brightness', -100, 100, 0)
        contrast = st.slider('Contrast', -100, 100, 0)
        adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
        st.image(adjusted_image, channels="BGR", use_column_width=True)

        if st.button('Find Contours'):
            contours = find_contours(image)
            st.subheader('Contours')
            image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
            st.image(image_with_contours, channels="BGR", use_column_width=True)
        
        st.subheader('Segmentasi')
        k = st.slider('Number of clusters (k)', 2, 10, 4)
        if st.button('Segment Image'):
            segmented_image = segment_image_kmeans(image, k)
            st.image(segmented_image, caption='Segmented Image', use_column_width=True)

if _name_ == '_main_':
    main()
```

# Hasil

# Upload gambar
![12](https://github.com/AbiyanfarasDanuyasa/UTS_pengolahan_citra/assets/115553474/480e46bf-a4b1-42be-a1e1-b780d75bd92c)


# Sudah di upload
![11](https://github.com/AbiyanfarasDanuyasa/UTS_pengolahan_citra/assets/115553474/eb44361b-2938-4cde-a6a1-773cc80689e7)



# HSV
![13](https://github.com/AbiyanfarasDanuyasa/UTS_pengolahan_citra/assets/115553474/912638e0-b0ed-42e2-b668-26cfd566e6e8)


# HISTOGRAM
![14](https://github.com/AbiyanfarasDanuyasa/UTS_pengolahan_citra/assets/115553474/3f1b5055-263b-4746-a1cd-a3719a342689)


# BRIGHTNESS DAN CONTRAS
![15](https://github.com/AbiyanfarasDanuyasa/UTS_pengolahan_citra/assets/115553474/92961cb0-8ced-46ea-9d99-747bc91ad613)

# CONTOURS
![16](https://github.com/AbiyanfarasDanuyasa/UTS_pengolahan_citra/assets/115553474/80ebc0a7-d183-43c9-8cb7-a2d79a2743fb)


# SEGMENTASI
![17](https://github.com/AbiyanfarasDanuyasa/UTS_pengolahan_citra/assets/115553474/cf86547b-d08a-4ab4-9af4-e1df8c21f555)







