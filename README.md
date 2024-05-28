# UTS PENGOLAHAN CITRA 
# APLIKASI MANIPULASI GAMBAR CITRAa

```
NAMA ANGGOTA KELOMPOK
Abiyanfaras Danuyasa    | 312210103
Birrham Efendi Lubis    | 312210272
Muhammad Zidan Fadillah | 312210277

Kelas : TI.22.A2
Mata Kuliah : Pengolahan Citra
```

# LINK PDF TUTORIAL ATAU CARA MENGGUNAKAN APLIKASI MANIPULASI GAMBAR CITRA
[Tutorial atau cara menggunakan Aplikasi Manipulasi Gambar Citra-.pdf](https://github.com/AbiyanfarasDanuyasa/UTS_Pengolahan_Citra/files/15474851/Tutorial.atau.cara.menggunakan.Aplikasi.Manipulasi.Gambar.Citra-.pdf)


# Meng import library yang akan di gunakan 

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
```
# konversi ke RGB Ke HSV
```
def RGB menjadi HSV(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image
```
# Menghitung dan juga Menampilkan HISTOGRAM
```
def Menghitung histogram(image):
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
```
# Mengatur sesuai keinginan pada BRIGHTNESS DAN CONTRAS
```
def Menyesuaikan_brignest_contras(image, brignest, contras):
    adjusted = cv2.convertScaleAbs(image, alpha=contrast/127.0, beta=brightness)
    return adjusted
```
# Mendeteksi CONTOURS
```
def Menemukan contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
```

# Desain interface pada streamlit
```
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

        st.subheader('Gambar Asli')
        st.image(image, channels="BGR", use_column_width=True)

        if st.button('RGB menjadi HSV'):
            hsv_image = convert_to_hsv(image)
            st.subheader('HSV Image')
            st.image(hsv_image, channels="HSV", use_column_width=True)

        if st.button('Menghitung Histogram'):
            st.subheader('Histogram')
            compute_histogram(image)

        st.subheader('Menyesuaikan Brignest and Contras')
        brightness = st.slider('Brightness', -100, 100, 0)
        contrast = st.slider('Contrast', -100, 100, 0)
        adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
        st.image(adjusted_image, channels="BGR", use_column_width=True)

        if st.button('Menemukan Contours'):
            contours = find_contours(image)
            st.subheader('Contours')
            image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
            st.image(image_with_contours, channels="BGR", use_column_width=True)

if _name_ == '_main_':
    main()
```

# Hasil

# Upload gambar
![image](https://github.com/AbiyanfarasDanuyasa/UTS_Pengolahan_Citra/assets/115553474/bbbc469f-8244-489b-bbc2-87283d7f322d)




# Sudah di upload
![image](https://github.com/AbiyanfarasDanuyasa/UTS_Pengolahan_Citra/assets/115553474/c99d70ad-2a3c-4bd5-b350-1c7d3d20a314)



# HSV

![image](https://github.com/AbiyanfarasDanuyasa/UTS_Pengolahan_Citra/assets/115553474/4a91ae58-0acc-4de4-a078-ac7f10364c57)



# HISTOGRAM

![image](https://github.com/AbiyanfarasDanuyasa/UTS_Pengolahan_Citra/assets/115553474/f556dd43-373b-4a17-92f2-3e33ae32a0d1)



# BRIGNEST DAN CONTRAS

![image](https://github.com/AbiyanfarasDanuyasa/UTS_Pengolahan_Citra/assets/115553474/83544348-ebfb-4707-a594-77fefa411c69)


# CONTOUR
![image](https://github.com/AbiyanfarasDanuyasa/UTS_Pengolahan_Citra/assets/115553474/8bb68dad-7d65-49f8-b606-bf9837ae2f01)











