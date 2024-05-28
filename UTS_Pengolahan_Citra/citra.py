import streamlit as st
import cv2
import numpy as np

def segment_image_kmeans(image, k):
    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image to a 2D array of pixels
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to 8-bit values
    centers = np.uint8(centers)
    
    # Map labels to center values
    segmented_image = centers[labels.flatten()]
    
    # Reshape segmented image to original image shape
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image

def main():
    st.title('Image Segmentation with Streamlit')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.subheader('Segment Image using K-Means Clustering')
        k = st.slider('Number of clusters (k)', 2, 10, 4)
        
        segmented_image = segment_image_kmeans(image, k)
        
        st.image(segmented_image, caption='Segmented Image', use_column_width=True)

if __name__ == '__main__':
    main()
import streamlit as st
import cv2
import numpy as np

def segment_image_kmeans(image, k):
    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image to a 2D array of pixels
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to 8-bit values
    centers = np.uint8(centers)
    
    # Map labels to center values
    segmented_image = centers[labels.flatten()]
    
    # Reshape segmented image to original image shape
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image

def main():
    st.title('Image Segmentation with Streamlit')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.subheader('Segment Image using K-Means Clustering')
        k = st.slider('Number of clusters (k)', 2, 10, 4)
        
        segmented_image = segment_image_kmeans(image, k)
        
        st.image(segmented_image, caption='Segmented Image', use_column_width=True)

if __name__ == '__main__':
    main()
