import streamlit as st
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
st.set_page_config(page_title="Otsu's Thresholding Simulator", layout="wide")
st.title("Interactive Principle Simulator: Otsu's Thresholding")
st.write("Upload an image to start the experiment.")
st.sidebar.header("control panel")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    method = st.sidebar.radio("Select Thresholding Method:",("Global Otsu's Method", "Adaptive Thresholding"))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Grayscale Image")
        st.image(img_array, use_container_width=True)
        st.subheader("pixel Intensity Histogram")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(img_array.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.7)
        ax.set_title("Histogram of Pixel Intensities")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        if method == "Global Otsu's Method":
            ret, thresh_img = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ax.axvline(ret, color='red', linestyle='dashed', linewidth=2, label=f"Otsu Threshold: {ret}")
            ax.legend()
            st.pyplot(fig)
            with col2:
                st.subheader(f"Otsu's Result ( T = {ret})")
                st.image(thresh_img, use_container_width=True)
                st.info("Otsu's method calculates a single global threshold by minimizing the within-class variance")
        elif method == "Adaptive Thresholding":
            st.pyplot(fig)
            with col2:
                st.subheader("Adaptive Thresholding Result")
                block_size = st.sidebar.slider("Block size (must me odd)",min_value=3, max_value=99, value=11, step=2)
                c_value = st.sidebar.slider("constant (c) subtracted from mean", min_value=-10, max_value=10, value=2, step=1)
                thresh_img = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)
                st.image(thresh_img, use_container_width=True)
                st.info("Adaptive thresholding calculates a different threshold for small regions of the image.")
else:
    st.info("Please upload an image from the sidebar to start the simulation.")
