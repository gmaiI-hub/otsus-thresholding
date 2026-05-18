import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Page Configuration
# Sets the browser tab title and expands the layout to use the full screen width
st.set_page_config(page_title="Otsu's Thresholding Simulator", layout="wide")

st.title("Interactive Learning Pack: Otsu's Thresholding")
st.markdown("""
This simulator explores Otsu's Method, an algorithm used to automatically perform 
histogram-based image thresholding by minimizing within-class variance.
""")

# Sidebar Control Panel
st.sidebar.header("Control Panel")

# Deliverable 4 Requirement: User must be able to upload an image 
uploaded_file = st.sidebar.file_uploader("1. Upload an image", type=["jpg", "jpeg", "png"])

# Deliverable 3 & 5 Requirement: Addressing failure cases and limitations
# This checkbox allows the user to see how non-uniform lighting affects the algorithm
simulate_failure = st.sidebar.checkbox("Simulate Bad Lighting (Failure Case)")

if uploaded_file is not None:
    # Convert uploaded image to Grayscale for processing
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)

    # Failure Case Logic
    # Otsu's method assumes a bimodal distribution. 
    # Adding a lighting gradient breaks this assumption to demonstrate failure .
    if simulate_failure:
        h, w = img_array.shape
        # Create a linear gradient from 0 to 150 intensity
        gradient = np.linspace(0, 150, w).astype(np.uint8)
        gradient = np.tile(gradient, (h, 1))
        # Blend the original image with the gradient
        img_array = cv2.addWeighted(img_array, 0.5, gradient, 0.5, 0)
        st.sidebar.warning("Lighting gradient applied. Global Otsu thresholding effectiveness may be reduced.")

    # Selection between Global and Local methods
    method = st.sidebar.radio("2. Select Thresholding Method:", 
                             ("Global Otsu's Method", "Adaptive Thresholding"))

    # Sidebar Theory Section: Displaying the mathematical foundation
    st.sidebar.markdown("---")
    st.sidebar.subheader("Mathematical Foundation")
    st.sidebar.latex(r"\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)")
    st.sidebar.caption("Otsu's goal is to find a threshold 't' that minimizes this within-class variance.")

    # Main Visualization Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img_array, use_container_width=True)
        
        # Histogram Logic
        # Visualizing the distribution of pixels is key to understanding Otsu's bimodal assumption
        st.subheader("Pixel Intensity Histogram")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(img_array.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.6)
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Frequency")
        
        if method == "Global Otsu's Method":
            # OpenCV's Otsu Implementation
            # The threshold parameter (0) is ignored as THRESH_OTSU calculates it automatically
            ret, thresh_img = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Draw the threshold line on the histogram
            ax.axvline(ret, color='red', linestyle='dashed', linewidth=2, label=f"Otsu T={int(ret)}")
            ax.legend()
            st.pyplot(fig)
            
            with col2:
                st.subheader(f"Otsu Result (Threshold = {int(ret)})")
                st.image(thresh_img, use_container_width=True)
                st.success(f"Optimal global threshold found at: {int(ret)}")
                st.info("Otsu performs best when the histogram is clearly bimodal.")

        elif method == "Adaptive Thresholding":
            st.pyplot(fig)
            with col2:
                st.subheader("Adaptive Thresholding Result")
                # Sliders allow real-time parameter tuning as requested in the project brief [cite: 69]
                block_size = st.sidebar.slider("Block Size (Area size, must be odd)", 3, 99, 11, step=2)
                c_val = st.sidebar.slider("Constant (C) subtracted from mean", -10, 10, 2)
                
                # Adaptive Gaussian Thresholding handles uneven lighting better than Global Otsu [cite: 152]
                thresh_img = cv2.adaptiveThreshold(
                    img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, block_size, c_val
                )
                st.image(thresh_img, use_container_width=True)
                st.info("Adaptive thresholding calculates different thresholds for local neighborhoods.")

else:
    # Initial state when no image is uploaded
    st.info("Please upload an image from the sidebar to start the simulator.")
