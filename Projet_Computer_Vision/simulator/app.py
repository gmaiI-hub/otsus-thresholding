import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Otsu's Thresholding Simulator", layout="wide")

st.title(" Interactive Learning Pack: Otsu's Thresholding")
st.markdown("""
This simulator explores **Otsu's Method**, which automatically finds the optimal threshold by minimizing **within-class variance**.
""")

# --- Sidebar ---
st.sidebar.header(" Control Panel")
uploaded_file = st.sidebar.file_uploader("1. Upload an image", type=["jpg", "jpeg", "png"])

# Option pour simuler un cas d'échec (Failure Case)
simulate_failure = st.sidebar.checkbox(" Simulate Bad Lighting (Failure Case)")

if uploaded_file is not None:
    # Lecture de l'image
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)

    # Simulation d'un gradient de lumière (Otsu échoue souvent ici)
    if simulate_failure:
        h, w = img_array.shape
        gradient = np.linspace(0, 150, w).astype(np.uint8)
        gradient = np.tile(gradient, (h, 1))
        img_array = cv2.addWeighted(img_array, 0.5, gradient, 0.5, 0)
        st.sidebar.warning("Lighting gradient applied. Notice how Global Otsu struggles!")

    method = st.sidebar.radio("2. Select Thresholding Method:", 
                             ("Global Otsu's Method", "Adaptive Thresholding"))

    # Affichage de la théorie dans le sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Mathematical Foundation")
    st.sidebar.latex(r"\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)")
    st.sidebar.caption("Otsu minimizes this within-class variance.")

    # --- Main Layout ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img_array, use_container_width=True)
        
        # Histogramme
        st.subheader("Pixel Intensity Histogram")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts, bins, _ = ax.hist(img_array.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.6)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")
        
        if method == "Global Otsu's Method":
            # Calcul Otsu
            ret, thresh_img = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ax.axvline(ret, color='red', linestyle='dashed', linewidth=2, label=f"Otsu Threshold (T={int(ret)})")
            ax.legend()
            st.pyplot(fig)
            
            with col2:
                st.subheader(f"Otsu Result (T = {int(ret)})")
                st.image(thresh_img, use_container_width=True)
                st.success(f"Optimal threshold found at {int(ret)}")
                st.info("Otsu works best on **bimodal** histograms (two clear peaks).")

        elif method == "Adaptive Thresholding":
            st.pyplot(fig)
            with col2:
                st.subheader("Adaptive Result (Gaussian)")
                block_size = st.sidebar.slider("Block Size (must be odd)", 3, 99, 11, step=2)
                c_val = st.sidebar.slider("Constant (C)", -10, 10, 2)
                
                thresh_img = cv2.adaptiveThreshold(
                    img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, block_size, c_val
                )
                st.image(thresh_img, use_container_width=True)
                st.info("Adaptive thresholding handles non-uniform lighting by calculating local thresholds.")

else:
    st.info("Please upload an image to start the simulation.")
