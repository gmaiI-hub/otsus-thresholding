Interactive Principle Simulator: Otsu's Thresholdin

Project Overview :

The simulator is a functional, interactive tool built using Python and Streamlit. It allows students and peers to independently explore the core building blocks of image segmentation through global and adaptive thresholding techniques.

Prerequisites :

Ensure you have Python 3.8+ installed on your system. The simulator requires the following libraries, which can be found in the requirements.txt file:
streamlit: For the web-based interactive interface.
opencv-python: For image processing algorithms.
numpy: For numerical operations.
matplotlib: For histogram visualization.
Pillow: For image loading and manipulation.

Installation :

To set up the environment, navigate to the simulator/ directory and run the following command:
pip install -r requirements.txt

Once the dependencies are installed, you can launch the simulator with a single command:
python -m streamlit run app.py

This will start a local web server and open the application in your default browser.

Key Features

Global Otsu = Automatically calculates the optimal threshold by minimizing within-class variance.

Adaptive Thresholding = Calculates thresholds for local regions, handling non-uniform lighting conditions.

Histogram Analysis = Displays a real-time pixel intensity histogram with the chosen threshold visualized.

Failure Case Simulator = A toggle to apply artificial lighting gradients to demonstrate where global methods fail.

