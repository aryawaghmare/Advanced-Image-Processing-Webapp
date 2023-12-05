import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage import feature
import matplotlib.pyplot as plt

# Set Streamlit app title and description
st.set_page_config(
    page_title="Advanced Image Processing based Webapp using Streamlit",
    page_icon="📷",
    layout="wide"
)

# Logo
logo_path = "logo.png"  # Replace with the path to your logo image file
logo = Image.open(r"C:\Users\HP\Downloads\FINAL PROJECT\logo.png")

st.sidebar.image(logo, use_column_width=True)

st.title("Advanced Image Processing based Webapp using Streamlit")
st.markdown(
    "Upload an image and apply advanced image processing techniques. "
    "Enhance, transform, segment, and visualize your images."
)

# File upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Sidebar options
st.sidebar.title("Image Processing Options")

# Add more image processing techniques and parameters here
selected_option = st.sidebar.selectbox(
    "Choose an image processing technique",
    ["None", "Grayscale", "Blur", "Custom Filter", "Histogram Equalization", "Image Enhancement", "Edge Detection", "Image Segmentation", "LBP Feature Extraction", "LTP Upper", "LTP Lower", "RGB Splitter", "K-Mean Clustering", "K-Mean Clustering (K=2)", "Feature Extraction", "Image Transformation", "Color Negation", "Log Transformation", "Gamma Correction"]
)

# Function to process the image based on the selected option
def process_image(input_image, option, params=None):
    img = np.array(input_image)
    
    if option == "Grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif option == "Blur":
        blur_level = params["blur_level"]
        img = cv2.GaussianBlur(img, (blur_level, blur_level), 0)
    elif option == "Custom Filter":
        kernel_size = params["kernel_size"]
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        img = cv2.filter2D(img, -1, kernel)
    elif option == "Histogram Equalization":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)
    elif option == "Image Enhancement":
        alpha = params["alpha"]
        beta = params["beta"]
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    elif option == "Edge Detection":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.Canny(img, 100, 200)
    elif option == "Image Segmentation":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    elif option == "LBP Feature Extraction":
        img = perform_lbp_feature_extraction(img)
    elif option == "LTP Upper":
        img = perform_ltp_upper(img)
    elif option == "LTP Lower":
        img = perform_ltp_lower(img)
    elif option == "RGB Splitter":
        img = rgb_splitter(img)
    elif option == "K-Mean Clustering":
        if params is not None and "k" in params:
            img = k_mean_clustering(img, params["k"])
    elif option == "K-Mean Clustering (K=2)":
        img = k_mean_clustering(img, 2)
    elif option == "Feature Extraction":
        img = feature_extraction(img)
    elif option == "Image Transformation":
        img = image_transformation(img)
    elif option == "Color Negation":
        img = color_negation(img)
    elif option == "Log Transformation":
        img = log_transformation(img)
    elif option == "Gamma Correction":
        img = gamma_correction(img, params["gamma"]) if params is not None and "gamma" in params else img
    
    return img

# Feature extraction functions
def perform_lbp_feature_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min())  # Normalize to [0, 1]
    lbp_colored = cv2.merge([lbp_normalized] * 3)
    return lbp_colored

def perform_ltp_upper(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ltp_upper = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return cv2.merge([ltp_upper] * 3)

def perform_ltp_lower(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ltp_lower = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return cv2.merge([ltp_lower] * 3)

def rgb_splitter(image):
    if isinstance(image, tuple) and len(image) > 0:
        image = image[0]  # Extract the first element of the tuple

    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        # Extract individual color channels
        blue_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 2]

        # Create separate images for each channel
        blue_image = np.zeros_like(image)
        blue_image[:, :, 0] = blue_channel

        green_image = np.zeros_like(image)
        green_image[:, :, 1] = green_channel

        red_image = np.zeros_like(image)
        red_image[:, :, 2] = red_channel

        return red_image, green_image, blue_image
    else:
        raise ValueError("Input image is not in a supported format")


def k_mean_clustering(image, k):
    # Implement K-Mean Clustering
    if image is not None:
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)
        return segmented_image
    else:
        return None

def feature_extraction(image):
    # Implement Feature Extraction
    # For example, extract some specific features from the image
    # Replace this with your actual feature extraction implementation
    # Here, we simply convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.merge([gray_image] * 3)  # Convert grayscale back to RGB format

def image_transformation(image):
    # Implement Image Transformation
    # For example, rotate or flip the image
    # Replace this with your actual image transformation implementation
    # Here, we rotate the image by 90 degrees
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image

def color_negation(image):
    # Implement Color Negation
    # Negate each color channel separately
    negated_image = 255 - image
    return negated_image

def log_transformation(image):
    # Implement Log Transformation
    c = 255 / np.log(1 + np.max(image))
    log_transformed = c * (np.log(image + 1))
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    return log_transformed

def gamma_correction(image, gamma):
    # Normalize the image to the range [0, 1]
    image_normalized = image / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(image_normalized, gamma)

    # Denormalize the gamma-corrected image to the range [0, 255]
    gamma_corrected = np.uint8(gamma_corrected * 255.0)

    return gamma_corrected

# Function to display histogram
def display_histogram(image, title):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate histogram
    hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])

    # Plot histogram
    plt.figure(figsize=(8, 3))
    plt.title(title)
    plt.plot(hist, color='black')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Process and display the image based on the selected option
if selected_option != "None" and uploaded_image:
    original_image = np.array(image)
    
    if selected_option == "Histogram Equalization":
        st.sidebar.markdown("### Original Image Histogram")
        display_histogram(original_image, "Original Image Histogram")

        img = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)

        processed_image = cv2.merge([img] * 3)
        
        st.sidebar.markdown("### Processed Image Histogram")
        display_histogram(processed_image, "Processed Image Histogram")
    else:
        params = None  # No additional parameters needed for other techniques

        if selected_option != "None":
            params = {}
            if selected_option == "Blur":
                params["blur_level"] = st.sidebar.slider("Select blur level", 1, 11, 3)
            elif selected_option == "Custom Filter":
                params["kernel_size"] = st.sidebar.slider("Select kernel size", 3, 11, 3)
            elif selected_option == "Image Enhancement":
                params["alpha"] = st.sidebar.slider("Select alpha (contrast)", 0.1, 3.0, 1.0)
                params["beta"] = st.sidebar.slider("Select beta (brightness)", 0, 100, 0)
            elif selected_option == "K-Mean Clustering":
                params["k"] = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)
            elif selected_option == "Gamma Correction":
                params["gamma"] = st.sidebar.slider("Select gamma value", 0.1, 5.0, 1.0)

        processed_image = process_image(original_image, selected_option, params)

    # Display side-by-side comparison of original and processed images
    col1, col2 = st.columns(2)
    col1.header("Original Image")
    col1.image(original_image, use_column_width=True, channels="RGB")
    col2.header("Processed Image")
    
    if selected_option == "RGB Splitter":
        # Display each channel separately
        red_channel, green_channel, blue_channel = rgb_splitter(processed_image)
        col2.image(red_channel, use_column_width=True, channels="RGB", caption="Red Channel")
        col2.image(green_channel, use_column_width=True, channels="RGB", caption="Green Channel")
        col2.image(blue_channel, use_column_width=True, channels="RGB", caption="Blue Channel")
    else:
        col2.image(processed_image, use_column_width=True, channels="RGB")

# Download the processed image
if selected_option != "None" and uploaded_image:
    st.sidebar.markdown("### Download Processed Image")
    download_button = st.sidebar.button("Download")
    if download_button:
        im = Image.fromarray((processed_image * 255).astype(np.uint8))  # Convert [0, 1] to [0, 255]
        im.save("processed_image.png")
        st.sidebar.success("Image Downloaded Successfully")

# Footer and credits
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Arya")
