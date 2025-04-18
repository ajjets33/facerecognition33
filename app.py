import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="wide"
)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def process_image(image, min_detection_confidence=0.5):
    """Process image using MediaPipe Face Detection"""
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_detection_confidence
    ) as face_detection:
        # Convert PIL Image to numpy array if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB
        image_rgb = image.copy()
        
        # Process the image and detect faces
        results = face_detection.process(image_rgb)
        
        # Draw face detections
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image_rgb, detection)
        
        return image_rgb, results.detections if results.detections else []

def main():
    st.title("Face Detection App")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        min_detection_confidence = st.slider(
            "Detection Confidence",
            0.0, 1.0, 0.5,
            help="Minimum confidence threshold for face detection"
        )
    
    # Main content
    st.header("Upload Image or Use Camera")
    
    # Image/Camera selection
    source = st.radio("Select Source", ["Upload Image", "Use Camera"])
    
    if source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            
            # Process image
            with st.spinner("Detecting faces..."):
                result_image, detections = process_image(
                    image, 
                    min_detection_confidence
                )
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Detected Faces")
                    st.image(result_image, use_column_width=True)
                
                # Display detection info
                st.subheader("Detection Results")
                st.write(f"Found {len(detections)} faces")
    
    else:  # Camera
        st.info("Note: Camera capture will process frames in real-time")
        
        # Initialize camera
        camera = st.camera_input("Take a photo")
        
        if camera is not None:
            # Read image
            image = Image.open(camera)
            
            # Process image
            with st.spinner("Detecting faces..."):
                result_image, detections = process_image(
                    image,
                    min_detection_confidence
                )
                
                # Display results
                st.subheader("Detected Faces")
                st.image(result_image, use_column_width=True)
                
                # Display detection info
                st.write(f"Found {len(detections)} faces")

if __name__ == "__main__":
    main()
