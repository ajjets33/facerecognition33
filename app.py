import streamlit as st
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="wide"
)

def process_image(image):
    """Process image using face_recognition"""
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
        
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image_array)
    
    # Draw boxes around faces
    pil_image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_image)
    
    for (top, right, bottom, left) in face_locations:
        # Draw box
        draw.rectangle(((left, top), (right, bottom)), outline="lime", width=2)
    
    return np.array(pil_image), face_locations

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
                result_image, face_locations = process_image(image)
                
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
                st.write(f"Found {len(face_locations)} faces")
    
    else:  # Camera
        st.info("Note: Camera capture will process frames in real-time")
        
        # Initialize camera
        camera = st.camera_input("Take a photo")
        
        if camera is not None:
            # Read image
            image = Image.open(camera)
            
            # Process image
            with st.spinner("Detecting faces..."):
                result_image, face_locations = process_image(image)
                
                # Display results
                st.subheader("Detected Faces")
                st.image(result_image, use_column_width=True)
                
                # Display detection info
                st.write(f"Found {len(face_locations)} faces")

if __name__ == "__main__":
    main()
