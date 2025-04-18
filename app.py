import streamlit as st
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Image Upload Demo",
    page_icon="📸",
    layout="wide"
)

def main():
    st.title("Image Upload and Camera Demo")
    
    # Main content
    st.header("Upload Image or Use Camera")
    
    # Image/Camera selection
    source = st.radio("Select Source", ["Upload Image", "Use Camera"])
    
    if source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add some image info
            st.subheader("Image Information")
            st.write(f"Image size: {image.size}")
            st.write(f"Image mode: {image.mode}")
    
    else:  # Camera
        # Initialize camera
        camera = st.camera_input("Take a photo")
        
        if camera is not None:
            # Read and display image
            image = Image.open(camera)
            st.image(image, caption="Captured Photo", use_column_width=True)
            
            # Add some image info
            st.subheader("Image Information")
            st.write(f"Image size: {image.size}")
            st.write(f"Image mode: {image.mode}")

if __name__ == "__main__":
    main()
