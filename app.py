import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="wide"
)

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return img

def process_uploaded_image(image):
    # Convert the image to numpy array
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img, len(faces)

def main():
    st.title("Face Detection App")
    
    # Add tabs for different functionalities
    tab1, tab2 = st.tabs(["Live Detection", "Image Upload"])
    
    with tab1:
        st.header("Live Face Detection")
        st.write("Click 'START' to begin face detection with your webcam")
        
        webrtc_streamer(
            key="example",
            video_transformer_factory=VideoTransformer,
            async_transform=True
        )
    
    with tab2:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            
            # Process image
            with st.spinner('Detecting faces...'):
                result_image, face_count = process_uploaded_image(image)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Detected Faces")
                    st.image(result_image, use_column_width=True, channels="BGR")
                
                # Display detection info
                st.success(f"Found {face_count} faces in the image!")

if __name__ == "__main__":
    main()

