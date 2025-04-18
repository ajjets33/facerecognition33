import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Face Detection DNN",
    page_icon="👤",
    layout="wide"
)

# Initialize face detection model
def load_face_detection_model():
    # Load the pre-trained model
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def download_models():
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Download model files if they don't exist
    if not os.path.exists("models/deploy.prototxt"):
        st.info("Downloading model files...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "models/deploy.prototxt"
        )
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "models/res10_300x300_ssd_iter_140000.caffemodel"
        )
        st.success("Model files downloaded successfully!")

def detect_faces(image, net, conf_threshold=0.5):
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    
    # Pass the blob through the network and get detections
    net.setInput(blob)
    detections = net.forward()
    
    # List to store face detections
    faces = []
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > conf_threshold:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            faces.append({
                'box': (startX, startY, endX, endY),
                'confidence': float(confidence)
            })
    
    return faces

def draw_faces(image, faces):
    """Draw bounding boxes and confidence scores on detected faces"""
    img = image.copy()
    
    for face in faces:
        (startX, startY, endX, endY) = face['box']
        confidence = face['confidence']
        
        # Draw the bounding box
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # Draw the confidence score
        text = f"{confidence * 100:.2f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(img, text, (startX, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    return img

def process_image(image, net, conf_threshold=0.5):
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert BGR to RGB if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detect_faces(image, net, conf_threshold)
    
    # Draw faces on image
    result_image = draw_faces(image, faces)
    
    return result_image, faces

def main():
    st.title("Face Detection with DNN")
    
    # Download models if needed
    download_models()
    
    # Load the face detection model
    try:
        net = load_face_detection_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5,
            help="Minimum confidence threshold for face detection"
        )
    
    # Main content
    st.header("Upload Image or Use Webcam")
    
    # Image/Webcam selection
    source = st.radio("Select Source", ["Upload Image", "Use Webcam"])
    
    if source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            
            # Process image
            with st.spinner("Detecting faces..."):
                result_image, faces = process_image(image, net, conf_threshold)
                
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
                st.write(f"Found {len(faces)} faces")
                
                for i, face in enumerate(faces, 1):
                    st.write(f"Face {i}: Confidence = {face['confidence']*100:.2f}%")
    
    else:  # Webcam
        st.info("Note: Webcam capture will process frames in real-time")
        
        # Create a placeholder for webcam feed
        img_placeholder = st.empty()
        
        # Start webcam capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Process frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_frame, faces = process_image(frame, net, conf_threshold)
                
                # Display the frame
                img_placeholder.image(result_frame, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()
