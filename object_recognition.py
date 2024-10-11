# import torch
# import torchvision
# import streamlit as st
# import gdown
# import os
# import cv2
# import numpy as np
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor
# from detectron2.visualizer import Visualizer

# # Constants
# MODEL_WEIGHTS_URL = 'https://drive.google.com/file/d/12U_q40a1cusJiCmgPK-9g4JFjFF6_rq2/view?usp=drive_link'  # Replace with your Google Drive file ID
# MODEL_WEIGHTS_PATH = 'model_weights.pth'
# CONFIG_FILE_PATH = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'

# # Download model weights if they don't exist
# if not os.path.exists(MODEL_WEIGHTS_PATH):
#     gdown.download(MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH, quiet=False)

# # Load model configuration
# cfg = get_cfg()
# cfg.merge_from_file(CONFIG_FILE_PATH)
# cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
# cfg.MODEL.EVAL_MODE = True
# predictor = DefaultPredictor(cfg)

# def run_inference(image):
#     # Prepare image for the model
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     outputs = predictor(image)
#     return outputs

# # Streamlit UI
# st.title("Object Detection App")
# mode = st.selectbox("Choose Input Mode", ("Upload Image", "Use Webcam"))

# if mode == "Upload Image":
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         # Read the image
#         image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
#         outputs = run_inference(image)

#         # Visualize the results
#         v = Visualizer(image[:, :, ::-1], metadata=None, scale=1.2)
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         st.image(out.get_image()[:, :, ::-1], caption="Processed Image")

# elif mode == "Use Webcam":
#     run = st.checkbox("Run")
#     frame_window = st.image([])

#     if run:
#         cap = cv2.VideoCapture(0)
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 st.write("Unable to capture video")
#                 break
            
#             # Run inference
#             outputs = run_inference(frame)

#             # Visualize the results
#             v = Visualizer(frame[:, :, ::-1], metadata=None, scale=1.2)
#             out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#             frame_window.image(out.get_image()[:, :, ::-1], channels="RGB")

#         cap.release()

# st.write("Instructions: Select an input mode to test the model.")


# Import necessary libraries
import streamlit as st
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import gdown
import os

# Download model weights and YAML config from Google Drive
def download_files():
    # Define Google Drive file IDs
    weights_file_id = '12U_q40a1cusJiCmgPK-9g4JFjFF6_rq2'  # Model weights file ID
    yaml_file_id = '19uMLkgQAtnlpxpZZF_T3CeqEsWip4Qs7'  # YAML file ID

    # Define file names to save
    weights_file = 'model_final.pth'
    yaml_file = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    
    # Download files
    gdown.download(f'https://drive.google.com/uc?id={weights_file_id}', weights_file, quiet=False)
    gdown.download(f'https://drive.google.com/uc?id={yaml_file_id}', yaml_file, quiet=False)

# Function to load the model
@st.cache_resource
def load_model():
    download_files()  # Ensure files are downloaded before loading the model
    cfg = get_cfg()
    cfg.merge_from_file("faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Use the downloaded YAML file
    cfg.MODEL.WEIGHTS = "model_final.pth"  # Use the downloaded model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    predictor = DefaultPredictor(cfg)
    return predictor

# Function to process image
def process_image(image, predictor):
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]

# Function to process video frame by frame
def process_video(video_file, predictor):
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = process_image(frame, predictor)
        out.write(result_frame)
    
    cap.release()
    out.release()

# Function to process webcam stream
def process_webcam(predictor):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = process_image(frame, predictor)
        stframe.image(result_frame, channels="BGR")
        
    cap.release()

# Streamlit app
def main():
    st.title("Object Detection with Detectron2")
    st.sidebar.title("Options")
    
    # Load model
    predictor = load_model()
    
    # Choose input type
    input_type = st.sidebar.selectbox("Select Input Type", ["Image", "Video", "Webcam"])
    
    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                result_image = process_image(image, predictor)
                st.image(result_image, caption='Processed Image', use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            video_path = "temp_video.mp4"
            with open(video_path, mode='wb') as f:
                f.write(uploaded_file.read())  # Save uploaded video to disk
            process_video(video_path, predictor)
            st.video("output.mp4")  # Display processed video
    
    elif input_type == "Webcam":
        st.write("Starting webcam...")
        process_webcam(predictor)

if __name__ == "__main__":
    main()
