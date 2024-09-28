pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install git+https://github.com/facebookresearch/detectron2.git@main


import streamlit as st
import gdown
import os
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.visualizer import Visualizer

# Constants
MODEL_WEIGHTS_URL = 'https://drive.google.com/file/d/12U_q40a1cusJiCmgPK-9g4JFjFF6_rq2/view?usp=drive_link'  # Replace with your Google Drive file ID
MODEL_WEIGHTS_PATH = 'model_weights.pth'
CONFIG_FILE_PATH = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'

# Download model weights if they don't exist
if not os.path.exists(MODEL_WEIGHTS_PATH):
    gdown.download(MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH, quiet=False)

# Load model configuration
cfg = get_cfg()
cfg.merge_from_file(CONFIG_FILE_PATH)
cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
cfg.MODEL.EVAL_MODE = True
predictor = DefaultPredictor(cfg)

def run_inference(image):
    # Prepare image for the model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    outputs = predictor(image)
    return outputs

# Streamlit UI
st.title("Object Detection App")
mode = st.selectbox("Choose Input Mode", ("Upload Image", "Use Webcam"))

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image
        image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
        outputs = run_inference(image)

        # Visualize the results
        v = Visualizer(image[:, :, ::-1], metadata=None, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1], caption="Processed Image")

elif mode == "Use Webcam":
    run = st.checkbox("Run")
    frame_window = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Unable to capture video")
                break
            
            # Run inference
            outputs = run_inference(frame)

            # Visualize the results
            v = Visualizer(frame[:, :, ::-1], metadata=None, scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            frame_window.image(out.get_image()[:, :, ::-1], channels="RGB")

        cap.release()

st.write("Instructions: Select an input mode to test the model.")
