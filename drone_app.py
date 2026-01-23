# ----------------------------------
# Import Libraries
# ----------------------------------
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import time

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(
    page_title="Drone Detection", 
    layout="wide")

st.title("🛡️ Autonomous Anti-Drone Detection & Classification System")

# ----------------------------------
# SIDEBAR 
# ----------------------------------
st.sidebar.header("⚙️ Detection Settings")

mode = st.sidebar.selectbox(
    "Choose Input Type", 
    ["Image", "Video"]
)

# ----------------------------------
# Confidence slider 
conf = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05
)
# ----------------------------------
# Decision Confidence slider
decision_conf = st.sidebar.slider(
    "Drone Threat Confidence",
    min_value=0.50,
    max_value=0.90,
    value=0.70,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Threat Rule:**")
st.sidebar.markdown("Drone + High Confidence -> Threat 😱")

# ----------------------------------
# THREAT LOGIC
# ----------------------------------
def threat_decision(class_name, confidence, threshold = 0.7):
    if class_name == "drone" and confidence >= threshold:
        return "THREAT"
    else:
        return "NON-THREAT"



# ----------------------------------
# IMAGE DETECTION + DECISION
# ----------------------------------
if mode == "Image":

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image) 

        st.image(image, caption="Original Image", width="stretch")

        if st.button("Detect Object"):    
            
            results = model.predict(image, conf = conf, iou = 0.5) 

            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.image(annotated, caption="Detect Drone", width="stretch")

            # ---------------- Decision Output ----------------
            st.subheader("Threat Assessment")

            threat_found = False

            for box in results[0].boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[cls_id]

                decision = threat_decision(
                    class_name,
                    confidence,
                    decision_conf
                )

                if decision == "THREAT":
                    
                    threat_found= True
                    st.error(f"DRONE THREAT DETECTED | Confidence: {confidence:.2f}")
                else:
                    st.info(f"{class_name.upper()} detected | Confidence: {confidence:.2f}")
            
            if not threat_found:
                st.success("No drone threat detected")


# ----------------------------------
# VIDEO DETECTION + TRACKING
# ----------------------------------
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        alert_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.time()

            results = model.track(
                frame,
                conf = conf, 
                iou = 0.5,                 # IoU = Intersection over Union. It measures how much two boxes overlap.
                tracker="bytetrack.yaml", 
                persist=True
            )
            
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, width="stretch")

            # ---------------- Decision Logic ----------------
            threat_detected = False

            for box in results[0].boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[cls_id]

                decision = threat_decision(
                    class_name,
                    confidence,
                    decision_conf
                )

                if decision == "THREAT":
                    threat_detected = True

            if threat_detected:
                alert_box.error(f"😱 DRONE THREAT DETECTED 🛩")
            else:
                alert_box.success(f"Airspace Clear 💨")

            time.sleep(0.05)

        cap.release()

