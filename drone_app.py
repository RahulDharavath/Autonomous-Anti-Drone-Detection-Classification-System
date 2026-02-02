# # ----------------------------------
# # Import Libraries
# # ----------------------------------
# import cv2
# import torch
# import numpy as np
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import tempfile
# import time

# # ----------------------------------
# # Load Model
# # ----------------------------------
# @st.cache_resource
# def load_model():
#     return YOLO("best.pt")

# model = load_model()

# # ----------------------------------
# # Streamlit UI
# # ----------------------------------
# st.set_page_config(
#     page_title="Drone Detection", 
#     layout="wide")

# st.title("🛡️ Autonomous Anti-Drone Detection & Classification System")

# # ----------------------------------
# # SIDEBAR 
# # ----------------------------------
# st.sidebar.header("⚙️ Detection Settings")

# mode = st.sidebar.selectbox(
#     "Choose Input Type", 
#     ["Image", "Video"]
# )

# # ----------------------------------
# # Confidence slider 
# conf = st.sidebar.slider(
#     "Confidence Threshold",
#     min_value=0.10,
#     max_value=0.90,
#     value=0.30,
#     step=0.05
# )
# # ----------------------------------
# # Decision Confidence slider
# decision_conf = st.sidebar.slider(
#     "Drone Threat Confidence",
#     min_value=0.50,
#     max_value=0.90,
#     value=0.70,
#     step=0.05
# )

# st.sidebar.markdown("---")
# st.sidebar.markdown("**Threat Rule:**")
# st.sidebar.markdown("Drone + High Confidence -> Threat 😱")

# # ----------------------------------
# # THREAT LOGIC
# # ----------------------------------
# def threat_decision(class_name, confidence, threshold = 0.7):
#     if class_name == "drone" and confidence >= threshold:
#         return "THREAT"
#     else:
#         return "NON-THREAT"



# # ----------------------------------
# # IMAGE DETECTION + DECISION
# # ----------------------------------
# if mode == "Image":

#     uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

#     if uploaded_image:
#         image = Image.open(uploaded_image) 

#         st.image(image, caption="Original Image", width="stretch")

#         if st.button("Detect Object"):    
            
#             results = model.predict(image, conf = conf, iou = 0.5) 

#             annotated = results[0].plot()
#             annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

#             st.image(annotated, caption="Detect Drone", width="stretch")

#             # ---------------- Decision Output ----------------
#             st.subheader("Threat Assessment")

#             threat_found = False

#             for box in results[0].boxes:
#                 cls_id = int(box.cls)
#                 confidence = float(box.conf)
#                 class_name = model.names[cls_id]

#                 decision = threat_decision(
#                     class_name,
#                     confidence,
#                     decision_conf
#                 )

#                 if decision == "THREAT":
                    
#                     threat_found= True
#                     st.error(f"DRONE THREAT DETECTED | Confidence: {confidence:.2f}")
#                 else:
#                     st.info(f"{class_name.upper()} detected | Confidence: {confidence:.2f}")
            
#             if not threat_found:
#                 st.success("No drone threat detected")


# # ----------------------------------
# # VIDEO DETECTION + TRACKING
# # ----------------------------------
# elif mode == "Video":
#     uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

#     if uploaded_video:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_video.read())

#         cap = cv2.VideoCapture(tfile.name)

#         stframe = st.empty()
#         alert_box = st.empty()

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             start_time = time.time()

#             results = model.track(
#                 frame,
#                 conf = conf, 
#                 iou = 0.5,                 # IoU = Intersection over Union. It measures how much two boxes overlap.
#                 tracker="bytetrack.yaml", 
#                 persist=True
#             )
            
#             annotated_frame = results[0].plot()
#             stframe.image(annotated_frame, width="stretch")

#             # ---------------- Decision Logic ----------------
#             threat_detected = False

#             for box in results[0].boxes:
#                 cls_id = int(box.cls)
#                 confidence = float(box.conf)
#                 class_name = model.names[cls_id]

#                 decision = threat_decision(
#                     class_name,
#                     confidence,
#                     decision_conf
#                 )

#                 if decision == "THREAT":
#                     threat_detected = True

#             if threat_detected:
#                 alert_box.error(f"😱 DRONE THREAT DETECTED 🛩")
#             else:
#                 alert_box.success(f"Airspace Clear 💨")

#             time.sleep(0.05)

#         cap.release()



import cv2
import tempfile
import time
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------
# CONFIGURATION & CONSTANTS
# ----------------------------------
FOCAL_LENGTH = 800      # Calibrated value for 1080p camera
REAL_DRONE_WIDTH = 0.4  # Average width of a drone in meters (e.g. 40cm)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ----------------------------------
# UTILITY FUNCTIONS
# ----------------------------------
def estimate_distance(pixel_width):
    """Geometric distance estimation: D = (W_real * Focal) / W_pixel"""
    if pixel_width > 0:
        return (REAL_DRONE_WIDTH * FOCAL_LENGTH) / pixel_width
    return 0

def threat_decision(class_name, confidence, threshold=0.7):
    """Decision Intelligence logic for Indrajaal-style classification"""
    if class_name == "drone" and confidence >= threshold:
        return "THREAT"
    return "NON-THREAT"

# ----------------------------------
# STREAMLIT UI
# ----------------------------------
st.set_page_config(page_title="Indrajaal AI Node", layout="wide")
st.title("🛡️ Autonomous Drone Defence System (Vision Node)")

st.sidebar.header("⚙️ System Parameters")
mode = st.sidebar.selectbox("Input Stream", ["Video Feed", "Static Image"])
conf_thresh = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3)
threat_thresh = st.sidebar.slider("Threat Trigger Threshold", 0.5, 0.9, 0.7)

# ----------------------------------
# PROCESSING LOGIC
# ----------------------------------
if mode == "Static Image":
    uploaded_file = st.file_uploader("Upload Recon Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        results = model.predict(img, conf=conf_thresh)
        
        annotated = results[0].plot()
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analysis Result")
        
        for box in results[0].boxes:
            cls = model.names[int(box.cls)]
            dist = estimate_distance(float(box.xywh[0][2]))
            status = threat_decision(cls, float(box.conf), threat_thresh)
            
            if status == "THREAT":
                st.error(f"🚨 TARGET IDENTIFIED: {cls.upper()} | Range: {dist:.1f}m")
            else:
                st.info(f"✅ CLEAR: {cls.upper()} | Range: {dist:.1f}m")

elif mode == "Video Feed":
    uploaded_video = st.file_uploader("Upload Drone Footage", type=["mp4", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        st_alert = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # USE BYTETRACK FOR OPTIMAL SPEED
            results = model.track(
                frame, 
                persist=True, 
                conf=conf_thresh, 
                tracker="bytetrack.yaml" 
            )

            if results[0].boxes.id is not None:
                for box in results[0].boxes:
                    # Logic: Get distance and threat status
                    w_pixel = float(box.xywh[0][2])
                    dist = estimate_distance(w_pixel)
                    cls = model.names[int(box.cls)]
                    
                    if threat_decision(cls, float(box.conf), threat_thresh) == "THREAT":
                        st_alert.error(f"😱 DRONE DETECTED IN DOME! Range: {dist:.1f}m")

            annotated_frame = results[0].plot()
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        cap.release()

