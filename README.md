# **Autonomous Anti-Drone Detection & Classification System Using YOLOv11m**
---

**🚀 Live Demo:** 👉 [Click here to view the deployed app](https://autonomous-anti-drone-detection-classification-system.streamlit.app)
---

# **Over View:**

* Unmanned Aerial Vehicles (UAVs), commonly known as drones, are increasingly used across multiple sectors, including border surveillance, security monitoring, and airspace safety.  
However, the growing presence of drones also introduces serious security risks, such as unauthorized intrusions and false threat alerts caused by birds or other aerial objects.

* This project presents a robust, real-time anti-drone detection and classification system built using YOLOv11m (20.03M parameters).  
The system detects and classifies aerial objects and triggers an alert only when a drone is detected with a confidence score greater than 70%, helping reduce false positives and improve operational reliability.
---
# 1. **PROBLEM STATEMENT:**
* The increasing use of drones has created serious security concerns across sensitive areas such as **international borders, restricted military zones, airports, critical infrastructure, tourist locations, and modern warfare environments**.

* Unauthorized drones pose a significant threat in restricted airspace, particularly around borders and airports.  
Manual monitoring systems are ineffective due to the **small size of drones** and their **visual similarity to birds**, which often results in **false alarms or missed detections**.

# **2. OBJECTIVE:**
* This project addresses these challenges by developing an **automated, AI-based anti-drone detection system** that generates alerts **only when the detection confidence crosses a safe threshold**, thereby reducing false alarms and improving detection reliability.

# **3. SYSTEM ARCHITECTURE:**
![SYSTEM ARCHITECTURE](https://github.com/user-attachments/assets/1c7ac727-8c94-453e-8dac-99cae6b6c52d)

# **4. DATA COLLECTION & PREPROCESSING:**
* Data source: https://app.roboflow.com/anti-drone-detection/anti-drone-detection-system/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
* I used a custom dataset with drones, birds, aircraft, and helicopters.

  The dataset was split into two subsets:

Training set: 4,307 images with 4,307 YOLO-format annotation TXT files.

Validation set: 928 images with 928 annotation TXT files.

Each annotation file used YOLO’s normalized format, containing the class ID and bounding box coordinates.

A data.yaml configuration file defined the dataset structure:
<img width="800" height="336" alt="Screenshot 2026-01-24 at 11 57 17 AM" src="https://github.com/user-attachments/assets/dad0c340-9d56-41d9-b57c-c12d955ba85d" />

# **5.TRAINING:**
Training was conducted on Google Colab T4 GPU + with torch-2.9.0+cu126 (Tesla T4, 16GB)
code:
<img width="568" height="226" alt="Screenshot 2026-01-24 at 12 28 52 PM" src="https://github.com/user-attachments/assets/9ca75b83-d85c-462f-bbbe-52d68ea482e2" />
Parameter breakdown:
  * `model=yolov11m.pt`: Pre-trained YOLOv11m weights.
  * `data`: Path to dataset configuration.
  * `epochs=100`: Training cycles (100 times training).
  * `imgsz=640`: Higher resolution for better small-object detection.
  * `batch=16`: Images per batch [Means: 4,307 training images / 16 = 269 batchs]
  * `patience=15`: If the model doesn't get better for 15 rounds in a row, it stops automatically to prevent "overfitting"
  * `workers=8`: Uses 8 CPU threads to load and move images to the GPU as fast as possible.
  * `device=0`: GPU device index.
    
default data augmentation hyperparameters used while training:
  * `mosaic=1.0`: It combines four training images into a single composite image.
  * `close_mosaic=10`: This will automatically turn off the mosaic augmentation for the final 10 epochs (epochs 90–100) to refine the model.
  * `hsv_h=0.015`: Image hue adjustment.
  * `hsv_s (0.7)`: Saturation adjustment.
  * `hsv_v (0.4)`: Value (brightness) adjustment.
  * `fliplr=0.5`: 50% chance of horizontal flipping.
  * `scale=0.5`: Zoom/scale factor of +/- 50%.
  * `translate=0.1`: Image translation (shifting) by 10%.
  * `erasing=0.4`: Randomly "erases" or masks 40% of a rectangular area in the image to help the model handle occlusion.
Training Summary:
  * Model: YOLOv11m(fused): 126 layers, 20,033,116 parameters, 67.7 GFLOPs.

