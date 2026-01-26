#     <h1 align="center">  Autonomous Anti-Drone Detection & Classification System Using YOLOv11m </h1>

---

**🚀 Live Demo:** 👉 [Click here to view the deployed app](https://autonomous-anti-drone-detection-classification-system.streamlit.app)

---

<img width="1000" height="8000" alt="Drone" src="https://github.com/user-attachments/assets/4199276f-29e2-463e-b58c-46f911692c92" />


---

## Overview

Unmanned Aerial Vehicles (UAVs), commonly known as drones, are increasingly used across multiple sectors, including border surveillance, security monitoring, and airspace safety. 
However, the growing presence of drones also introduces serious security risks, such as unauthorized intrusions and false threat alerts caused by birds or other aerial objects.

This project presents a robust, real-time anti-drone detection and classification system built using YOLOv11m (20.03M parameters).  
The system detects and classifies aerial objects and triggers an alert only when a drone is detected with a confidence score greater than 70%, helping reduce false positives and improve operational reliability.

---

## 1. Problem Statement

The increasing use of drones has created serious security concerns across sensitive areas such as **international borders, restricted military zones, airports, critical infrastructure, tourist locations, and modern warfare environments**.

---

## 2. Objective

The objective of this project is to develop an **automated, AI-based anti-drone detection system** that triggers alerts **only when the detection confidence crosses a safe threshold**, thereby **reducing** false alarms and **improving** detection reliability.

---

## 3. System Architecture

![SYSTEM ARCHITECTURE](https://github.com/user-attachments/assets/1c7ac727-8c94-453e-8dac-99cae6b6c52d)

---

## 4. Data Collection & Preprocessing

- **Data Source:**
  
  https://app.roboflow.com/anti-drone-detection/anti-drone-detection-system/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
  
- A custom dataset containing **drones, birds, aircraft, and helicopters** was used.

 ### Dataset Split: 

**Training set:** 4,307 images with 4,307 YOLO-format annotation TXT files.

**Validation set:** 928 images with 928 annotation TXT files.

Each annotation file used YOLO’s normalized format, containing the class ID and bounding box coordinates.

A `data.yaml` configuration file was used to define dataset paths & class labels:

    import yaml
    
    yaml_data = {
        'path': 'Anti-Drone-Detection-System--1',
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 4,                # No.of classes
        'names': ['aircraft', 'bird', 'helicopter', 'drone']
    }           # ID's :- aircraft -> 0, bird -> 1, helicopter -> 2, 'drone -> 3
    
    
    with open('Anti-Drone-Detection-System--1/data.yaml', 'w') as file:
        yaml.dump(yaml_data, file)
    
    print("saved to data.yaml")


---

## 5.Training

Training was conducted on **Google Colab using an NVIDIA Tesla T4 GPU (16GB)** with **PyTorch 2.9.0 + CUDA 12.6**.

### code:

    #  Train the YOLO model on custom dataset
    model.train(
        data="Anti-Drone-Detection-System--1/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        workers=8,
        device=0
    )


### Parameter breakdown:

  * `model=yolov11m.pt`: Pre-trained YOLOv11m weights.
  * `data`: Path to dataset configuration.
  * `epochs=100`: Training cycles (100 times training).
  * `imgsz=640`: Higher resolution for better small-object detection.
  * `batch=16`: Images per batch [Means: 4,307 training images / 16 = 269 batchs]
  * `patience=15`: If the model doesn't get better for 15 rounds in a row, it stops automatically to prevent "overfitting"
  * `workers=8`: Uses 8 CPU threads to load and move images to the GPU as fast as possible.
  * `device=0`: GPU device index.
    
### Data Augmentation (Default Hyperparameters):
  * `mosaic=1.0`: It combines four training images into a single composite image.
  * `close_mosaic=10`: This will automatically turn off the mosaic augmentation for the final 10 epochs (epochs 90–100) to refine the model.
  * `hsv_h=0.015`: Image hue adjustment.
  * `hsv_s (0.7)`: Saturation adjustment.
  * `hsv_v (0.4)`: Value (brightness) adjustment.
  * `fliplr=0.5`: 50% chance of horizontal flipping.
  * `scale=0.5`: Zoom/scale factor of +/- 50%.
  * `translate=0.1`: Image translation (shifting) by 10%.
  * `erasing=0.4`: Randomly "erases" or masks 40% of a rectangular area in the image to help the model handle occlusion.

### Training Summary:
  * Model: YOLOv11m(fused): 126 layers, 20,033,116 parameters, 67.7 GFLOPs.
  * mAP50: 0.957, mAP50-95: 0.724.
  * Speed: ~9.1 ms inference per image.
 Training results were saved in `runs/detect/train`.

---

# **6. Results**

The model was evaluated using standard object detection metrics, including **precision, recall, and mean Average Precision (mAP)**.

### Validation Performance (928 images / 928 instances):
- **Precision:** 0.967  
- **Recall:** 0.900  
- **mAP@50:** 0.957  
- **mAP@50–95:** 0.724

![Results](https://github.com/user-attachments/assets/8d5da4c2-99bb-4712-9223-14fad83e5a79)

---

## 7. Inference & Tracking

After training, the YOLOv11m model was used for real-time drone detection on images and video streams.

For object persistence across frames, detection outputs can be integrated with **multi-object tracking (MOT) algorithms** such as **ByteTrack or DeepSORT**, enabling stable tracking even under partial occlusion or rapid motion.

### Sample Inference Outputs:

https://github.com/user-attachments/assets/a1c808ac-39ea-4e42-9038-74bde8cf91cb

https://github.com/user-attachments/assets/b0dc01b3-a7e2-4fe0-b559-b85374e8fc8a

https://github.com/user-attachments/assets/d83c8373-9752-4ff7-b461-9370d7014e09

---

## 8. Deployment

The trained model was deployed using **Streamlit**, providing an interactive web-based interface for real-time drone detection and alert visualization.









