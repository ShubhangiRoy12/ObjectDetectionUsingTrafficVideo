# ObjectDetectionUsingTrafficVideo
## Overview
This project performs real-time object detection on a traffic video using **SSD MobileNet V3** with OpenCV’s DNN module. It identifies objects such as pedestrians, vehicles, and traffic signs, drawing bounding boxes with confidence scores.

## Features
- Uses **SSD MobileNet V3** for object detection  
- Reads class labels from `LABELS.txt`  
- Processes video frame-by-frame  
- Draws bounding boxes with confidence scores  
- Stops detection when 'q' is pressed  

## Requirements
- Python 3.x  
- OpenCV (`cv2`)  

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ObjectDetectionTraffic.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ObjectDetectionTraffic
   ```
3. Install dependencies:
   ```bash
   pip install opencv-python
   ```

## Usage
1. Ensure you have the following files:
   - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` (Model Configuration)  
   - `frozen_inference_graph.pb` (Trained Model Weights)  
   - `LABELS.txt` (Class Labels)  
   - Input video file (e.g., `mixkit-busy-street-in-the-city-4000-medium.mp4`)  

2. Run the detection script:
   ```bash
   python detect_video.py
   ```

## Code Explanation
- **Load the model**:  
  ```python
  model = cv2.dnn.DetectionModel(frozen_model, config_file)
  ```
- **Read class labels**:
  ```python
  with open(file_name, 'rt') as fpt:
      classLabels = fpt.read().rstrip('\n').split('\n')
  ```
- **Initialize video capture**:
  ```python
  cap = cv2.VideoCapture('video.mp4')
  ```
- **Preprocess frames for the model**:
  ```python
  model.setInputSize(320,320)
  model.setInputScale(1.0/127.5)
  model.setInputMean((127.5,127.5,127.5))
  model.setInputSwapRB(True)
  ```
- **Perform detection on each frame**:
  ```python
  ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
  ```
- **Draw bounding boxes and labels**:
  ```python
  for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
      cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[0]+boxes[2], boxes[1]+boxes[3]), (255, 0, 0), 2)
      text = f"{classLabels[ClassInd-1]}: {conf:.2f}"
      cv2.putText(frame, text, (boxes[0], boxes[1]-10), font, fontScale=font_scale, color=(0,255,0), thickness=3)
  ```
- **Display the video output**:
  ```python
  cv2.imshow('Video Object Detection', frame)
  ```
- **Exit detection when 'q' is pressed**:
  ```python
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  ```

## Sample Output
The script detects objects in the video and overlays bounding boxes and labels in real-time.
