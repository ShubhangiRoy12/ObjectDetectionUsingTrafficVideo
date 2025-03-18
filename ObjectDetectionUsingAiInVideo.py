import cv2

config_file = 'C:\\Users\\sroya\\Downloads\\ObjectDetectionUsingAi\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'C:\\Users\\sroya\\Downloads\\ObjectDetectionUsingAi\\frozen_inference_graph.pb'

model = cv2.dnn.DetectionModel(frozen_model,config_file)

classLabels = []
file_name = 'C:\\Users\\sroya\\Downloads\\ObjectDetectionUsingAi\\LABELS.txt' #text file path
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

cap = cv2.VideoCapture('C:\\Users\\sroya\\Downloads\\object detection materials\\mixkit-busy-street-in-the-city-4000-medium.mp4')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ClassIndex , confidence , bbox = model.detect(frame,confThreshold=0.5)

    for ClassInd , conf , boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        if (ClassInd-1) < len(classLabels):  # Add this check
            # Draw the bounding box
            cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[0]+boxes[2], boxes[1]+boxes[3]), (255, 0, 0), 2)
            # Put the class label and confidence score
            text = f"{classLabels[ClassInd-1]}: {conf:.2f}"
            cv2.putText(frame, text, (boxes[0], boxes[1]-10), font ,fontScale=font_scale, color=(0,255,0),thickness=3)

    # Display the frame with bounding boxes
    cv2.imshow('Video Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video Object Detection',cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
