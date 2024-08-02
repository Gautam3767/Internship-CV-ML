import cv2
from gtts import gTTS
from playsound import playsound

# Load the TensorFlow model and config file
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load the class labels
classLabels = []
filename = 'yolo3.txt'
with open(filename, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print("Number of Classes")
print(len(classLabels))
print("Class labels")
print(classLabels)

# Model training
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()

    # Perform object detection
    class_ids, confidences, bbox = model.detect(frame, confThreshold=0.5)

    # Draw bounding boxes and labels on the frame
    for i in range(len(class_ids)):
        class_id = class_ids[i]
        confidence = confidences[i]
        box = bbox[i]

        label = classLabels[class_id - 1]
        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if label not in labels:
            labels.append(label)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print(labels)
