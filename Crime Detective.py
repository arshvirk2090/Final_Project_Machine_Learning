import cv2
import numpy as np

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Function to detect objects using YOLO and draw bounding boxes
def detect_objects(frame):
    height, width, channels = frame.shape

    # Preprocess frame and pass it through the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detections
    class_ids = []
    confidences = []
    boxes = []
    is_suspicious = False
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Check if a gun or knife is detected
                if class_id in [43, 71]:  # Class IDs for gun and knife
                    is_suspicious = True

    # Draw rectangles and labels around detected objects
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for outline
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return is_suspicious


# Function to display warning message
def display_warning(frame, is_suspicious):
    if is_suspicious:
        warning_message = "Warning: Suspicious Activity Detected!"
        color = (0, 0, 255)  # Red color for outline
    else:
        warning_message = "No Suspicious Activity Detected"
        color = (0, 255, 0)  # Green color for outline
    cv2.putText(frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


# Function to open camera in a separate window
def open_camera():
    cap = cv2.VideoCapture(0)  # Access the default camera

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame

        if ret:
            is_suspicious = detect_objects(frame)

            display_warning(frame, is_suspicious)

            cv2.imshow('Camera Feed', frame)  # Display the frame in a window named 'Camera Feed'

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


# Call the function to open the camera
open_camera()