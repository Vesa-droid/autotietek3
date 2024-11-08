
from ultralytics import YOLO
import cv2
import math

# Try different indices for the USB camera if 0 doesn't work. Start with 1.
cap = cv2.VideoCapture(1)  # Change the index to 1 or 2 if your USB camera is not at index 0
cap.set(3, 640)  # Set the width of the video frame
cap.set(4, 480)  # Set the height of the video frame

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes (you can change these according to your custom classes)
classNames = ["pohja", "laatikko"]

while True:
    success, img = cap.read()  # Read a frame from the USB camera
    if not success:  # If the frame wasn't captured properly
        print("Failed to capture image")
        break

    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            if len(box.conf) > 0:
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)
            else:
                print("No confidence score available for this box.")
                confidence = 0  # tai aseta joku oletusarvo, esim. 0
                
            # class name
            cls = int(box.cls[0])

            # Tarkistetaan, että cls on listan sisällä
            if cls < len(classNames):
                class_name = classNames[cls]
            else:
                print(f"Class index {cls} out of range, using 'Unknown'")
                class_name = "Unknown"  # Käytetään oletusluokkaa

            # calculate center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # print the center coordinates
            print(f"Center of {class_name}: ({center_x}, {center_y})")

            # draw the center point (circle) on the image
            cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)  # Green dot at the center

            # draw the class name and confidence
            org = (x1, y1 - 10)  # Positioning text above the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)  # Blue color for text
            thickness = 2
            cv2.putText(img, f"{class_name} {confidence}", org, font, fontScale, color, thickness)

    # display the result
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):  # press 'q' to quit
        break

cap.release()  # release the video capture object
cv2.destroyAllWindows()  # close all OpenCV windows