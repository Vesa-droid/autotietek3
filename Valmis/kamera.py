
from ultralytics import YOLO
import cv2
import math
import time
import paho.mqtt.client as mqtt
import json

# MQTT Säädöt
broker_address = "e122536150904cdb94e304fc437a99d1.s1.eu.hivemq.cloud"
username = "Jorma"  # MQTT käyttäjänimi
password = "1234Qwerty"  # MQTT salasana
topic = "yolo/coordinates"

# MQTT client
client = mqtt.Client()
client.username_pw_set(username, password)
client.tls_set()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

client.on_connect = on_connect

# Yhdistys MQTT broukkerille
try:
    client.connect(broker_address, port=8883) # MQTT portti
    client.loop_start()
except Exception as e:
    print(f"Could not connect to MQTT broker: {e}")
    exit()

# Kamera ja YOLO resoluutio
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 960) 

model = YOLO("C:/Python projekti/runs/detect/train20/weights/best.pt")
classNames = ["pohja", "laatikko"]

last_sent_time = 0
send_interval = 2  # Dataa lähetetään 2 sekunin välein (tätä voi muuttaa halutessaan)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            if len(box.conf) > 0:
                confidence = math.ceil((box.conf[0] * 100)) / 100
            else:
                confidence = 0

            cls = int(box.cls[0])
            if cls < len(classNames):
                class_name = classNames[cls]
            else:
                class_name = "Unknown"

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            print(f"Detected: {class_name}, Center: ({center_x}, {center_y}), Confidence: {confidence}")

            
            cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{class_name} {confidence}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            
            current_time = time.time()
            if current_time - last_sent_time > send_interval and class_name == "laatikko":
                last_sent_time = current_time
                payload = {
                    "object": class_name,
                    "confidence": confidence,
                    "center_coordinates": {
                        "x": center_x,
                        "y": center_y
                    }
                }
                print(f"Preparing to publish payload: {payload}")
                try:
                    result = client.publish(topic, json.dumps(payload))
                    if result.rc == 0:
                        print("Successfully published payload to MQTT")
                    else:
                        print(f"Failed to publish payload, result code: {result.rc}")
                except Exception as e:
                    print(f"Failed to publish to MQTT: {e}")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.loop_stop()  
client.disconnect() 