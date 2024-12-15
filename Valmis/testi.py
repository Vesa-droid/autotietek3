import os
import cv2
import random
import time

# Set paths
train_images_dir = "dataset/train/images"
train_labels_dir = "dataset/train/labels"
class_names = ["pohja", "laatikko"]

def draw_bboxes(image, label_path):
    # Read label file and draw bounding boxes
    with open(label_path, "r") as file:
        for line in file:
            # YOLO format: class_id center_x center_y width height (normalized)
            class_id, x, y, w, h = map(float, line.strip().split())
            class_id = int(class_id)
            
            # Convert normalized values to image dimensions
            img_h, img_w = image.shape[:2]
            x1 = int((x - w / 2) * img_w)
            y1 = int((y - h / 2) * img_h)
            x2 = int((x + w / 2) * img_w)
            y2 = int((y + h / 2) * img_h)
            
            # Draw rectangle and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            label = f"{class_names[class_id]}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue label

    return image

# Display loop
while True:
    # Randomly select an image and its corresponding label
    image_file = random.choice(os.listdir(train_images_dir))
    image_path = os.path.join(train_images_dir, image_file)
    label_path = os.path.join(train_labels_dir, os.path.splitext(image_file)[0] + ".txt")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        continue

    # Draw bounding boxes
    image_with_bboxes = draw_bboxes(image, label_path)
    
    # Display the image
    cv2.imshow("Training Feed", image_with_bboxes)
    
    # Display each frame for a short period
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit
        break
    
    # Optional delay to simulate a feed (adjust as desired)
    time.sleep(0.5)

cv2.destroyAllWindows()