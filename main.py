import cv2
import torch
import numpy as np
from deepface import DeepFace

# Load YOLOv5 model for face detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO face detection
    results = yolo_model(frame)
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        
        # Convert face to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        try:
            # Get face embedding
            face_embedding = DeepFace.represent(face_rgb, model_name="Facenet")
            
            # Save extracted face embedding
            np.save("user_face.npy", face_embedding)
            print("Face data saved successfully!")
            break  # Stop after first face detection
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
    
    cv2.imshow("Face Extraction", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
