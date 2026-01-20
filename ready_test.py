import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
IMG_SIZE = 128

try:
    model = keras.models.load_model('face_detection.h5')
except:
    exit()

def preprocess_frame(frame):
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    batched = np.expand_dims(normalized, axis=0)
    return batched

def detect_face(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame, verbose=0)
    confidence = prediction[0][0]
    
    if confidence > 0.5:
        return True, confidence
    else:
        return False, confidence

def draw_result(frame, has_face, confidence):
    if has_face:
        color = (0, 255, 0) 
        text = f"Face: {confidence:.2%}"
    else:
        color = (0, 0, 255) 
        text = f"No Face: {confidence:.2%}"
    
    cv2.putText(frame, text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), color, 3)
    
    cv2.putText(frame, "Press 'Q' to quit", (20, CAMERA_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def run_face_detection():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("@_@")
        return
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        has_face, confidence = detect_face(rgb_frame)
        
        draw_result(frame, has_face, confidence)
        
        cv2.imshow('Face Detection - Neural Network', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_detection()