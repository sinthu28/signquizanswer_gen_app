import cv2
import numpy as np

def load_video_frames(video_path, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames)