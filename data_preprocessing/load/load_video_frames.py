import cv2
import numpy as np

def load_video_frames(video_path, frame_size=(224, 224)):
    """
    Load frames from a video file and resize them to the desired frame size.
    
    Args:
        video_path (str): Path to the video file.
        frame_size (tuple): Size to resize the frames (default: 224x224).
    
    Returns:
        frames (numpy array): Array of preprocessed frames from the video.
    """
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