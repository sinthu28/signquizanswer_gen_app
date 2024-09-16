import logging
import os
import cv2
import numpy as np

class VideoLoader:
    def __init__(self, log_dir='logs'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.logger = logging.getLogger(__name__)
        log_file = os.path.join(log_dir, "frame_normaliser.log")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def load_frames(self, video_path: str, max_frames: int = None) -> np.ndarray:
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file: {video_path}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break

            cap.release()
            if len(frames) == 0:
                self.logger.warning(f"No frames extracted from video: {video_path}")
            return np.array(frames)

        except Exception as e:
            self.logger.error(f"Error loading frames from video {video_path}: {e}")
            return None