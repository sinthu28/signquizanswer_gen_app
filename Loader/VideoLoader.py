import cv2
import numpy as np
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoLoader:
    def __init__(self, frame_size=(224, 224), max_frames=None, log_dir='logs'):
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.setup_logging()

    def setup_logging(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"video_loader_{current_date}.log"
        log_path = os.path.join(self.log_dir, log_filename)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

    def load_frames(self, video_path):
        if not os.path.exists(video_path):
            logging.error(f"Video file not found at: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Unable to open video file: {video_path}")
            return None
        
        frames = []
        frame_count = 0

        try:
            while cap.isOpened():
                _bool, frame = cap.read()
                
                if not _bool:
                    logging.info("End of video reached or failed to read frame.")
                    break

                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
                frame_count += 1

                if self.max_frames is not None and frame_count >= self.max_frames:
                    logging.info(f"Maximum frames limit ({self.max_frames}) reached.")
                    break
        
        except cv2.error as e:
            logging.error(f"OpenCV error occurred: {e}")
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if not frames:
            logging.warning("No frames were loaded from the video.")
            return None

        logging.info(f"Successfully loaded {len(frames)} frames from {video_path}")
        return np.array(frames)



    """
        # Create an instance of VideoLoader
        video_loader = VideoLoader(frame_size=(224, 224), max_frames=100)

        # Load frames from a video file
        frames = video_loader.load_frames('path/to/video.mp4')
    """

