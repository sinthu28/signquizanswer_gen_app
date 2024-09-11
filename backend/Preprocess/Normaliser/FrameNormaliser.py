import os
import numpy as np
import logging

class FrameNormaliser:
    def __init__(self, dtype=np.float32, method="standard", log_dir="logs"):
        self.dtype = dtype
        self.method = method
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.logger = logging.getLogger(__name__)
        log_file = os.path.join(log_dir, "frame_normaliser.log")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def normalize(self, frames):
        if not isinstance(frames, np.ndarray):
            self.logger.error("Input frames must be a NumPy array.")
            raise ValueError("Input frames must be a NumPy array.")
        
        if frames.size == 0:
            self.logger.error("Input frames array is empty.")
            raise ValueError("Input frames array is empty.")
        
        if frames.ndim not in [3, 4]:
            self.logger.error("Input frames must be 3D (single video) or 4D (batch of videos).")
            raise ValueError("Input frames must be 3D (single video) or 4D (batch of videos).")

        self.logger.info(f"Normalizing frames with method '{self.method}' and dtype '{self.dtype}'.")

        if self.method == "standard":
            normalized_frames = frames.astype(self.dtype) / 255.0
        elif self.method == "zscore":
            mean = np.mean(frames)
            std = np.std(frames)
            normalized_frames = (frames - mean) / std
        elif self.method == "minmax":
            min_val = np.min(frames)
            max_val = np.max(frames)
            normalized_frames = (frames - min_val) / (max_val - min_val)
        else:
            self.logger.error(f"Unknown normalization method: {self.method}")
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return normalized_frames
    



    # Initializes the FrameNormaliser class with options for normalization type.

    # Args:
    #         dtype: Data type for frames, default is np.float32.
    #         method: Normalization method - "standard" (div by 255), "zscore" (z-score normalization), "minmax" (min-max scaling).
    #         log_dir: Directory to save logs.
    
    #      Normalizes the input video frames based on the specified method.

    #         Args:
    #             frames: 3D or 4D NumPy array of video frames.

    #         Returns:
    #             Normalized frames as a NumPy array.

