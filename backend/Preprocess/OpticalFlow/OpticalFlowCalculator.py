import cv2
import numpy as np
import logging
import os
from datetime import datetime

class OpticalFlowCalculator:
    def __init__(self, use_gpu=False, log_dir='logs'):
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        else:
            self.gpu_available = False

        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.setup_logging()

    def setup_logging(self):
        """Sets up logging to both file and console."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"OpticalFlowCalculator_{current_date}.log"
        log_path = os.path.join(self.log_dir, log_filename)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def calculate(self, frames, bbox=None):
        if len(frames) < 2:
            self.logger.error("Need at least two frames to calculate optical flow.")
            raise ValueError("Need at least two frames to calculate optical flow.")
        
        try:
            self.logger.info(f"Calculating optical flow for {len(frames)} frames.")
            return self._calculate_optical_flow(frames, bbox)
        except Exception as e:
            self.logger.error(f"Error calculating optical flow: {e}")
            return None

    def _calculate_optical_flow(self, frames, bbox=None):
        optical_flows = []
        
        self.logger.info(f"Processing optical flow using {'GPU' if self.use_gpu and self.gpu_available else 'CPU'}")

        prev_frame = self._crop_frame_to_bbox(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), bbox)

        for i in range(1, len(frames)):
            next_frame = self._crop_frame_to_bbox(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), bbox)

            if self.use_gpu and self.gpu_available:
                self.logger.info(f"Using CUDA-based optical flow for frame {i}")
                prev_frame_gpu = cv2.cuda_GpuMat()
                next_frame_gpu = cv2.cuda_GpuMat()
                prev_frame_gpu.upload(prev_frame)
                next_frame_gpu.upload(next_frame)

                flow_gpu = cv2.cuda_FarnebackOpticalFlow.create(0.5, 3, False, 15, 3, 5, 1.2, 0)
                flow = flow_gpu.calc(prev_frame_gpu, next_frame_gpu, None)
                flow = flow.download()  # Download flow back to the CPU for further use
            else:
                self.logger.info(f"Using CPU-based optical flow for frame {i}")
                flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            optical_flows.append(flow)
            prev_frame = next_frame

        self.logger.info(f"Successfully calculated optical flow for {len(optical_flows)} pairs of frames.")
        return np.array(optical_flows)

    def _crop_frame_to_bbox(self, frame, bbox):
        if bbox:
            x1, y1, x2, y2 = bbox
            self.logger.info(f"Cropping frame to bounding box {bbox}")
            return frame[y1:y2, x1:x2]
        self.logger.info("No bounding box provided, using full frame.")
        return frame  # If no bounding box, return the full frame

