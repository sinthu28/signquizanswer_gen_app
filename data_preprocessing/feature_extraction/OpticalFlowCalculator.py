import cv2
import numpy as np

class OpticalFlowCalculator:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu

    def calculate(self, frames):
        try:
            return self._calculate_optical_flow(frames)
        except Exception as e:
            print(f"Error calculating optical flow: {e}")
            return None

    def _calculate_optical_flow(self, frames):
        optical_flows = []
        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            next_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            optical_flows.append(flow)
            prev_frame = next_frame

        return np.array(optical_flows)

