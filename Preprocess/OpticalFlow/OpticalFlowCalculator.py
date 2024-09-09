import cv2
import numpy as np

class OpticalFlowCalculator:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        else:
            self.gpu_available = False

    def calculate(self, frames, bbox=None):
        if len(frames) < 2:
            raise ValueError("Need at least two frames to calculate optical flow.")
        
        try:
            return self._calculate_optical_flow(frames, bbox)
        except Exception as e:
            print(f"Error calculating optical flow: {e}")
            return None

    def _calculate_optical_flow(self, frames, bbox=None):
        optical_flows = []

        ### crop the region of interest (ROI) ### computation speed - phase 2 implementations ###
        prev_frame = self._crop_frame_to_bbox(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), bbox)

        for i in range(1, len(frames)):
            next_frame = self._crop_frame_to_bbox(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), bbox)

            if self.use_gpu and self.gpu_available:
                # If GPU detected, CUDA-based optical flow (now in manual selection)
                prev_frame_gpu = cv2.cuda_GpuMat()
                next_frame_gpu = cv2.cuda_GpuMat()
                prev_frame_gpu.upload(prev_frame)
                next_frame_gpu.upload(next_frame)

                flow_gpu = cv2.cuda_FarnebackOpticalFlow.create(0.5, 3, False, 15, 3, 5, 1.2, 0)
                flow = flow_gpu.calc(prev_frame_gpu, next_frame_gpu, None)
                flow = flow.download() ######## Consider this in phase 3 development
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            optical_flows.append(flow)
            prev_frame = next_frame

        return np.array(optical_flows)

    def _crop_frame_to_bbox(self, frame, bbox):
        if bbox:
            x1, y1, x2, y2 = bbox
            return frame[y1:y2, x1:x2]
        return frame  ######## If no bounding box, return the full frame ########
    


# # Example usage with bounding box
# bbox = [385, 37, 885, 720]  # Bounding box for a specific instance
# optical_flow_calculator = OpticalFlowCalculator(use_gpu=False)

# optical_flows = optical_flow_calculator.calculate(frames, bbox=bbox)