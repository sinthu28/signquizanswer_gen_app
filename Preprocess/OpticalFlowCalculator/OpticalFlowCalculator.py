import cv2
import numpy as np

class OpticalFlowCalculator:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        else:
            self.gpu_available = False

    def calculate(self, frames):
        if len(frames) < 2:
            raise ValueError("Need at least two frames to calculate optical flow.")
        
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

            if self.use_gpu and self.gpu_available:
                # If GPU is available and desired, use CUDA-based optical flow (if applicable)
                prev_frame_gpu = cv2.cuda_GpuMat()
                next_frame_gpu = cv2.cuda_GpuMat()
                prev_frame_gpu.upload(prev_frame)
                next_frame_gpu.upload(next_frame)

                flow_gpu = cv2.cuda_FarnebackOpticalFlow.create(0.5, 3, False, 15, 3, 5, 1.2, 0)
                flow = flow_gpu.calc(prev_frame_gpu, next_frame_gpu, None)
                flow = flow.download()  # Download result from GPU
            else:
                # CPU-based optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            optical_flows.append(flow)
            prev_frame = next_frame

        return np.array(optical_flows)