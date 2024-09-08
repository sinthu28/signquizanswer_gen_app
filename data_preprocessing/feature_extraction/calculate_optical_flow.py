import cv2
import numpy as np

def calculate_optical_flow(frames):
    optical_flows = []
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        next_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flows.append(flow)
        prev_frame = next_frame

    return np.array(optical_flows)

    """
    Calculate dense optical flow between consecutive frames.

    Args:
        frames (numpy array): Array of frames (in grayscale).

    Returns:
        optical_flows (numpy array): Optical flow data between consecutive frames.
    """