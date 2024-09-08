import cv2
import numpy as np

def augment_data(frames, flip=True, brightness_adjust=True):
    """
    Augment the gesture data using flipping and brightness adjustment.
    
    Args:
        frames (numpy array): Array of frames.
        flip (bool): Whether to apply horizontal flipping (default: True).
        brightness_adjust (bool): Whether to adjust the brightness of frames.
    
    Returns:
        augmented_frames (numpy array): Augmented frame data.
    """
    augmented_frames = []

    for frame in frames:
        if flip:
            frame = cv2.flip(frame, 1)  # Horizontal flip
        
        if brightness_adjust:
            factor = np.random.uniform(0.7, 1.3)
            frame = np.clip(frame * factor, 0, 255).astype(np.uint8)  # Brightness adjustment
        
        augmented_frames.append(frame)

    return np.array(augmented_frames)