import cv2
import numpy as np

def advanced_augment_data(frames, rotate=True, add_noise=True, zoom=True):
    """
    Apply advanced data augmentation techniques such as rotations, noise, and zoom.
    
    Args:
        frames (numpy array): Array of frames.
        rotate (bool): Whether to apply random rotations.
        add_noise (bool): Whether to add random noise to frames.
        zoom (bool): Whether to apply random zoom on frames.
    
    Returns:
        augmented_frames (numpy array): Augmented frame data.
    """
    augmented_frames = []

    for frame in frames:
        if rotate:
            angle = np.random.uniform(-15, 15)
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
            augmented_frames.append(rotated)
        
        if add_noise:
            noise = np.random.normal(0, 0.1, frame.shape)
            noisy_frame = frame + noise * 255
            noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
            augmented_frames.append(noisy_frame)

        if zoom:
            zoom_factor = np.random.uniform(1.0, 1.3)
            h, w, _ = frame.shape
            new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
            top, left = (h - new_h) // 2, (w - new_w) // 2
            zoomed_frame = frame[top:top + new_h, left:left + new_w]
            zoomed_frame = cv2.resize(zoomed_frame, (w, h))
            augmented_frames.append(zoomed_frame)

    return np.array(augmented_frames)