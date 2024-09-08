import cv2
import numpy as np

def augment_data(frames, rotate=True, add_noise=True, zoom=True):
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


# import cv2
# import numpy as np

# def augment_data(frames, flip=True, brightness_adjust=True):
#     augmented_frames = []

#     for frame in frames:
#         if flip:
#             frame = cv2.flip(frame, 1)  # Horizontal flip
        
#         if brightness_adjust:
#             factor = np.random.uniform(0.7, 1.3)
#             frame = np.clip(frame * factor, 0, 255).astype(np.uint8)  # Brightness adjustment
        
#         augmented_frames.append(frame)

#     return np.array(augmented_frames)

#     """
#     Augment the gesture data using flipping and brightness adjustment.
    
#     Args:
#         frames (numpy array): Array of frames.
#         flip (bool): Whether to apply horizontal flipping (default: True).
#         brightness_adjust (bool): Whether to adjust the brightness of frames.
    
#     Returns:
#         augmented_frames (numpy array): Augmented frame data.
#     """