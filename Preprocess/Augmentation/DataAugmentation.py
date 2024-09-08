import cv2
import numpy as np

class DataAugmentation:
    def __init__(self, advanced=True, flip=True, brightness_adjust=True, rotate=True, add_noise=True, zoom=True):
        self.advanced = advanced
        self.flip = flip
        self.brightness_adjust = brightness_adjust
        self.rotate = rotate
        self.add_noise = add_noise
        self.zoom = zoom

    def augment(self, frames):
        try:
            if self.advanced:
                return self.advanced_augment(frames)
            else:
                return self.basic_augment(frames)
        except Exception as e:
            print(f"Error during augmentation: {e}")
            return frames

    def basic_augment(self, frames):
        augmented_frames = []
        for frame in frames:
            if self.flip:
                frame = cv2.flip(frame, 1)
            if self.brightness_adjust:
                factor = np.random.uniform(0.7, 1.3)
                frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
            augmented_frames.append(frame)
        return np.array(augmented_frames)

    def advanced_augment(self, frames):
        augmented_frames = []
        for frame in frames:
            if self.rotate:
                angle = np.random.uniform(-15, 15)
                center = (frame.shape[1] // 2, frame.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
                augmented_frames.append(rotated)

            if self.add_noise:
                noise = np.random.normal(0, 0.1, frame.shape)
                noisy_frame = frame + noise * 255
                noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
                augmented_frames.append(noisy_frame)

            if self.zoom:
                zoom_factor = np.random.uniform(1.0, 1.3)
                h, w, _ = frame.shape
                new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
                top, left = (h - new_h) // 2, (w - new_w) // 2
                zoomed_frame = frame[top:top + new_h, left:left + new_w]
                zoomed_frame = cv2.resize(zoomed_frame, (w, h))
                augmented_frames.append(zoomed_frame)

        return np.array(augmented_frames)
