import cv2
import numpy as np

class DataAugmentation:
    def __init__(self, advanced=True, flip=True, brightness_adjust=True, rotate=True, add_noise=True, zoom=True, bbox=None):
        self.advanced = advanced
        self.flip = flip
        self.brightness_adjust = brightness_adjust
        self.rotate = rotate
        self.add_noise = add_noise
        self.zoom = zoom
        self.bbox = bbox  # Optional: bounding box for zoom handling

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
                frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))

            if self.add_noise:
                noise_std = 0.01 * np.max(frame)  # 1% of the max pixel value
                noise = np.random.normal(0, noise_std, frame.shape)
                frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

            if self.zoom:
                zoom_factor = np.random.uniform(1.0, 1.3)
                h, w, _ = frame.shape
                new_h = int(h / zoom_factor)
                new_w = int(w / zoom_factor)
                new_h = min(new_h, h)  # Ensure dimensions are within the frame
                new_w = min(new_w, w)
                
                if self.bbox is not None:
                    center_y = (self.bbox[1] + self.bbox[3]) // 2
                    center_x = (self.bbox[0] + self.bbox[2]) // 2
                    top = max(center_y - new_h // 2, 0)
                    left = max(center_x - new_w // 2, 0)
                else:
                    top = (h - new_h) // 2
                    left = (w - new_w) // 2
                
                frame = frame[top:top + new_h, left:left + new_w]
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            if self.brightness_adjust:
                factor = np.random.uniform(0.7, 1.3)
                frame = np.clip(frame * factor, 0, 255).astype(np.uint8)

            augmented_frames.append(frame)

        return np.array(augmented_frames)