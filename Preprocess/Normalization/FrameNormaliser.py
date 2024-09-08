import numpy as np

class FrameNormaliser:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def normalize(self, frames):
        if not isinstance(frames, np.ndarray):
            raise ValueError("Input frames must be a NumPy array.")
        
        if frames.size == 0:
            raise ValueError("Input frames array is empty.")

        normalized_frames = frames.astype(self.dtype) / 255.0

        return normalized_frames


    """
        # Create an instance of FrameNormalizer
        normalizer = FrameNormalizer(dtype=np.float32)

        # Normalize frames
        frames = np.random.randint(0, 256, (10, 224, 224, 3), dtype=np.uint8)  # Example frame data
        normalized_frames = normalizer.normalize(frames)
    """
