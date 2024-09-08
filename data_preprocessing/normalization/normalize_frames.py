def normalize_frames(frames):
    """
    Normalize frame pixel values to range [0, 1].

    Args:
        frames (numpy array): Array of frames.

    Returns:
        Normalized frames.
    """
    return frames / 255.0