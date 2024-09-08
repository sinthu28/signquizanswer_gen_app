import os
from sklearn.model_selection import train_test_split
from data_preprocessing.parallel_processing.preprocess_in_parallel import preprocess_in_parallel
from data_preprocessing.augmentation.advanced_augment_data import advanced_augment_data
from data_preprocessing.feature_extraction.optical_flow import calculate_optical_flow

def full_preprocessing_pipeline(video_dir, load_video_frames, normalize_frames, test_size=0.2, apply_optical_flow=False):
    """
    Full preprocessing pipeline that includes loading, normalizing, augmenting, and optional optical flow computation.
    
    Args:
        video_dir (str): Directory containing video files.
        load_video_frames (function): Function to load video frames.
        normalize_frames (function): Function to normalize video frames.
        test_size (float): Fraction of data to be used for testing (default 0.2).
        apply_optical_flow (bool): Whether to compute optical flow between frames (default False).
    
    Returns:
        train_data (list): Preprocessed training data.
        test_data (list): Preprocessed test data.
    """
    
    # Step 1: Collect all video paths from the given directory
    video_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith('.mp4')]

    # Step 2: Preprocess videos in parallel (loading and normalization)
    preprocessed_data = preprocess_in_parallel(video_paths, load_video_frames, normalize_frames)
    
    # Step 3: Split the data into train and test sets
    train_data, test_data = train_test_split(preprocessed_data, test_size=test_size)
    
    # Step 4: Data augmentation
    augmented_train_data = []
    for frames in train_data:
        augmented_train_data.append(advanced_augment_data(frames, rotate=True, add_noise=True, zoom=True))
    
    # Step 5: Optional - Compute optical flow for motion-based gesture recognition
    if apply_optical_flow:
        optical_train_data = [calculate_optical_flow(frames) for frames in train_data]
        optical_test_data = [calculate_optical_flow(frames) for frames in test_data]
        return optical_train_data, optical_test_data

    return augmented_train_data, test_data