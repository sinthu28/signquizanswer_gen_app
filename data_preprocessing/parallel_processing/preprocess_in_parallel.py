import concurrent.futures
import os

def preprocess_in_parallel(video_paths, load_video_frames, normalize_frames):
    """
    Preprocess multiple video files in parallel using concurrent processing.

    Args:
        video_paths (list): List of paths to video files.
        load_video_frames (function): Function to load video frames.
        normalize_frames (function): Function to normalize frames.

    Returns:
        all_preprocessed_data (list): List of preprocessed video frames.
    """
    all_preprocessed_data = []

    def process_video(video_path):
        frames = load_video_frames(video_path)
        frames = normalize_frames(frames)
        return frames

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_video, video_path) for video_path in video_paths]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_preprocessed_data.append(result)
            except Exception as e:
                print(f"Error processing video: {e}")
    
    return all_preprocessed_data