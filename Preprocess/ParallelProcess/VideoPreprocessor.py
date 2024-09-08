import concurrent.futures
import os
import logging
import torch  # Assuming PyTorch for GPU usage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoPreprocessor:
    def __init__(self, load_video_frames, normalize_frames, use_gpu=True, max_workers=None, use_process_pool=False):
        self.load_video_frames = load_video_frames
        self.normalize_frames = normalize_frames
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool

    def is_gpu_available(self):
        return torch.cuda.is_available()

    def preprocess_video(self, video_path):
        try:
            logging.info(f"Processing video: {video_path}")
            
            frames = self.load_video_frames(video_path)
            
            if frames is None:
                raise ValueError(f"Failed to load frames from {video_path}")
            
            if self.use_gpu and self.is_gpu_available():
                logging.info(f"Using GPU for video processing: {video_path}")
                frames = torch.tensor(frames).to('cuda')
            
            frames = self.normalize_frames(frames)
            
            if self.use_gpu and self.is_gpu_available():
                frames = frames.cpu().numpy()
            
            return frames
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")
            return None

    def preprocess_in_parallel(self, video_paths):
        all_preprocessed_data = []

        Executor = concurrent.futures.ProcessPoolExecutor if self.use_process_pool else concurrent.futures.ThreadPoolExecutor

        with Executor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.preprocess_video, video_path) for video_path in video_paths]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        all_preprocessed_data.append(result)
                except Exception as e:
                    logging.error(f"Exception during video processing: {e}")
        
        logging.info(f"Successfully processed {len(all_preprocessed_data)} out of {len(video_paths)} videos.")
        return all_preprocessed_data
    

    """
        # Define your video loading and normalization functions
        def load_video_frames(video_path):
            # Implementation here...
            pass

        def normalize_frames(frames):
            # Implementation here...
            pass

        # Create an instance of VideoPreprocessor
        preprocessor = VideoPreprocessor(
            load_video_frames=load_video_frames,
            normalize_frames=normalize_frames,
            use_gpu=True,
            max_workers=4,
            use_process_pool=False
        )

        # List of video file paths
        video_paths = ['path/to/video1.mp4', 'path/to/video2.mp4']

        # Process videos
        processed_frames = preprocessor.preprocess_in_parallel(video_paths)
    """
