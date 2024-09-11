import concurrent.futures
import logging
import numpy as np
import torch
import os
from datetime import datetime
from typing import List, Callable

class VideoPreprocessor:
    def __init__(self, normalizer, augmenter, optical_flow_calculator, sequence_aligner, use_gpu=True, max_workers=2, log_dir='logs'):
        self.normalizer = normalizer
        self.augmenter = augmenter
        self.optical_flow_calculator = optical_flow_calculator
        self.sequence_aligner = sequence_aligner
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_workers = max_workers
        self.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.setup_logging()

    def setup_logging(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"VideoPreprocessor_{current_date}.log"
        log_path = os.path.join(self.log_dir, log_filename)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def preprocess_video(self, frames):
        try:
            self.logger.info(f"Processing {len(frames)} frames.")
            
            self.logger.info("Starting normalization")
            normalized_frames = self.normalizer.normalize(frames)

            self.logger.info("Starting augmentation")
            augmented_frames = self.augmenter.augment(normalized_frames)

            self.logger.info("Starting optical flow calculation")
            optical_flows = self.optical_flow_calculator.calculate(augmented_frames)

            self.logger.info("Starting sequence alignment")
            if len(optical_flows) > 0 and isinstance(optical_flows, (list, np.ndarray)):
                distance, aligned_sequence1, aligned_sequence2 = self.sequence_aligner.align(optical_flows, optical_flows)
            else:
                self.logger.warning("Optical flows are empty or invalid. Skipping alignment.")
                aligned_sequence1, aligned_sequence2 = [], []

            return distance, aligned_sequence1, aligned_sequence2
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return None, None, None

    def preprocess_in_parallel(self, video_paths: List[str], load_video_frames: Callable[[str], np.ndarray]):
        all_preprocessed_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.preprocess_video, load_video_frames(path)) for path in video_paths]
            for future in concurrent.futures.as_completed(futures):
                try:
                    distance, aligned_sequence1, aligned_sequence2 = future.result()
                    if aligned_sequence1 is not None and aligned_sequence2 is not None:
                        all_preprocessed_data.append({
                            'distance': distance,
                            'aligned_sequence1': aligned_sequence1,
                            'aligned_sequence2': aligned_sequence2
                        })
                except Exception as e:
                    self.logger.error(f"Exception during video processing: {e}")

        self.logger.info(f"Successfully processed {len(all_preprocessed_data)} out of {len(video_paths)} videos.")
        return all_preprocessed_data