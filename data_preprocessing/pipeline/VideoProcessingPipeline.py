import os

class VideoProcessingPipeline:
    def __init__(self, load_video_frames, normalize_frames, augmentation, optical_flow, data_splitter, max_workers=4, use_gpu=True, augment_test=False):
        self.load_video_frames = load_video_frames
        self.normalize_frames = normalize_frames
        self.augmentation = augmentation
        self.optical_flow = optical_flow
        self.data_splitter = data_splitter
        self.max_workers = max_workers
        self.use_gpu = use_gpu
        self.augment_test = augment_test

    def full_pipeline(self, video_dir, apply_optical_flow=False):
        video_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith('.mp4')]

        preprocessed_data = self.process_videos_parallel(video_paths)

        train_data, test_data = self.data_splitter.split(preprocessed_data)

        augmented_train_data = [self.augmentation.augment(frames) for frames in train_data]

        if self.augment_test:
            test_data = [self.augmentation.augment(frames) for frames in test_data]

        if apply_optical_flow:
            optical_train_data = [self.optical_flow.calculate(frames) for frames in augmented_train_data]
            optical_test_data = [self.optical_flow.calculate(frames) for frames in test_data]
            return optical_train_data, optical_test_data

        return augmented_train_data, test_data