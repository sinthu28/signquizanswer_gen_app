import os
from Preprocess.Augmentation.DataAugmentation import DataAugmentation

class VideoProcessingPipeline:
    def __init__(self, video_preprocessor, data_splitter, load_video_frames, augment_test=False):
        self.video_preprocessor = video_preprocessor
        self.data_splitter = data_splitter
        self.load_video_frames = load_video_frames
        self.augment_test = augment_test
        

    def full_pipeline(self, video_dir, apply_optical_flow=False):
        video_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith('.mp4')]

        preprocessed_data = self.video_preprocessor.preprocess_in_parallel(video_paths, self.load_video_frames)

        train_data, test_data = self.data_splitter.split(preprocessed_data)

        if self.augment_test:
            self.video_preprocessor.logger.info("Augmenting test data")
            test_data = [self.video_preprocessor.augmenter.augment(frames) for frames in test_data]

        if apply_optical_flow:
            self.video_preprocessor.logger.info("Applying optical flow on the data")
            optical_train_data = [self.video_preprocessor.optical_flow_calculator.calculate(frames) for frames in train_data]
            optical_test_data = [self.video_preprocessor.optical_flow_calculator.calculate(frames) for frames in test_data]
            return optical_train_data, optical_test_data

        return train_data, test_data