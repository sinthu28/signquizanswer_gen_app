import os
from Loader.WLASLDatasetLoader import WLASLDatasetLoader
from Loader.DataSplitter import DataSplitter
from Preprocess.Normaliser import FrameNormaliser
from Preprocess.Augmentation import DataAugmentation
from Preprocess.OpticalFlow import OpticalFlowCalculator
from Preprocess.SequenceAligner import HierarchicalMethod
from Preprocess.ParallelProcess import VideoPreprocessor

def main():
    json_path = '/Users/dxt/Desktop/beta_/data/WLASL_v0.3.json'
    missing_file_path = '/Users/dxt/Desktop/beta_/data/missing.txt'
    video_dir = '/Users/dxt/Desktop/beta_/data/videos'
    
    dataset_loader = WLASLDatasetLoader(json_path, missing_file_path, video_dir)

    dataset = dataset_loader.load_dataset()
    print("Dataset loaded : WLASLDatasetLoader Task done !!")
    # stats = dataset_loader.get_statistics()
    # print(f"Dataset Statistics: {stats}")

    normalizer = FrameNormaliser()  # Specify normalization method if needed
    augmenter = DataAugmentation()  # Configure augmentations if needed
    optical_flow_calculator = OpticalFlowCalculator()  # Configure as needed
    sequence_aligner = HierarchicalMethod()  # Configure alignment if needed
    video_preprocessor = VideoPreprocessor(
        normalizer=normalizer,
        augmenter=augmenter,
        optical_flow_calculator=optical_flow_calculator,
        sequence_aligner=sequence_aligner,
        use_gpu=True,
        max_workers=4,
        log_dir='logs'
    )

    print("Preprocessing parallel started.....")

    processed_data = video_preprocessor.preprocess_in_parallel(
        video_paths=[os.path.join(video_dir, f"{instance['video_id']}.mp4") for instance in dataset],
        load_video_frames=dataset_loader._load_frames_for_instance  # Pass the frame loading function
    )

    print("Preprocessing completed : VideoPreprocessor Task done !!")
    
    print("Data Splitting..")
    data_splitter = DataSplitter()
    training_data, testing_data = data_splitter.split(processed_data)

    print(f"Processed Data Split: {len(training_data)} training samples, {len(testing_data)} testing samples.")

if __name__ == "__main__":
    main()


## <---------------------- 1 ----------------------> ##

# from Loader.WLASLDatasetLoader import WLASLDatasetLoader

# def main():
    # json_path = '/Users/dxt/Desktop/beta_/data/WLASL_v0.3.json'
    # missing_file_path = '/Users/dxt/Desktop/beta_/data/missing.txt'
    # video_dir = '/Users/dxt/Desktop/beta_/data/videos'

#     dataset_loader = WLASLDatasetLoader(
#         json_path=json_path,
#         missing_file_path=missing_file_path,
#         video_dir=video_dir,
#         max_workers=4  
#     )

#     dataset = dataset_loader.load_dataset()
#     print(f"Loaded {len(dataset)} video entries.")

#     stats = dataset_loader.get_statistics()
#     print(f"Statistics: {stats}")

# if __name__ == "__main__":
#     main()