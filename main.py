import os
from data_preprocessing.pipeline.full_preprocessing_pipeline import full_preprocessing_pipeline
from data_preprocessing.load.load_video_frames import load_video_frames
from data_preprocessing.normalization.normalize_frames import normalize_frames
from data_preprocessing.utils.helper_functions import save_preprocessed_data

def main():
    # Specify the directory containing video data
    video_dir = './videos/'
    
    # Check if the directory exists
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"The directory {video_dir} does not exist.")
    
    # Specify the output directory for saving preprocessed data
    output_dir = './preprocessed_data/'
    
    # Run the full preprocessing pipeline
    train_data, test_data = full_preprocessing_pipeline(
        video_dir=video_dir,
        load_video_frames=load_video_frames,
        normalize_frames=normalize_frames,
        test_size=0.2,
        apply_optical_flow=True  # Set this to False if you don't need optical flow
    )
    
    # Save the preprocessed training and testing data
    save_preprocessed_data(train_data, output_dir, prefix='train')
    save_preprocessed_data(test_data, output_dir, prefix='test')
    
    print("Preprocessing complete. Data saved to", output_dir)

if __name__ == "__main__":
    main()