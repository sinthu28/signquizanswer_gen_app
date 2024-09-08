from Preprocess.Load.VideoLoader import VideoLoader
from Preprocess.Normalization.FrameNormaliser import FrameNormaliser
from Preprocess.ParallelProcess.VideoPreprocessor import VideoPreprocessor
from Preprocess.Augmentation.DataAugmentation import DataAugmentation
from Preprocess.FeatureExtraction.OpticalFlowCalculator import OpticalFlowCalculator
from Preprocess.SequenceAligner.SequenceAligner import SequenceAligner
from Preprocess.Utils.DataSplitter import DataSplitter
from Models.CNN import CNNLSTMModelBuilder
from Models.Transformers import TransformerBlock
from Gesture.GestureRecogniser import GestureRecogniser
from QuestionUnderstanding.QuestionUnderstanding import QuestionUnderstanding

import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    video_dir = 'path/to/video_directory'
    test_size = 0.2
    use_gpu = True
    max_workers = 4
    use_process_pool = False
    apply_optical_flow = True
    
    video_loader = VideoLoader(frame_size=(224, 224), max_frames=100)
    frame_normalizer = FrameNormaliser(dtype=np.float32)
    gesture_recognizer = GestureRecogniser()
    video_preprocessor = VideoPreprocessor(
        load_video_frames=video_loader.load_video_frames,
        normalize_frames=frame_normalizer.normalize,
        use_gpu=use_gpu,
        max_workers=max_workers,
        use_process_pool=use_process_pool
    )
    data_augmenter = DataAugmentation()
    optical_flow_calculator = OpticalFlowCalculator()
    sequence_aligner = SequenceAligner()
    data_splitter = DataSplitter(test_size=test_size)
    question_understanding = QuestionUnderstanding()  # Initialize question understanding model

    video_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith('.mp4')]
    if not video_paths:
        raise FileNotFoundError("No video files found in the specified directory.")

    logging.info("Starting video preprocessing...")
    preprocessed_data = video_preprocessor.preprocess_in_parallel(video_paths)

    if not preprocessed_data:
        raise ValueError("No data was processed from the videos.")

    logging.info("Extracting gestures and augmenting data...")
    gesture_data = []
    for frames in preprocessed_data:
        gestures = [gesture_recognizer.process_frame(frame) for frame in frames]
        gesture_data.append(np.array(gestures))

    augmented_data = [data_augmenter.augment(frames) for frames in gesture_data]

    if apply_optical_flow:
        logging.info("Calculating optical flow...")
        optical_data = [optical_flow_calculator.calculate_optical_flow(frames) for frames in augmented_data]
        data = optical_data
    else:
        data = augmented_data
    
    logging.info("Splitting data into train and test sets...")
    train_data, test_data = data_splitter.split_data(data)
    if not train_data or not test_data:
        raise ValueError("Training or testing data is empty after splitting.")

    logging.info("Building models...")
    cnn_lstm_model = CNNLSTMModelBuilder(input_shape=(None, 224, 224, 3), num_classes=10).build_model()
    transformer_model = TransformerBlock(input_shape=(None, 224, 224, 3), num_classes=10).build_model()

    cnn_lstm_model.summary()
    transformer_model.summary()

    context = "Transformers are a type of model architecture that has achieved state-of-the-art results on various NLP tasks."
    question = "What are transformers?"
    answer = question_understanding.answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    logging.info("Phase 1 completed successfully.")

    # Here you would add the training code, e.g.:
    # cnn_lstm_model.train(train_data)
    # transformer_model.train(train_data)
    
if __name__ == "__main__":
    main()

###########################################################################################
#             Without Gesture Recogniser class implementation Below                       #                                                    #
###########################################################################################

# import os
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def main():
#     video_dir = 'path/to/video_directory'
#     test_size = 0.2
#     use_gpu = True
#     max_workers = 4
#     use_process_pool = False
#     apply_optical_flow = True

#     try:
#         # Initialize components
#         video_loader = VideoLoader(frame_size=(224, 224), max_frames=100)
#         frame_normalizer = FrameNormaliser(dtype=np.float32)
#         video_preprocessor = VideoPreprocessor(
#             load_video_frames=video_loader.load_video_frames,
#             normalize_frames=frame_normalizer.normalize,
#             use_gpu=use_gpu,
#             max_workers=max_workers,
#             use_process_pool=use_process_pool
#         )
#         data_augmenter = DataAugmentation()
#         optical_flow_calculator = OpticalFlowCalculator()
#         sequence_aligner = SequenceAligner()
#         data_splitter = DataSplitter(test_size=test_size)

#         # Load video paths
#         video_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith('.mp4')]
#         if not video_paths:
#             raise FileNotFoundError("No video files found in the specified directory.")

#         # Preprocess videos
#         logging.info("Starting video preprocessing...")
#         preprocessed_data = video_preprocessor.preprocess_in_parallel(video_paths)

#         if not preprocessed_data:
#             raise ValueError("No data was processed from the videos.")

#         # Augment data
#         logging.info("Starting data augmentation...")
#         augmented_data = [data_augmenter.augment(frames) for frames in preprocessed_data]

#         # Apply optical flow if needed
#         if apply_optical_flow:
#             logging.info("Calculating optical flow...")
#             optical_data = [optical_flow_calculator.calculate_optical_flow(frames) for frames in augmented_data]
#             data = optical_data
#         else:
#             data = augmented_data
        
#         # Split data
#         logging.info("Splitting data into train and test sets...")
#         train_data, test_data = data_splitter.split_data(data)
#         if not train_data or not test_data:
#             raise ValueError("Training or testing data is empty after splitting.")

#         # Initialize and build models
#         logging.info("Building models...")
#         cnn_lstm_model = CNNLSTMModelBuilder(input_shape=(None, 224, 224, 3), num_classes=10).build_model()
#         transformer_model = TransformerBlock(input_shape=(None, 224, 224, 3), num_classes=10).build_model()

#         # Summary of models (for debugging purposes)
#         cnn_lstm_model.summary()
#         transformer_model.summary()

#         logging.info("Phase 1 completed successfully.")

#         # Here you would add the training code, e.g.:
#         # cnn_lstm_model.train(train_data)
#         # transformer_model.train(train_data)
               
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         raise

# if __name__ == "__main__":
#     main()