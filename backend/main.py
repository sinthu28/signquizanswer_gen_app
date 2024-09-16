import os
import logging
from Loader.WLASLDatasetLoader import WLASLDatasetLoader
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from Preprocess.Normaliser.FrameNormaliser import FrameNormaliser
from Preprocess.Augmentation.DataAugmentation import DataAugmentation
from Preprocess.OpticalFlow.OpticalFlowCalculator import OpticalFlowCalculator
from Preprocess.SequenceAligner.DynamicTimeWarp import SequenceAligner
from Preprocess.ParallelProcess.VideoPreprocessor import VideoPreprocessor
from Models.CNN.CNNLSTMModel import CNNLSTMModel

dataset_metadata = '/Users/dxt/Desktop/beta_/data/WLASL_v0.3.json'
missing_metadata = '/Users/dxt/Desktop/beta_/data/missing.txt'
videos_dataset = '/Users/dxt/Desktop/beta_/data/videos'
max_workers = 4
batch_size = 4
num_epochs = 2
learning_rate = 0.001
num_classes = 100
log_dir = '/Users/dxt/Desktop/beta_/logs'

if not os.path.exists(log_dir):
            os.makedirs(log_dir)

logger = logging.getLogger(__name__)
log_file = os.path.join(log_dir, "WLASLDatasetLoader.log")
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)    

load_wlasl_vidoes = WLASLDatasetLoader(
    json_path=dataset_metadata,
    missing_file_path=missing_metadata,
    video_dir=videos_dataset,
    max_workers=max_workers,
    batch_size=batch_size,
    log_dir=log_dir
)

dataset = load_wlasl_vidoes.load_dataset(limit=5)

Load_Dataset = WLASLDatasetLoader()


normaliser = FrameNormaliser(method="standard", log_dir=log_dir)
augmenter = DataAugmentation()
optical_flow_calculator = OpticalFlowCalculator()
sequence_aligner = SequenceAligner(distance_metric=np.linalg.norm)

video_preprocessor = VideoPreprocessor(
    normalizer=normaliser,
    augmenter=augmenter,
    optical_flow_calculator=optical_flow_calculator,
    sequence_aligner=sequence_aligner
)

preprocessed_data = video_preprocessor.preprocess_in_parallel(
    video_paths=[data['frames'] for data in dataset],
    load_video_frames=lambda x: x  
)
print(f"Number of preprocessed samples: {len(preprocessed_data)}")

frames = [data['frames'] for data in preprocessed_data]
labels = [data['gloss'] for data in preprocessed_data]
print(f"Frames sample shape: {np.array(frames).shape}")
print(f"Labels sample: {labels[:5]}")

if len(preprocessed_data) == 0:
    raise ValueError("No data was processed. Please check the video preprocessing step.")

X = np.array(frames)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = CNNLSTMModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

model_save_path = 'cnn_lstm_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


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