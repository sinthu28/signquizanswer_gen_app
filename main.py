import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import euclidean_distances
from pickletools import optimize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from Loader.WLASLDatasetLoader import WLASLDatasetLoader
from Loader.DataSplitter import DataSplitter
from Preprocess.Normaliser import FrameNormaliser
from Preprocess.Augmentation import DataAugmentation
from Preprocess.OpticalFlow import OpticalFlowCalculator
from Preprocess.SequenceAligner import HierarchicalMethod
from Preprocess.ParallelProcess import VideoPreprocessor
from Models.CNN import CNNLSTMModel

json_path = '/Users/dxt/Desktop/beta_/data/WLASL_v0.3.json'
missing_file_path = '/Users/dxt/Desktop/beta_/data/missing.txt'
video_dir = '/Users/dxt/Desktop/beta_/data/videos'
batch_size = 4
num_epochs = 10
learning_rate = 0.001

dataset_loader = WLASLDatasetLoader(json_path, missing_file_path, video_dir)
dataset = dataset_loader.load_dataset()

normaliser = FrameNormaliser()
augmenter = DataAugmentation()
optical_flow_calculator = OpticalFlowCalculator()
sequence_aligner = HierarchicalMethod(distance_metric=euclidean_distances)
video_preprocessor = VideoPreprocessor(
    normaliser=normaliser,
    augmenter=augmenter,
    optical_flow_calculator=optical_flow_calculator,
    sequence_aligner=sequence_aligner
)

preprocessed_data = video_preprocessor.preprocess_in_parallel(
    video_paths=[data['frames'] for data in dataset],
    load_video_frames=lambda x: x
)

frames = [data['frames'] for data in preprocessed_data]
labels = [data['gloss'] for data in preprocessed_data]
X = np.array(frames)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = CNNLSTMModel(num_classes=len(set(y)))
criterion = nn.CrossEntropyLoss()
optimizer = optimize.Adam(model.parameters(), lr=learning_rate)

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