import os
import numpy as np

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_preprocessed_data(data, output_dir, prefix):
    create_dir_if_not_exists(output_dir)
    
    for i, frames in enumerate(data):
        file_path = os.path.join(output_dir, f"{prefix}_preprocessed_{i}.npy")
        np.save(file_path, frames)
        print(f"Saved preprocessed data to {file_path}")

def load_preprocessed_data(file_path):
    return np.load(file_path)