import os

import numpy as np

def create_dir_if_not_exists(dir_path):
    """
    Creates the directory if it does not already exist.
    
    Args:
        dir_path (str): Path to the directory to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_preprocessed_data(data, output_dir, prefix):
    """
    Save the preprocessed data as numpy files for later use.
    
    Args:
        data (list): Preprocessed frame data.
        output_dir (str): Directory to save the files.
        prefix (str): Prefix for the saved filenames.
    """
    create_dir_if_not_exists(output_dir)
    
    for i, frames in enumerate(data):
        file_path = os.path.join(output_dir, f"{prefix}_preprocessed_{i}.npy")
        np.save(file_path, frames)
        print(f"Saved preprocessed data to {file_path}")

def load_preprocessed_data(file_path):
    """
    Load the preprocessed data from a numpy file.
    
    Args:
        file_path (str): Path to the numpy file.
    
    Returns:
        data (numpy array): Loaded preprocessed data.
    """
    return np.load(file_path)