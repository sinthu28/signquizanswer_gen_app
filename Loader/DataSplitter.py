import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime

class DataSplitter:
    def __init__(self, test_size=0.2, shuffle=True, random_state=None, log_dir='logs'):
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.log_dir = log_dir
        self.setup_logging()

    def setup_logging(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"DataSplitter_{current_date}.log"
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

    def split(self, data, labels=None):
        if not isinstance(data, (np.ndarray, list)):
            raise ValueError("Data must be a NumPy array or list.")
        if labels is not None and not isinstance(labels, (np.ndarray, list)):
            raise ValueError("Labels must be a NumPy array or list if provided.")
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        if labels is not None and len(data) != len(labels):
            raise ValueError("Length of data and labels must match.")

        self.logger.info("Starting data split")
        try:
            if labels is None:
                train_data, test_data = train_test_split(data, test_size=self.test_size, shuffle=self.shuffle, random_state=self.random_state)
                self.logger.info(f"Data split into train ({len(train_data)}) and test ({len(test_data)})")
                return train_data, test_data
            else:
                train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=self.test_size, shuffle=self.shuffle, random_state=self.random_state)
                self.logger.info(f"Data split into train ({len(train_data)}) and test ({len(test_data)})")
                self.logger.info(f"Labels split into train ({len(train_labels)}) and test ({len(test_labels)})")
                return train_data, test_data, train_labels, test_labels
        except Exception as e:
            self.logger.error(f"Error during data split: {e}")
            raise