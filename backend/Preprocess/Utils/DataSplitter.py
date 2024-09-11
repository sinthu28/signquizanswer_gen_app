from sklearn.model_selection import train_test_split
import numpy as np

class DataSplitter:
    def __init__(self, test_size=0.2, shuffle=True):
        self.test_size = test_size
        self.shuffle = shuffle

    def split(self, data, labels=None):
        if not isinstance(data, (np.ndarray, list)):
            raise ValueError("Data must be a NumPy array or a list.")
        if labels is not None and not isinstance(labels, (np.ndarray, list)):
            raise ValueError("Labels must be a NumPy array or a list.")

        if labels is not None:
            if len(data) != len(labels):
                raise ValueError("Data and labels must have the same length.")
            try:
                train_data, test_data, train_labels, test_labels = train_test_split(
                    data, labels, test_size=self.test_size, shuffle=self.shuffle
                )
                return train_data, test_data, train_labels, test_labels
            except Exception as e:
                print(f"Error during train-test split: {e}")
                return None, None, None, None
        else:
            try:
                train_data, test_data = train_test_split(data, test_size=self.test_size, shuffle=self.shuffle)
                return train_data, test_data
            except Exception as e:
                print(f"Error during train-test split: {e}")
                return None, None