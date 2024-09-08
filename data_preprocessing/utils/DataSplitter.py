from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, test_size=0.2, shuffle=True):
        self.test_size = test_size
        self.shuffle = shuffle

    def split(self, data):
        try:
            train_data, test_data = train_test_split(data, test_size=self.test_size, shuffle=self.shuffle)
            return train_data, test_data
        except Exception as e:
            print(f"Error during train-test split: {e}")
            return None, None