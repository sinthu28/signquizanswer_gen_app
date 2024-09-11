import numpy as np

def split_data(data, labels, test_size=0.2):
    idx = np.random.permutation(len(data))
    split_idx = int(len(data) * (1 - test_size))
    train_idx, test_idx = idx[:split_idx], idx[split_idx:]
    return (data[train_idx], labels[train_idx]), (data[test_idx], labels[test_idx])