import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class SequenceAligner:
    def __init__(self, distance_metric=euclidean, segment_size=50):
        self.distance_metric = distance_metric
        self.segment_size = segment_size

    def align(self, sequence1, sequence2):
        if not (isinstance(sequence1, (list, np.ndarray)) and isinstance(sequence2, (list, np.ndarray))):
            raise ValueError("Both sequences must be lists or NumPy arrays.")
        
        if np.ndim(sequence1) != np.ndim(sequence2):
            raise ValueError("Both sequences must have the same number of dimensions.")

        if len(sequence1) == 0 or len(sequence2) == 0:
            raise ValueError("Sequences must not be empty.")

        # Hierarchical alignment
        aligned_sequence1 = []
        aligned_sequence2 = []

        segments1 = [sequence1[i:i + self.segment_size] for i in range(0, len(sequence1), self.segment_size)]
        segments2 = [sequence2[i:i + self.segment_size] for i in range(0, len(sequence2), self.segment_size)]

        for seg1, seg2 in zip(segments1, segments2):
            try:
                distance, path = fastdtw(seg1, seg2, dist=self.distance_metric)
                aligned_chunk1 = [seg1[i] for i, j in path]
                aligned_sequence1.extend(aligned_chunk1)
                aligned_sequence2.extend(seg2)
            except Exception as e:
                raise RuntimeError(f"Error aligning segments: {e}")

        return distance, aligned_sequence1, aligned_sequence2