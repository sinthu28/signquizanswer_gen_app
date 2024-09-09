import numpy as np
import fastdtw
from scipy.spatial.distance import euclidean


class SequenceAligner:
    def __init__(self, distance_metric=euclidean, chunk_size=100):
        self.distance_metric = distance_metric
        self.chunk_size = chunk_size

    def align(self, sequence1, sequence2):
        if not (isinstance(sequence1, (list, np.ndarray)) and isinstance(sequence2, (list, np.ndarray))):
            raise ValueError("Both sequences must be lists or NumPy arrays.")
        
        if np.ndim(sequence1) != np.ndim(sequence2):
            raise ValueError("Both sequences must have the same number of dimensions.")

        if len(sequence1) == 0 or len(sequence2) == 0:
            raise ValueError("Sequences must not be empty.")

        # Process in chunks
        aligned_sequence1 = []
        aligned_sequence2 = []
        total_distance = 0

        for i in range(0, len(sequence1), self.chunk_size):
            chunk1 = sequence1[i:i + self.chunk_size]
            chunk2 = sequence2[i:i + self.chunk_size]
            
            try:
                distance, path = fastdtw(chunk1, chunk2, dist=self.distance_metric)
                total_distance += distance
                
                # Reconstruct the aligned sequences from the path
                aligned_chunk1 = [chunk1[i] for i, j in path]
                aligned_sequence1.extend(aligned_chunk1)
                aligned_sequence2.extend(chunk2)
            except Exception as e:
                raise RuntimeError(f"Error aligning chunks: {e}")

        return total_distance, aligned_sequence1, aligned_sequence2