from scipy.spatial.distance import euclidean
import numpy as np
import fastdtw

class SequenceAligner:
    def __init__(self, distance_metric=euclidean):
        self.distance_metric = distance_metric

    def align(self, sequence1, sequence2):
        if not (isinstance(sequence1, (list, np.ndarray)) and isinstance(sequence2, (list, np.ndarray))):
            raise ValueError("Both sequences must be lists or NumPy arrays.")
        
        if np.ndim(sequence1) != np.ndim(sequence2):
            raise ValueError("Both sequences must have the same number of dimensions.")

        if len(sequence1) == 0 or len(sequence2) == 0:
            raise ValueError("Sequences must not be empty.")

        try:
            distance, path = fastdtw(sequence1, sequence2, dist=self.distance_metric)
            
            aligned_sequence1 = [sequence1[i] for i, j in path]
            return distance, aligned_sequence1, sequence2
        except Exception as e:
            raise RuntimeError(f"Error aligning sequences: {e}")

"""
        if __name__ == "__main__":
            sequence1 = [1, 2, 3, 4, 5]
            sequence2 = [2, 3, 4, 5, 6]

            aligner = SequenceAligner()

            distance, path = aligner.align(sequence1, sequence2)

            print(f"Distance: {distance}")
            print(f"Path: {path}")
    
"""