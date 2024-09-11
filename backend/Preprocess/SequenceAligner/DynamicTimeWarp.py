import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class SequenceAligner:
    def __init__(self, distance_metric=euclidean, radius=10):
        self.distance_metric = distance_metric
        self.radius = radius

    def _validate_sequences(self, sequence1, sequence2):
        if not isinstance(sequence1, (list, np.ndarray)):
            raise ValueError(f"Invalid sequence1 type: {type(sequence1)}. Expected list or np.ndarray.")
        if not isinstance(sequence2, (list, np.ndarray)):
            raise ValueError(f"Invalid sequence2 type: {type(sequence2)}. Expected list or np.ndarray.")
        
        if np.ndim(sequence1) != np.ndim(sequence2):
            raise ValueError(f"Sequences must have the same number of dimensions. Got {np.ndim(sequence1)} and {np.ndim(sequence2)}.")
        
        if len(sequence1) == 0 or len(sequence2) == 0:
            raise ValueError("Sequences must not be empty.")
        
        # Convert to numpy arrays for consistency
        sequence1 = np.asarray(sequence1)
        sequence2 = np.asarray(sequence2)
        
        return sequence1, sequence2

    def align(self, sequence1, sequence2):
        try:
            # Validate the input sequences
            sequence1, sequence2 = self._validate_sequences(sequence1, sequence2)
            
            # Perform FastDTW
            distance, path = fastdtw(sequence1, sequence2, dist=self.distance_metric, radius=self.radius)

            aligned_sequence1 = []
            aligned_sequence2 = []

            for (i, j) in path:
                # Check bounds to ensure no index errors
                if i < len(sequence1) and j < len(sequence2):
                    aligned_sequence1.append(sequence1[i])
                    aligned_sequence2.append(sequence2[j])

            return distance, aligned_sequence1, aligned_sequence2

        except ValueError as ve:
            raise ValueError(f"Validation error: {ve}")
        except Exception as e:
            raise RuntimeError(f"Error aligning sequences: {e}")

