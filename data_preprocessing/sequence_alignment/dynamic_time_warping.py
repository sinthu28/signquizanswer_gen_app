from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def align_sequences(sequence1, sequence2):
    """
    Apply Dynamic Time Warping to align two gesture sequences.

    Args:
        sequence1 (numpy array): First gesture sequence (optical flow or frame sequence).
        sequence2 (numpy array): Second gesture sequence (optical flow or frame sequence).

    Returns:
        distance (float): Distance between the sequences.
        path (list): Optimal alignment path.
    """
    distance, path = fastdtw(sequence1, sequence2, dist=euclidean)
    return distance, path