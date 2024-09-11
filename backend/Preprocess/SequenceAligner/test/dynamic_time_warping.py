import fastdtw
from scipy.spatial.distance import euclidean

def align_sequences(sequence1, sequence2):
    distance, path = fastdtw(sequence1, sequence2, dist=euclidean)
    return distance, path

"""
    # Example sequences
    sequence1 = [1, 2, 3, 4, 5]
    sequence2 = [2, 3, 4, 5, 6]

    # Align sequences and get the distance and path
    distance, path = align_sequences(sequence1, sequence2)

    print(f"Distance: {distance}")
    print(f"Path: {path}")
"""