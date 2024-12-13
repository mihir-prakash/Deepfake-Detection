import numpy as np
from sklearn.neighbors import NearestNeighbors

class LSH:
    def __init__(self, n_neighbors=5):
        self.lsh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')

    def fit(self, embeddings):
        self.lsh.fit(embeddings)

    def query(self, embedding):
        distances, indices = self.lsh.kneighbors([embedding])
        return distances, indices

def preload_lsh(lsh, embeddings):
    """
    Preload LSH with embeddings generated from the Wild Deepfake dataset.
    """
    lsh.fit(embeddings)
