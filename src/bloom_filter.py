from pybloom_live import BloomFilter
import os

class DeepfakeBloomFilter:
    def __init__(self, capacity=100000, error_rate=0.001):
        self.bloom_filter = BloomFilter(capacity, error_rate)

    def add_hash(self, video_hash):
        self.bloom_filter.add(video_hash)

    def check_hash(self, video_hash):
        return video_hash in self.bloom_filter

def preload_bloom_filter(bloom_filter, dataset_path):
    """
    Preload Bloom Filter with video hashes from the Celeb-DF dataset.
    """
    for subfolder in ["Celeb-real", "Celeb-synthesis", "YouTube-real"]:
        folder_path = os.path.join(dataset_path, subfolder)
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)
            video_hash = hash(open(video_path, 'rb').read())
            bloom_filter.add_hash(video_hash)
