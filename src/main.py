from bloom_filter import DeepfakeBloomFilter, preload_bloom_filter
from feature_extraction import FeatureExtractor
from lsh import LSH
from preprocess import preprocess_huggingface_dataset
from dataset_loader import load_wild_deepfake
import os

# Initialize components
bloom_filter = DeepfakeBloomFilter()
feature_extractor = FeatureExtractor()
lsh = LSH()

def main():
    # Load Wild Deepfake Dataset
    print("Loading Wild Deepfake dataset...")
    train_data, test_data = load_wild_deepfake()

    # Preprocess Videos into Frames
    print("Extracting frames from Wild Deepfake dataset...")
    preprocess_huggingface_dataset(train_data, "data/frames/train/")
    preprocess_huggingface_dataset(test_data, "data/frames/test/")

    # Generate Embeddings and Train LSH
    print("Generating embeddings for LSH...")
    embeddings = []
    for frame_file in os.listdir("data/frames/train/"):
        frame_path = os.path.join("data/frames/train/", frame_file)
        embedding = feature_extractor.extract_features(frame_path)
        embeddings.append(embedding)

    print("Training LSH...")
    lsh.fit(embeddings)

   
    print("Testing on test dataset...")
    for frame_file in os.listdir("data/frames/test/"):
        frame_path = os.path.join("data/frames/test/", frame_file)
        embedding = feature_extractor.extract_features(frame_path)
        distances, indices = lsh.query(embedding)
        if distances[0][0] < 0.1: 
            print(f"Deepfake detected in {frame_file}")

if __name__ == "__main__":
    main()
