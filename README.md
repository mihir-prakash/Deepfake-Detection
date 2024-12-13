# Enhanced Deepfake Detection System

## Overview
This project implements an efficient deepfake detection system leveraging Bloom Filters and Locality-Sensitive Hashing (LSH) for scalable and fast identification of deepfake content. The system is optimized for both known and modified deepfakes, addressing scalability and accuracy challenges faced by traditional methods.

## Key Features
- **Bloom Filters:** Efficient detection of known deepfakes by maintaining a compact database of hashes.
- **Locality-Sensitive Hashing (LSH):** Robust similarity detection for modified or derivative deepfakes.
- **Preprocessing:** Extraction of frames from videos for analysis.
- **Feature Extraction:** Use of a pre-trained ResNet-18 model for generating high-dimensional embeddings.
- **Performance Evaluation:** Metrics include precision, recall, F1-score, false positive rate, and query time.

## Project Structure
```plaintext
DEEPFAKE-DETECTION
├── data
│   ├── celeb_df
│   └── frames
├── models
│   └── cnn_model.pth
├── plots
│   ├── f1_scores.png
│   ├── false_positive_rates.png
│   ├── fpr_vs_filter_size.png
│   ├── query_times_known.png
│   └── query_times_similarity.png
├── src
│   ├── bloom_filter.py
│   ├── dataset_loader.py
│   ├── feature_extraction.py
│   ├── lsh.py
│   ├── main.py
│   ├── preprocess.py
│   └── utils.py
├── generate_plots.py
├── README.md
├── requirements.txt
└── venv
```

## Dependencies
Ensure you have the required dependencies installed by running:
```bash
pip install -r requirements.txt
```

## How to Run
### 1. Preprocess the Data
Extract frames from the input video dataset:
```bash
python src/preprocess.py
```

### 2. Generate Features
Extract embeddings from video frames:
```bash
python src/feature_extraction.py
```

### 3. Run the Main Script
Detect deepfakes using Bloom Filters and LSH:
```bash
python src/main.py
```

### 4. Generate Plots
Visualize performance metrics:
```bash
python generate_plots.py
```

## Code Details
### `main.py`
The entry point for the project:
- Integrates Bloom Filters and LSH.
- Configures datasets and evaluates performance metrics.

### `preprocess.py`
Handles video preprocessing:
- Extracts frames from video files.
- Saves frames for feature extraction.

### `feature_extraction.py`
Responsible for feature extraction:
- Uses a pre-trained ResNet-18 model.
- Generates embeddings for each frame.

### `bloom_filter.py`
Implements Bloom Filter functionality:
- Configurable hash functions and sizes.
- Optimized for reducing false positives.

### `generate_plots.py`
Creates visualizations for evaluation:
- Plots metrics such as F1-score, false positive rate, and query times.

## Datasets
1. **Celeb-DF v2**: Used for detecting known deepfakes.
2. **Wild Deepfake**: Sourced from Hugging Face for similarity detection.



## Plots
The following plots demonstrate the system's performance:
- `f1_scores.png`: F1-scores for various configurations.
- `false_positive_rates.png`: False positive rates for Bloom Filters.
- `fpr_vs_filter_size.png`: Effect of filter size on false positive rates.
- `query_times_known.png`: Query times for detecting known deepfakes.
- `query_times_similarity.png`: Query times for similarity detection.

## Contributing
Feel free to open issues or submit pull requests to enhance the system.




