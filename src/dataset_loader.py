from datasets import load_dataset

def load_wild_deepfake():
    """
    Load the Wild Deepfake dataset from Hugging Face.
    Returns:
        Dataset splits (train, test)
    """
    dataset = load_dataset("xingjunm/WildDeepfake")
    train_data = dataset['train']
    test_data = dataset['test']
    return train_data, test_data
