import torch

class ModelConfig:
    # Data
    CSV_PATH = "data/processed/processed_papers.csv"
    
    # Model
    MODEL_NAME = "allenai/scibert_scivocab_uncased"
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    
    # Training
    BATCH_SIZE = 4
    MAX_INPUT_LEN = 4096  # Full paper
    MAX_TARGET_LEN = 512  # Abstract
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
config = ModelConfig()