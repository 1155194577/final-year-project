
from dataclasses import dataclass

@dataclass
class TrainConfig: 
    embedding_vector_dim = 1024
    num_of_class = 40 
    n_q = 8 
    criterion = "cross_entropy"
    optimizer = "sgd"
    lr = 0.01 
    epochs = 10
    batch_size = 32
    val_fraction = 0.2
    split = "utterance"  # "speaker" holds out whole speakers
    seed = 42
    data_dir = "data/libritts_preprocessed/dev.clean_36speakers_20utts"
    model_dir = None   # SpeechTokenizer snapshot root; downloads if None
    device = None      # torch device; auto if None
    features_dir = None  # where features.npz is cached; dataset root if None
    regularizer = "l1" ## or l2 or None 
    reg_weight = 1e-4  # multiplier on sum(|w|) for l1, sum(w^2) for l2
