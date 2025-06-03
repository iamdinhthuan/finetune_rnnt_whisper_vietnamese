# constran.py
# Audio parameters (FIXED)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 16
MAX_EPOCHS = 50
ATTENTION_CONTEXT_SIZE = (40, 2)

# Path
PRETRAINED_ENCODER_WEIGHT = './weights/phowhisper_small_encoder.pt'
BG_NOISE_PATH = ["./datatest/noise/fsdnoisy18k"]
TRAIN_MANIFEST = ["./datatest/audio_logs_vie.jsonl"]
VAL_MANIFEST = ["./datatest/audio_logs_vie.jsonl"]
LOG_DIR = './checkpoints'

# Optimizer and scheduler parameters
TOTAL_STEPS = 3000000
WARMUP_STEPS = 2000
LR = 1e-4
MIN_LR = 1e-5

# Tokenizer parameters
VOCAB_SIZE = 1024
TOKENIZER_MODEL_PATH = './weights/tokenizer_spe_bpe_v1024_pad/tokenizer.model'
RNNT_BLANK = 1024
PAD = 1 # tokenizer.pad_id()

# Greedy decoding paramesters
MAX_SYMBOLS = 5

# Whisper-small parameters
N_STATE = 768
N_HEAD = 12
N_LAYER = 12


# ==============================================================================
#                        temp constants accelerate
# ==============================================================================

# # Audio parameters (fixed)
# SAMPLE_RATE = 16000
# N_FFT = 400
# HOP_LENGTH = 160
# N_MELS = 80

# # Training parameters (configurable)
# BATCH_SIZE = 32
# NUM_WORKERS = 2
# MAX_EPOCHS = 50
# ATTENTION_CONTEXT_SIZE = 80  # Default value, can be overridden

# # File paths
# PRETRAINED_ENCODER_WEIGHT = './weights/small_encoder.pt'
# BG_NOISE_PATH = None
# TRAIN_MANIFEST = "/kaggle/working/data/train/train_data.jsonl"
# VAL_MANIFEST = "/kaggle/working/data/test/test_data.jsonl"
# LOG_DIR = './checkpoints'

# # Optimizer and scheduler parameters
# TOTAL_STEPS = 3000000
# WARMUP_STEPS = 2000
# LR = 1e-4
# MIN_LR = 1e-5

# # Tokenizer parameters
# VOCAB_SIZE = 1024
# TOKENIZER_MODEL_PATH = './weights/tokenizer_spe_bpe_v1024_pad/tokenizer.model'
# RNNT_BLANK = 1024
# PAD = 1  # tokenizer.pad_id()

# # Greedy decoding parameters
# MAX_SYMBOLS = 3

# # Whisper-small parameters
# N_STATE = 768
# N_HEAD = 12
# N_LAYER = 12