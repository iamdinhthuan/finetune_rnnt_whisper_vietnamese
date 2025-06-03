# RNN Transducer (RNN-T) Training

This repository contains code for training an RNN Transducer model for speech recognition using PyTorch Lightning.

## Requirements

```
requirements.txt
pytorch
pytorch-lightning
torch.utils.data
sentencepiece
```

## Project Structure

```
.
├── models/
│   ├── encoder.py     # Audio encoder model
│   ├── decoder.py     # Text decoder model
│   └── jointer.py     # Joint network
├── utils/
│   ├── dataset.py     # Dataset loading and preprocessing
│   └── scheduler.py   # Learning rate scheduler
├── constants.py       # Configuration constants
├── train.py          # Training loop implementation
└── run.py            # Main training script
```

## Configuration

Key parameters in constants.py:

```python
TRAIN_MANIFEST        # Path to training manifest file
VAL_MANIFEST         # Path to validation manifest file
BG_NOISE_PATH        # Path to background noise files for augmentation
TOKENIZER_MODEL_PATH # Path to SentencePiece tokenizer model
BATCH_SIZE          # Training batch size
NUM_WORKERS         # Number of data loading workers
MAX_EPOCHS          # Maximum training epochs
ATTENTION_CONTEXT_SIZE # Size of attention context window
VOCAB_SIZE          # Vocabulary size
LOG_DIR             # Directory for saving checkpoints and logs
```

## Usage

1. Prepare your data and update paths in constants.py

2. Run training:

```bash
python run.py
```

The script will:

-   Load training and validation datasets
-   Create data loaders with specified batch size and workers
-   Initialize the RNN-T model
-   Configure model checkpointing
-   Train using PyTorch Lightning

## Training Features

-   Mixed precision training (BF16)
-   GPU acceleration when available
-   Automatic model checkpointing
-   TensorBoard logging
-   Learning rate warmup
-   Data augmentation with background noise
-   Supports resuming from checkpoints

## Resume Training

To resume training from a checkpoint:

```python
trainer.fit(model, train_dataloader, val_dataloader,
           ckpt_path="path/to/checkpoint.ckpt")
```

## Monitoring

Training metrics are logged to TensorBoard. View them with:

```bash
tensorboard --logdir=LOG_DIR
```

## Model Outputs

Checkpoints are saved to `LOG_DIR` with naming format:

```
rnnt-{epoch:02d}-{val_loss:.2f}.ckpt
```

The best models are kept based on validation loss.

# Complete Implementation with Accelerate for Kaggle Multi-GPU

I'll create three complete files that work together:

1. **models/streaming_rnnt.py** - The model definition
2. **train_accelerate.py** - Main training script with Accelerate
3. **run_accelerate.py** - Command-line runner script

Let's start with the model:

Now, let's create the training script with Accelerate:

Finally, let's create the run script to simplify launching the training:

## How to Use This Implementation

I've created three complete files that replace PyTorch Lightning with Accelerate for better Kaggle multi-GPU support:

### 1. models/streaming_rnnt.py

This file contains the model definition, converted from PyTorch Lightning to a standard nn.Module.

### 2. train.py

The main training script using Accelerate. It handles:

-   Dataloaders and datasets
-   Model creation and training
-   Checkpointing and logging
-   Validation

### 3. run.py

A command-line runner script that:

-   Parses arguments
-   Sets up directories
-   Launches training with Accelerate

### Getting Started

1. **Install Accelerate**:

```bash
pip install accelerate
```

2. **Run Basic Training**:

```bash
python run.py \
  --train_manifest=/path/to/train.json \
  --val_manifest=/path/to/val.json \
  --tokenizer_model_path=/path/to/tokenizer.model \
  --bg_noise_path=/path/to/noise \
  --batch_size=32 \
  --output_dir=./my_experiment
```

3. **Resume Training**:

```bash
python run.py \
  --train_manifest=/path/to/train.json \
  --val_manifest=/path/to/val.json \
  --tokenizer_model_path=/path/to/tokenizer.model \
  --resume_from_checkpoint=./my_experiment/rnnt-latest.pt
```

4. **Advanced Configuration**:

```bash
python run.py \
  --train_manifest=/path/to/train.json \
  --val_manifest=/path/to/val.json \
  --tokenizer_model_path=/path/to/tokenizer.model \
  --batch_size=64 \
  --lr=3e-4 \
  --max_epochs=100 \
  --precision=bf16-mixed \
  --num_workers=16 \
  --experiment_name=large_batch
```

## License

[Your license information here]
