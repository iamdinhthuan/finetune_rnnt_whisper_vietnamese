# Whisper RNN-T: Vietnamese Speech Recognition

A streaming Vietnamese speech recognition system combining Whisper encoder with RNN-T decoder for real-time transcription.

## Features

- 🎯 **Step-based Model Saving**: Automatic checkpoint saving every N steps with WER-based selection
- 🔄 **Streaming Recognition**: Real-time audio processing with streaming RNN-T decoder
- 🎛️ **Whisper Base Encoder**: Pre-trained Whisper Base encoder for robust feature extraction
- 🇻🇳 **Vietnamese Optimized**: Trained specifically for Vietnamese speech recognition
- 💾 **Smart Storage**: Saves only model weights, keeps top-K best models automatically
- 📊 **Comprehensive Logging**: Detailed training metrics and model performance tracking

## Architecture

```
Audio Input → Whisper Encoder → RNN-T Decoder → Vietnamese Text
```

- **Encoder**: Whisper Base (6 layers, 512 dim, 8 heads)
- **Decoder**: LSTM-based RNN-T decoder
- **Joint Network**: Projects encoder-decoder outputs to vocabulary space
- **Loss**: RNN-T loss with blank token handling

## Model Saving Strategy

The system implements intelligent step-based model saving:

- ✅ **Save every N steps** (configurable, default: 1000)
- ✅ **Keep top-K models** based on validation WER (default: 3)
- ✅ **Automatic cleanup** of worse-performing models
- ✅ **Weights-only saving** to reduce storage
- ✅ **Fallback mechanism** using train_loss when val_wer unavailable

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd whisper_rnnt

# Install dependencies
pip install torch pytorch-lightning
pip install librosa sentencepiece
pip install warprnnt-numba
pip install loguru tqdm
pip install gradio  # for UI demo
```

## Quick Start

### 1. Download Pre-trained Weights

```bash
cd weights
python download_whisper_base.py
python export_encoder.py
```

### 2. Training

```bash
# Configure your data paths in config.yaml
python train.py
```

### 3. Real-time Demo

```bash
python infer_stream_ui.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Model saving strategy
model_saving:
  save_every_n_steps: 1000  # Save frequency
  keep_top_k: 3            # Number of best models to keep
  monitor_metric: 'val_wer' # Metric for model selection
  save_weights_only: true   # Save only weights
```

## Training Features

- **Step-based validation**: Runs validation every N steps instead of epochs
- **Automatic model selection**: Keeps only the best performing models
- **Comprehensive metrics**: WER, loss tracking with detailed logging
- **Memory efficient**: Saves only model weights, not full checkpoints
- **Robust training**: Handles missing metrics with fallback mechanisms

## File Structure

```
whisper_rnnt/
├── config.yaml              # Configuration file
├── train.py                 # Training script
├── infer_stream_ui.py       # Gradio demo
├── models/                  # Model architectures
│   ├── encoder.py          # Whisper encoder
│   ├── decoder.py          # RNN-T decoder
│   └── jointer.py          # Joint network
├── utils/                   # Utilities
│   ├── model_checkpoint.py # Step-based saving
│   ├── dataset.py          # Data loading
│   └── scheduler.py        # Learning rate scheduling
└── weights/                 # Model weights
    ├── download_whisper_base.py
```

## Model Performance

The step-based saving system ensures you always have access to the best performing models:

- Models are saved with descriptive filenames: `rnnt-step-001000-wer-0.1234.pt`
- Automatic cleanup removes worse-performing checkpoints
- Detailed metrics tracking in `model_metrics.json`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI Whisper for the pre-trained encoder
- PyTorch Lightning for training framework
- RNN-T implementation from warprnnt-numba
