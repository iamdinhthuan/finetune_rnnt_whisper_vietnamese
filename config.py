#!/usr/bin/env python3
"""
Configuration management for Whisper RNN-T model.

This module provides configuration loading, validation, and access
for all model parameters, replacing the old constants.py system.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80

@dataclass
class WhisperConfig:
    """Whisper model configuration."""
    n_state: int = 512  # Base model
    n_head: int = 8     # Base model
    n_layer: int = 6    # Base model

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    attention_context_size: tuple = (80, 3)

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    max_duration: float = 15.1
    min_duration: float = 0.9
    min_text_len: int = 3
    max_text_len: int = 200
    train_manifest: list = field(default_factory=lambda: ["./data/sample.jsonl"])
    val_manifest: list = field(default_factory=lambda: ["./data/sample.jsonl"])
    bg_noise_paths: list = field(default_factory=list)

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    lr: float = 1e-4
    min_lr: float = 1e-5
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.01

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    total_steps: int = 3000000
    warmup_steps: int = 2000
    type: str = "linear"  # "linear" or "cosine_annealing"
    min_lr_ratio: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    num_workers: int = 16
    max_epochs: int = 50
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: str = "norm"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    vocab_size: int = 1024
    model_path: str = './utils/tokenizer_spe_bpe_v1024_pad/tokenizer.model'
    rnnt_blank: int = 1024
    pad_token: int = 1

@dataclass
class PathsConfig:
    """File paths configuration."""
    pretrained_encoder_weight: str = './weights/base_encoder.pt'
    log_dir: str = './checkpoints'

@dataclass
class ModelSavingConfig:
    """Model saving strategy configuration."""
    save_every_n_steps: int = 1000
    keep_top_k: int = 3
    monitor_metric: str = 'val_wer'
    mode: str = 'min'  # 'min' for WER, 'max' for accuracy
    save_weights_only: bool = True
    filename_template: str = 'rnnt-step-{step:06d}-wer-{val_wer:.4f}'

@dataclass
class Config:
    """Main configuration class."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model_saving: ModelSavingConfig = field(default_factory=ModelSavingConfig)

class ConfigManager:
    """Configuration manager for loading and validating configs."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager.
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml in current directory.
        """
        self.config_path = config_path or self._find_config_file()
        self._config = None
    
    def _find_config_file(self) -> str:
        """Find config file in current directory or parent directories."""
        current_dir = Path.cwd()
        
        # Look for config.yaml in current directory and parent directories
        for path in [current_dir] + list(current_dir.parents):
            config_file = path / "config.yaml"
            if config_file.exists():
                return str(config_file)
        
        # If not found, return default path
        return "config.yaml"
    
    def load_config(self) -> Config:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config
        
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            logger.info("Using default configuration")
            self._config = Config()
            return self._config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            self._config = self._dict_to_config(config_dict)
            logger.info(f"Configuration loaded from: {self.config_path}")
            
            # Validate configuration
            self._validate_config(self._config)
            
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            self._config = Config()
            return self._config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        config = Config()
        
        # Audio config
        if 'audio' in config_dict:
            audio_dict = config_dict['audio']
            config.audio = AudioConfig(**audio_dict)
        
        # Model config
        if 'model' in config_dict:
            model_dict = config_dict['model']
            whisper_config = WhisperConfig()
            if 'whisper' in model_dict:
                whisper_config = WhisperConfig(**model_dict['whisper'])
            
            config.model = ModelConfig(
                whisper=whisper_config,
                attention_context_size=tuple(model_dict.get('attention_context_size', (80, 3)))
            )
        
        # Dataset config
        if 'dataset' in config_dict:
            dataset_dict = config_dict['dataset']
            # Ensure proper types
            dataset_dict = {
                'max_duration': float(dataset_dict.get('max_duration', 15.1)),
                'min_duration': float(dataset_dict.get('min_duration', 0.9)),
                'min_text_len': int(dataset_dict.get('min_text_len', 3)),
                'max_text_len': int(dataset_dict.get('max_text_len', 200)),
                'train_manifest': dataset_dict.get('train_manifest', ["./data/sample.jsonl"]),
                'val_manifest': dataset_dict.get('val_manifest', ["./data/sample.jsonl"]),
                'bg_noise_paths': dataset_dict.get('bg_noise_paths', [])
            }
            config.dataset = DatasetConfig(**dataset_dict)
        
        # Training config
        if 'training' in config_dict:
            training_dict = config_dict['training']
            optimizer_config = OptimizerConfig()
            scheduler_config = SchedulerConfig()
            
            if 'optimizer' in training_dict:
                opt_dict = training_dict['optimizer']
                # Ensure proper types for optimizer config
                opt_dict = {
                    'lr': float(opt_dict.get('lr', 1e-4)),
                    'min_lr': float(opt_dict.get('min_lr', 1e-5)),
                    'betas': tuple(opt_dict.get('betas', [0.9, 0.98])),
                    'eps': float(opt_dict.get('eps', 1e-9)),
                    'weight_decay': float(opt_dict.get('weight_decay', 0.01))
                }
                optimizer_config = OptimizerConfig(**opt_dict)
            
            if 'scheduler' in training_dict:
                sched_dict = training_dict['scheduler']
                # Ensure proper types for scheduler config
                sched_dict = {
                    'total_steps': int(sched_dict.get('total_steps', 3000000)),
                    'warmup_steps': int(sched_dict.get('warmup_steps', 2000)),
                    'type': str(sched_dict.get('type', 'linear')),
                    'min_lr_ratio': float(sched_dict.get('min_lr_ratio', 0.1))
                }
                scheduler_config = SchedulerConfig(**sched_dict)
            
            # Remove nested configs from training_dict and ensure proper types
            training_dict_clean = {
                'batch_size': int(training_dict.get('batch_size', 32)),
                'accumulate_grad_batches': int(training_dict.get('accumulate_grad_batches', 1)),
                'num_workers': int(training_dict.get('num_workers', 16)),
                'max_epochs': int(training_dict.get('max_epochs', 50)),
                'gradient_clip_val': training_dict.get('gradient_clip_val'),
                'gradient_clip_algorithm': str(training_dict.get('gradient_clip_algorithm', 'norm'))
            }

            config.training = TrainingConfig(
                optimizer=optimizer_config,
                scheduler=scheduler_config,
                **training_dict_clean
            )
        
        # Tokenizer config
        if 'tokenizer' in config_dict:
            config.tokenizer = TokenizerConfig(**config_dict['tokenizer'])
        
        # Paths config
        if 'paths' in config_dict:
            config.paths = PathsConfig(**config_dict['paths'])
        
        # Model saving config
        if 'model_saving' in config_dict:
            config.model_saving = ModelSavingConfig(**config_dict['model_saving'])
        
        return config
    
    def _validate_config(self, config: Config):
        """Validate configuration values."""
        try:
            # Basic validation - just check that values exist and are reasonable
            assert config.audio.sample_rate > 0
            assert config.audio.n_fft > 0
            assert config.audio.hop_length > 0
            assert config.audio.n_mels > 0

            assert config.model.whisper.n_state > 0
            assert config.model.whisper.n_head > 0
            assert config.model.whisper.n_layer > 0

            assert config.dataset.max_duration > config.dataset.min_duration
            assert config.dataset.min_text_len >= 0
            assert config.dataset.max_text_len > config.dataset.min_text_len

            assert config.training.batch_size > 0
            assert config.training.num_workers >= 0
            assert config.training.max_epochs > 0
            assert config.training.optimizer.lr > 0
            assert config.training.scheduler.total_steps > 0

            logger.info("✅ Configuration validation passed")

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            logger.info("Using default configuration instead")
            # Don't raise, just log and continue with defaults

# Global config instance
_config_manager = None
_config = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance."""
    global _config_manager, _config
    
    if _config is None:
        _config_manager = ConfigManager(config_path)
        _config = _config_manager.load_config()
    
    return _config

def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config_manager, _config
    
    _config_manager = ConfigManager(config_path)
    _config = _config_manager.load_config()
    
    return _config
