#!/usr/bin/env python3
"""
Custom model checkpoint callback for step-based saving with WER-based selection.

This module implements a custom PyTorch Lightning callback that:
1. Saves models based on training steps instead of epochs
2. Keeps only the top K models based on WER performance
3. Saves only model weights to reduce storage
4. Automatically cleans up worse performing models
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ModelMetrics:
    """Store metrics for a saved model."""
    step: int
    epoch: int
    val_wer: float
    val_loss: float
    timestamp: str
    filepath: str

class StepBasedModelCheckpoint(Callback):
    """
    Custom callback for step-based model checkpointing with WER-based selection.
    
    Features:
    - Save models every N training steps
    - Keep only top K models based on WER
    - Save weights only to reduce storage
    - Automatic cleanup of worse models
    - Detailed logging and metrics tracking
    """
    
    def __init__(
        self,
        dirpath: str,
        save_every_n_steps: int = 1000,
        keep_top_k: int = 3,
        monitor_metric: str = 'val_wer',
        mode: str = 'min',
        save_weights_only: bool = True,
        filename_template: str = 'rnnt-step-{step:06d}-wer-{val_wer:.4f}',
        verbose: bool = True
    ):
        """
        Initialize the checkpoint callback.
        
        Args:
            dirpath: Directory to save checkpoints
            save_every_n_steps: Save frequency in training steps
            keep_top_k: Number of best models to keep
            monitor_metric: Metric to monitor ('val_wer' or 'val_loss')
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            save_weights_only: If True, save only model weights
            filename_template: Template for checkpoint filenames
            verbose: Enable verbose logging
        """
        super().__init__()
        
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        self.save_every_n_steps = save_every_n_steps
        self.keep_top_k = keep_top_k
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.filename_template = filename_template
        self.verbose = verbose
        
        # Track saved models
        self.saved_models: List[ModelMetrics] = []
        self.metrics_file = self.dirpath / "model_metrics.json"
        
        # Load existing metrics if available
        self._load_existing_metrics()
        
        # Validation
        assert mode in ['min', 'max'], f"Mode must be 'min' or 'max', got {mode}"
        assert monitor_metric in ['val_wer', 'val_loss'], \
            f"Monitor metric must be 'val_wer' or 'val_loss', got {monitor_metric}"
        
        if self.verbose:
            logger.info(f"StepBasedModelCheckpoint initialized:")
            logger.info(f"  - Save every {save_every_n_steps} steps")
            logger.info(f"  - Keep top {keep_top_k} models")
            logger.info(f"  - Monitor: {monitor_metric} ({mode})")
            logger.info(f"  - Save weights only: {save_weights_only}")
            logger.info(f"  - Directory: {dirpath}")
    
    def _load_existing_metrics(self):
        """Load existing model metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                self.saved_models = [
                    ModelMetrics(**metrics) for metrics in metrics_data
                ]
                
                # Verify files still exist
                self.saved_models = [
                    model for model in self.saved_models
                    if Path(model.filepath).exists()
                ]
                
                if self.verbose:
                    logger.info(f"Loaded {len(self.saved_models)} existing model metrics")
                    
            except Exception as e:
                logger.warning(f"Failed to load existing metrics: {e}")
                self.saved_models = []
    
    def _save_metrics(self):
        """Save model metrics to file."""
        try:
            metrics_data = [asdict(model) for model in self.saved_models]
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _should_save(self, trainer: pl.Trainer) -> bool:
        """Check if model should be saved at current step."""
        return trainer.global_step % self.save_every_n_steps == 0 and trainer.global_step > 0
    
    def _get_metric_value(self, trainer: pl.Trainer, logs: Dict[str, Any]) -> Optional[float]:
        """Get the monitored metric value."""
        if self.monitor_metric in logs:
            return float(logs[self.monitor_metric])
        
        # Try to get from trainer's logged metrics
        if hasattr(trainer, 'logged_metrics') and self.monitor_metric in trainer.logged_metrics:
            return float(trainer.logged_metrics[self.monitor_metric])
        
        return None
    
    def _is_better_model(self, current_metric: float, best_metric: float) -> bool:
        """Check if current model is better than the best saved model."""
        if self.mode == 'min':
            return current_metric < best_metric
        else:
            return current_metric > best_metric
    
    def _get_filename(self, step: int, metrics: Dict[str, Any]) -> str:
        """Generate filename for the checkpoint."""
        try:
            # Prepare metrics for filename formatting
            format_dict = {
                'step': step,
                'epoch': metrics.get('epoch', 0),
                'val_wer': metrics.get('val_wer', 0.0),
                'val_loss': metrics.get('val_loss', 0.0)
            }
            
            filename = self.filename_template.format(**format_dict)
            return filename + '.pt'
            
        except Exception as e:
            logger.warning(f"Failed to format filename: {e}")
            return f"rnnt-step-{step:06d}.pt"
    
    def _save_model(self, trainer: pl.Trainer, step: int, metrics: Dict[str, Any]) -> str:
        """Save the model and return the filepath."""
        filename = self._get_filename(step, metrics)
        filepath = self.dirpath / filename
        
        try:
            if self.save_weights_only:
                # Save only model weights
                torch.save(trainer.model.state_dict(), filepath)
            else:
                # Save full checkpoint
                trainer.save_checkpoint(filepath)
            
            if self.verbose:
                logger.info(f"Model saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def _cleanup_models(self):
        """Remove models that are not in the top K."""
        if len(self.saved_models) <= self.keep_top_k:
            return

        # Map metric names to ModelMetrics attributes
        metric_attr_map = {
            'val_wer': 'val_wer',
            'val_loss': 'val_loss',
            'train_loss': 'val_loss'  # Use val_loss field for train_loss fallback
        }

        attr_name = metric_attr_map.get(self.monitor_metric, 'val_wer')

        # Sort models by the monitored metric
        if self.mode == 'min':
            # For WER/loss, lower is better
            self.saved_models.sort(key=lambda x: getattr(x, attr_name))
        else:
            # For accuracy, higher is better
            self.saved_models.sort(key=lambda x: getattr(x, attr_name), reverse=True)

        # Keep only top K models
        models_to_remove = self.saved_models[self.keep_top_k:]
        self.saved_models = self.saved_models[:self.keep_top_k]

        # Delete files for removed models
        for model in models_to_remove:
            try:
                filepath = Path(model.filepath)
                if filepath.exists():
                    filepath.unlink()
                    if self.verbose:
                        logger.info(f"Removed model: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove {model.filepath}: {e}")
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """Called when training batch ends - check if we should save."""
        if not self._should_save(trainer):
            return

        # Get current metrics
        logs = trainer.logged_metrics

        # For step-based saving, we might not have val_wer yet
        # Use train_loss as fallback if val_wer is not available
        if self.monitor_metric == 'val_wer' and 'val_wer' not in logs:
            # Use train_loss as temporary metric until we have val_wer
            current_metric = logs.get('train_loss', float('inf'))
            monitor_metric_name = 'train_loss'
            mode = 'min'  # Lower loss is better
        else:
            current_metric = self._get_metric_value(trainer, logs)
            monitor_metric_name = self.monitor_metric
            mode = self.mode

        if current_metric is None:
            if self.verbose:
                logger.debug(f"No metric available at step {trainer.global_step}. Skipping save.")
            return

        # Check if we should save this model
        should_save = True
        if len(self.saved_models) >= self.keep_top_k:
            # Map metric names to ModelMetrics attributes
            metric_attr_map = {
                'val_wer': 'val_wer',
                'val_loss': 'val_loss',
                'train_loss': 'val_loss'  # Use val_loss field for train_loss fallback
            }

            attr_name = metric_attr_map.get(monitor_metric_name, 'val_loss')

            # Check if current model is better than the worst saved model
            if mode == 'min':
                worst_metric = max(getattr(m, attr_name) for m in self.saved_models)
                should_save = current_metric < worst_metric
            else:
                worst_metric = min(getattr(m, attr_name) for m in self.saved_models)
                should_save = current_metric > worst_metric

        if should_save:
            # Save the model
            metrics_dict = {
                'val_wer': logs.get('val_wer', 0.0),
                'val_loss': logs.get('val_loss', logs.get('train_loss', 0.0)),
                'epoch': trainer.current_epoch
            }

            filepath = self._save_model(trainer, trainer.global_step, metrics_dict)

            # Add to saved models list
            model_metrics = ModelMetrics(
                step=trainer.global_step,
                epoch=trainer.current_epoch,
                val_wer=float(logs.get('val_wer', 0.0)),
                val_loss=float(logs.get('val_loss', logs.get('train_loss', 0.0))),
                timestamp=datetime.now().isoformat(),
                filepath=filepath
            )

            self.saved_models.append(model_metrics)

            # Cleanup old models
            self._cleanup_models()

            # Save metrics
            self._save_metrics()

            if self.verbose:
                metric_name = monitor_metric_name if monitor_metric_name != 'train_loss' else 'train_loss (fallback)'
                logger.info(f"💾 Step {trainer.global_step}: {metric_name}={current_metric:.4f}")
                logger.info(f"📁 Keeping {len(self.saved_models)} best models")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when validation ends - update saved models with val_wer if available."""
        if not self._should_save(trainer):
            return

        # Get current metrics
        logs = trainer.logged_metrics
        current_metric = self._get_metric_value(trainer, logs)

        if current_metric is None:
            logger.warning(f"Metric {self.monitor_metric} not found in logs after validation.")
            return

        # If we have val_wer now, we might want to re-evaluate our saved models
        if 'val_wer' in logs and self.monitor_metric == 'val_wer':
            # Check if we should save this model based on val_wer
            should_save = True
            if len(self.saved_models) >= self.keep_top_k:
                if self.mode == 'min':
                    worst_metric = max(getattr(m, 'val_wer') for m in self.saved_models)
                    should_save = current_metric < worst_metric
                else:
                    worst_metric = min(getattr(m, 'val_wer') for m in self.saved_models)
                    should_save = current_metric > worst_metric

            if should_save:
                # Save the model with proper val_wer
                metrics_dict = {
                    'val_wer': logs.get('val_wer', 0.0),
                    'val_loss': logs.get('val_loss', 0.0),
                    'epoch': trainer.current_epoch
                }

                filepath = self._save_model(trainer, trainer.global_step, metrics_dict)

                # Add to saved models list
                model_metrics = ModelMetrics(
                    step=trainer.global_step,
                    epoch=trainer.current_epoch,
                    val_wer=float(logs.get('val_wer', 0.0)),
                    val_loss=float(logs.get('val_loss', 0.0)),
                    timestamp=datetime.now().isoformat(),
                    filepath=filepath
                )

                self.saved_models.append(model_metrics)

                # Cleanup old models
                self._cleanup_models()

                # Save metrics
                self._save_metrics()

                if self.verbose:
                    logger.info(f"🎯 Step {trainer.global_step}: val_wer={current_metric:.4f} (validation)")
                    logger.info(f"📁 Keeping {len(self.saved_models)} best models")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when training ends."""
        # Save final model regardless of performance
        logs = trainer.logged_metrics
        metrics_dict = {
            'val_wer': logs.get('val_wer', 0.0),
            'val_loss': logs.get('val_loss', 0.0),
            'epoch': trainer.current_epoch
        }
        
        filename = f"rnnt-final-step-{trainer.global_step:06d}.pt"
        filepath = self.dirpath / filename
        
        try:
            if self.save_weights_only:
                torch.save(trainer.model.state_dict(), filepath)
            else:
                trainer.save_checkpoint(filepath)
            
            if self.verbose:
                logger.info(f"Final model saved: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
    
    def get_best_model_path(self) -> Optional[str]:
        """Get the path to the best saved model."""
        if not self.saved_models:
            return None

        # Map metric names to ModelMetrics attributes
        metric_attr_map = {
            'val_wer': 'val_wer',
            'val_loss': 'val_loss',
            'train_loss': 'val_loss'  # Use val_loss field for train_loss fallback
        }

        attr_name = metric_attr_map.get(self.monitor_metric, 'val_wer')

        if self.mode == 'min':
            best_model = min(self.saved_models, key=lambda x: getattr(x, attr_name))
        else:
            best_model = max(self.saved_models, key=lambda x: getattr(x, attr_name))

        return best_model.filepath
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of saved models."""
        if not self.saved_models:
            return {"num_models": 0, "models": []}
        
        return {
            "num_models": len(self.saved_models),
            "best_model": self.get_best_model_path(),
            "models": [asdict(model) for model in self.saved_models]
        }
