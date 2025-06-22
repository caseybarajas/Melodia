# melodia/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from tqdm import tqdm

from ..model.architecture import MelodiaModel, MelodiaTrainer
from ..config import ModelConfig, TrainingConfig
from ..data.processor import DataProcessor

logger = logging.getLogger(__name__)

class Trainer:
    """PyTorch-based trainer for the Melodia model with progress bars"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_processor: DataProcessor,
        model_dir: str = "models"
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_processor = data_processor
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = MelodiaModel(model_config)
        self.pytorch_trainer = MelodiaTrainer(self.model, model_config, training_config)
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch_times': []
        }
        
        # Device info
        self.device = self.pytorch_trainer.device
        logger.info(f"Using device: {self.pytorch_trainer.get_device_info()}")
        
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """Train the model with progress bars"""
        
        logger.info("Starting training...")
        logger.info(f"Training for {self.training_config.max_epochs} epochs starting from epoch {self.current_epoch}")
        
        start_time = time.time()
        
        # Create epoch progress bar
        epoch_pbar = tqdm(
            range(self.current_epoch, self.training_config.max_epochs),
            desc="Epochs",
            position=0,
            leave=True
        )
        
        try:
            for epoch in epoch_pbar:
                epoch_start_time = time.time()
                
                logger.info(f"Starting epoch {epoch+1}/{self.training_config.max_epochs}")
                
                # Training phase
                train_loss, train_accuracy = self._train_epoch(
                    train_dataloader, 
                    epoch, 
                    progress_callback
                )
                
                # Validation phase
                val_loss, val_accuracy = 0.0, 0.0
                if val_dataloader is not None:
                    val_loss, val_accuracy = self._validate_epoch(val_dataloader)
                
                # Update learning rate
                self.pytorch_trainer.scheduler.step()
                
                # Record metrics
                epoch_time = time.time() - epoch_start_time
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_accuracy'].append(train_accuracy)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                self.training_history['epoch_times'].append(epoch_time)
                
                self.current_epoch = epoch + 1
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'loss': f'{train_loss:.4f}',
                    'acc': f'{train_accuracy:.3f}',
                    'time': f'{epoch_time/60:.1f}m'
                })
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                    self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
                
                # Progress callback for GUI
                if progress_callback:
                    progress_callback({
                        'epoch': epoch + 1,
                        'total_epochs': self.training_config.max_epochs,
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                        'epoch_time': epoch_time
                    })
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        
        finally:
            epoch_pbar.close()
            
            # Save final model
            self.save_model()
            
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return self.training_history
    
    def _train_epoch(
        self, 
        train_dataloader: DataLoader, 
        epoch: int,
        progress_callback: Optional[callable] = None
    ) -> Tuple[float, float]:
        """Train for one epoch with batch progress bar"""
        
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(train_dataloader)
        
        # Create batch progress bar
        batch_pbar = tqdm(
            train_dataloader,
            desc=f"Training",
            position=1,
            leave=False
        )
        
        for batch_idx, (batch_inputs, batch_targets) in enumerate(batch_pbar):
            # Training step
            loss, accuracy = self.pytorch_trainer.train_step(batch_inputs, batch_targets)
            
            total_loss += loss
            total_accuracy += accuracy
            
            # Update batch progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_accuracy = total_accuracy / (batch_idx + 1)
            
            batch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_accuracy:.3f}'
            })
            
            # Progress callback for GUI (more frequent updates)
            if progress_callback and batch_idx % 10 == 0:
                progress_callback({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'total_batches': num_batches,
                    'batch_loss': loss,
                    'avg_loss': avg_loss,
                    'avg_accuracy': avg_accuracy
                })
        
        batch_pbar.close()
        
        avg_train_loss = total_loss / num_batches
        avg_train_accuracy = total_accuracy / num_batches
        
        return avg_train_loss, avg_train_accuracy
    
    def _validate_epoch(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_dataloader:
                loss, accuracy = self.pytorch_trainer.validate_step(batch_inputs, batch_targets)
                total_loss += loss
                total_accuracy += accuracy
        
        avg_val_loss = total_loss / num_batches
        avg_val_accuracy = total_accuracy / num_batches
        
        return avg_val_loss, avg_val_accuracy
    
    def save_model(self, filename: str = "melodia_model.pt"):
        """Save the complete model"""
        model_path = self.model_dir / filename
        
        # Save model
        self.pytorch_trainer.save_model(str(model_path))
        
        # Save additional training info
        training_info = {
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'current_epoch': self.current_epoch,
            'training_history': self.training_history,
            'vocab_size': self.data_processor.tokenizer.vocab_size
        }
        
        info_path = self.model_dir / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training info saved to {info_path}")
    
    def save_checkpoint(self, filename: str):
        """Save a training checkpoint"""
        checkpoint_path = self.model_dir / "checkpoints" / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        self.pytorch_trainer.save_model(str(checkpoint_path))
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        self.pytorch_trainer.load_model(model_path)
        
        # Try to load training info
        info_path = Path(model_path).parent / "training_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                training_info = json.load(f)
                self.current_epoch = training_info.get('current_epoch', 0)
                self.training_history = training_info.get('training_history', {
                    'train_loss': [],
                    'train_accuracy': [],
                    'val_loss': [],
                    'val_accuracy': [],
                    'epoch_times': []
                })
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': str(self.device),
            'current_epoch': self.current_epoch,
            'vocab_size': self.data_processor.tokenizer.vocab_size
        }