# melodia/training/callbacks.py

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Union
import logging
from ..data.loader import MusicEvent
from ..generation.generator import MusicGenerator

logger = logging.getLogger(__name__)

class GenerationCallback(tf.keras.callbacks.Callback):
    """Callback for generating music samples during training"""
    
    def __init__(
        self,
        generator: MusicGenerator,
        sample_length: int = 512,
        num_samples: int = 1,
        save_dir: Union[str, Path] = "samples",
        generation_frequency: int = 5
    ):
        super().__init__()
        self.generator = generator
        self.sample_length = sample_length
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.generation_frequency = generation_frequency
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Generate samples at the end of specified epochs"""
        if (epoch + 1) % self.generation_frequency == 0:
            logger.info(f"Generating sample at epoch {epoch + 1}")
            
            for i in range(self.num_samples):
                try:
                    # Generate sample
                    events = self.generator.generate(
                        max_length=self.sample_length
                    )
                    
                    # Save as MIDI
                    sample_path = self.save_dir / f"epoch_{epoch + 1}_sample_{i + 1}.mid"
                    self.generator.tokenizer.save_events_as_midi(events, sample_path)
                    
                except Exception as e:
                    logger.error(f"Error generating sample {i + 1}: {str(e)}")

class PerformanceTracker(tf.keras.callbacks.Callback):
    """Tracks and logs detailed performance metrics"""
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        metrics_frequency: int = 100
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.metrics_frequency = metrics_frequency
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_start_time = None
        self.batch_start_time = None
        self.metrics_history = {
            'batch_metrics': [],
            'epoch_metrics': [],
            'performance_metrics': []
        }
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize training metrics"""
        self.training_start_time = time.time()
    
    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Track batch start time"""
        self.batch_start_time = time.time()
    
    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Track batch metrics"""
        if batch % self.metrics_frequency == 0:
            batch_time = time.time() - self.batch_start_time
            
            metrics = {
                'batch': batch,
                'batch_time': batch_time,
                'samples_per_second': self.model.config.batch_size / batch_time
            }
            
            if logs:
                metrics.update(logs)
            
            self.metrics_history['batch_metrics'].append(metrics)
            self._save_metrics()
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Track epoch metrics"""
        epoch_metrics = {
            'epoch': epoch + 1,
            'epoch_time': time.time() - self.training_start_time
        }
        
        if logs:
            epoch_metrics.update(logs)
        
        self.metrics_history['epoch_metrics'].append(epoch_metrics)
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.log_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

class AdaptiveLearningRateScheduler(tf.keras.callbacks.Callback):
    """Adaptive learning rate scheduler with warmup and decay"""
    
    def __init__(
        self,
        initial_lr: float = 0.0001,
        warmup_steps: int = 1000,
        min_lr: float = 1e-6,
        patience: int = 5,
        decay_rate: float = 0.1
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.patience = patience
        self.decay_rate = decay_rate
        
        self.current_lr = 0.0
        self.best_loss = float('inf')
        self.wait = 0
        self.total_steps = 0
    
    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Adjust learning rate before each batch"""
        self.total_steps += 1
        
        if self.total_steps <= self.warmup_steps:
            # Linear warmup
            self.current_lr = (self.total_steps / self.warmup_steps) * self.initial_lr
        
        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate,
            self.current_lr
        )
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check for improvement and adjust learning rate if needed"""
        current_loss = logs.get('loss', 0)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Decay learning rate
                self.current_lr = max(
                    self.current_lr * self.decay_rate,
                    self.min_lr
                )
                self.wait = 0
                
                logger.info(f"Reducing learning rate to {self.current_lr}")

class MemoryOptimizer(tf.keras.callbacks.Callback):
    """Optimizes memory usage during training"""
    
    def __init__(self, memory_limit: Optional[int] = None):
        super().__init__()
        self.memory_limit = memory_limit
        
        if tf.config.list_physical_devices('GPU'):
            if memory_limit:
                # Set memory growth and limit
                for device in tf.config.list_physical_devices('GPU'):
                    try:
                        tf.config.experimental.set_memory_growth(device, True)
                        tf.config.experimental.set_virtual_device_configuration(
                            device,
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )]
                        )
                    except RuntimeError as e:
                        logger.warning(f"Error setting memory limit: {str(e)}")
    
    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Clear unnecessary memory after each batch"""
        if tf.config.list_physical_devices('GPU'):
            # Clear GPU memory cache
            tf.keras.backend.clear_session()

class MusicTheoryConstraints(tf.keras.callbacks.Callback):
    """Applies music theory constraints during training"""
    
    def __init__(
        self,
        allowed_scales: Optional[List[str]] = None,
        allowed_chords: Optional[List[str]] = None
    ):
        super().__init__()
        self.allowed_scales = allowed_scales
        self.allowed_chords = allowed_chords
        
        # Initialize theory constraints
        self._initialize_constraints()
    
    def _initialize_constraints(self):
        """Initialize music theory constraints"""
        # Default scales if none provided
        if not self.allowed_scales:
            self.allowed_scales = ['C', 'G', 'D', 'A', 'E', 'B', 'F']
        
        # Default chords if none provided
        if not self.allowed_chords:
            self.allowed_chords = ['maj', 'min', '7', 'maj7', 'min7']
        
        # Create constraint matrices
        self.scale_constraints = self._create_scale_constraints()
        self.chord_constraints = self._create_chord_constraints()
    
    def _create_scale_constraints(self) -> tf.Tensor:
        """Create constraint matrix for scales"""
        # Implement scale constraints logic
        pass
    
    def _create_chord_constraints(self) -> tf.Tensor:
        """Create constraint matrix for chords"""
        # Implement chord constraints logic
        pass
    
    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Apply music theory constraints after each batch"""
        # Apply constraints to model weights
        pass

def create_training_callbacks(
    model_dir: Union[str, Path],
    generator: Optional[MusicGenerator] = None,
    **kwargs
) -> List[tf.keras.callbacks.Callback]:
    """Create standard set of training callbacks"""
    callbacks = []
    
    # Model checkpointing
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(model_dir) / "weights-{epoch:02d}-{loss:.2f}.h5"),
            save_best_only=True,
            monitor='loss',
            mode='min'
        )
    )
    
    # Early stopping
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=kwargs.get('patience', 10),
            restore_best_weights=True
        )
    )
    
    # Performance tracking
    callbacks.append(
        PerformanceTracker(
            log_dir=Path(model_dir) / "logs",
            metrics_frequency=kwargs.get('metrics_frequency', 100)
        )
    )
    
    # Learning rate scheduling
    callbacks.append(
        AdaptiveLearningRateScheduler(
            initial_lr=kwargs.get('initial_lr', 0.0001),
            warmup_steps=kwargs.get('warmup_steps', 1000)
        )
    )
    
    # Memory optimization
    callbacks.append(
        MemoryOptimizer(
            memory_limit=kwargs.get('memory_limit', None)
        )
    )
    
    # Sample generation during training
    if generator:
        callbacks.append(
            GenerationCallback(
                generator=generator,
                save_dir=Path(model_dir) / "samples",
                generation_frequency=kwargs.get('generation_frequency', 5)
            )
        )
    
    # Music theory constraints
    if kwargs.get('apply_theory_constraints', False):
        callbacks.append(
            MusicTheoryConstraints(
                allowed_scales=kwargs.get('allowed_scales'),
                allowed_chords=kwargs.get('allowed_chords')
            )
        )
    
    return callbacks