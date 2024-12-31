# melodia/training/trainer.py

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Callable
import time
import logging
from pathlib import Path
from datetime import datetime
import json
from ..config import TrainingConfig
from .callbacks import create_training_callbacks
from ..evaluation.metrics import MusicEvaluator
from ..data.loader import MusicEvent
from ..generation.generator import MusicGenerator

logger = logging.getLogger(__name__)

class Trainer:
    """Handles model training and validation"""
    
    def __init__(
        self,
        model: tf.keras.Model,
        config: TrainingConfig,
        generator: Optional[MusicGenerator] = None,
        evaluator: Optional[MusicEvaluator] = None,
        strategy: Optional[tf.distribute.Strategy] = None
    ):
        self.model = model
        self.config = config
        self.generator = generator
        self.evaluator = evaluator or MusicEvaluator()
        
        # Set up distributed training if strategy provided
        self.strategy = strategy or self._setup_default_strategy()
        with self.strategy.scope():
            self._setup_model()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        self.validation_metrics = []
        
        # Create checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.config.checkpoint_dir,
            max_to_keep=5
        )
    
    def _setup_default_strategy(self) -> tf.distribute.Strategy:
        """Set up default distribution strategy"""
        if len(tf.config.list_physical_devices('GPU')) > 1:
            return tf.distribute.MirroredStrategy()
        return tf.distribute.get_strategy()
    
    def _setup_model(self):
        """Set up model, optimizer, and loss"""
        # Create optimizer with learning rate schedule
        self.learning_rate = self._create_learning_rate_schedule()
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.config.weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Set up loss functions
        self.loss_fn = self._create_loss_function()
        
        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy']
        )
    
    def _create_learning_rate_schedule(self) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """Create learning rate schedule with warmup and decay"""
        initial_learning_rate = self.config.learning_rate
        decay_steps = self.config.max_epochs * 1000  # Approximate steps per epoch
        
        # Linear warmup followed by cosine decay
        warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0,
            decay_steps=self.config.warmup_steps,
            end_learning_rate=initial_learning_rate
        )
        
        cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps - self.config.warmup_steps,
            alpha=0.1  # Minimum learning rate factor
        )
        
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[self.config.warmup_steps],
            values=[warmup_schedule, cosine_schedule]
        )
    
    def _create_loss_function(self) -> Callable:
        """Create main loss function with potential regularization"""
        def loss_fn(y_true, y_pred):
            # Main prediction loss
            prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true, y_pred, from_logits=True
            )
            
            # Optional regularization losses
            reg_loss = sum(self.model.losses) if self.model.losses else 0
            
            return prediction_loss + reg_loss
        
        return loss_fn
    
    @tf.function
    def _train_step(
        self,
        inputs: tf.Tensor,
        targets: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Execute single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(targets, predictions)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(
            gradients,
            self.config.gradient_clip_val
        )
        
        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        
        return {
            'loss': loss,
            'accuracy': tf.keras.metrics.sparse_categorical_accuracy(targets, predictions)
        }
    
    @tf.function
    def _validate_step(
        self,
        inputs: tf.Tensor,
        targets: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Execute single validation step"""
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(targets, predictions)
        
        return {
            'loss': loss,
            'accuracy': tf.keras.metrics.sparse_categorical_accuracy(targets, predictions)
        }
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: Optional[tf.data.Dataset] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> Dict:
        """Train the model"""
        logger.info("Starting training...")
        start_time = time.time()
        
        # Distribute datasets
        train_dist = self.strategy.experimental_distribute_dataset(train_dataset)
        if validation_dataset is not None:
            val_dist = self.strategy.experimental_distribute_dataset(validation_dataset)
        
        # Set up callbacks
        callbacks = callbacks or create_training_callbacks(
            self.config.checkpoint_dir,
            self.generator
        )
        
        # Training loop
        try:
            for epoch in range(self.current_epoch, self.config.max_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self._train_epoch(train_dist)
                
                # Validation phase
                val_metrics = {}
                if validation_dataset is not None:
                    val_metrics = self._validate_epoch(val_dist)
                
                # Update metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }
                
                # Generate validation samples if generator available
                if self.generator and (epoch + 1) % self.config.generation_frequency == 0:
                    self._generate_validation_samples(epoch)
                
                # Save checkpoint
                if val_metrics.get('loss', float('inf')) < self.best_loss:
                    self.best_loss = val_metrics.get('loss', float('inf'))
                    self.checkpoint_manager.save()
                
                # Log progress
                epoch_time = time.time() - epoch_start_time
                self._log_progress(epoch, metrics, epoch_time)
                
                # Update history
                self.training_history.append(metrics)
                
                # Run callbacks
                for callback in callbacks:
                    callback.on_epoch_end(epoch, metrics)
                
                # Check early stopping
                if self._should_stop_early(val_metrics.get('loss', float('inf'))):
                    logger.info("Early stopping triggered")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Final save
            self.save_training_state()
            
            # Training summary
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
            
            return self.training_history
    
    def _train_epoch(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Train for one epoch"""
        metrics = []
        
        for batch in dataset:
            batch_metrics = self.strategy.run(
                self._train_step,
                args=(batch[0], batch[1])
            )
            metrics.append(self._reduce_metrics(batch_metrics))
        
        return self._average_metrics(metrics)
    
    def _validate_epoch(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Validate for one epoch"""
        metrics = []
        
        for batch in dataset:
            batch_metrics = self.strategy.run(
                self._validate_step,
                args=(batch[0], batch[1])
            )
            metrics.append(self._reduce_metrics(batch_metrics))
        
        return self._average_metrics(metrics)
    
    def _reduce_metrics(self, metrics: Dict[str, tf.Tensor]) -> Dict[str, float]:
        """Reduce metrics across devices in distributed training"""
        return {
            k: self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                v,
                axis=None
            )
            for k, v in metrics.items()
        }
    
    def _average_metrics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across batches"""
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics])
        return avg_metrics
    
    def _generate_validation_samples(self, epoch: int):
        """Generate and evaluate validation samples"""
        if not self.generator:
            return
        
        samples_dir = Path(self.config.checkpoint_dir) / "samples" / f"epoch_{epoch + 1}"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.config.num_validation_samples):
            try:
                # Generate sample
                events = self.generator.generate(
                    max_length=self.config.max_sequence_length
                )
                
                # Save sample
                sample_path = samples_dir / f"sample_{i + 1}.mid"
                self.generator.save_midi(events, sample_path)
                
                # Evaluate sample
                if self.evaluator:
                    metrics = self.evaluator.evaluate(events)
                    self.validation_metrics.append(metrics)
                    
                    # Save metrics
                    metrics_path = samples_dir / f"sample_{i + 1}_metrics.json"
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            except Exception as e:
                logger.error(f"Error generating sample {i + 1}: {str(e)}")
    
    def _should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        if len(self.training_history) < self.config.early_stopping_patience:
            return False
        
        recent_losses = [
            h.get('val_loss', float('inf'))
            for h in self.training_history[-self.config.early_stopping_patience:]
        ]
        
        return all(loss >= val_loss for loss in recent_losses)
    
    def _log_progress(
        self,
        epoch: int,
        metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log training progress"""
        metrics_str = " - ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
            if k != 'epoch'
        )
        logger.info(
            f"Epoch {epoch + 1}/{self.config.max_epochs} - "
            f"Time: {epoch_time:.2f}s - {metrics_str}"
        )
    
    def save_training_state(self):
        """Save complete training state"""
        state_dir = Path(self.config.checkpoint_dir) / "training_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        history_path = state_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save validation metrics
        if self.validation_metrics:
            metrics_path = state_dir / "validation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.validation_metrics, f, indent=2)
        
        # Save current state
        state = {
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'timestamp': datetime.now().isoformat()
        }
        state_path = state_dir / "state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_training_state(self):
        """Load training state"""
        state_dir = Path(self.config.checkpoint_dir) / "training_state"
        
        # Load training history
        history_path = state_dir / "history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        # Load validation metrics
        metrics_path = state_dir / "validation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.validation_metrics = json.load(f)
        
        # Load current state
        state_path = state_dir / "state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
                self.current_epoch = state['current_epoch']
                self.best_loss = state['best_loss']