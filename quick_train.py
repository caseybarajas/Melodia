#!/usr/bin/env python3
"""
Quick training script with minimal configuration for testing
"""

import logging
import tensorflow as tf
from pathlib import Path

from melodia.config import TrainingConfig, ModelConfig, DataConfig
from melodia.data.loader import MIDILoader
from melodia.data.processor import DataProcessor
from melodia.model.architecture import MelodiaModel
from melodia.training.trainer import Trainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Quick training with minimal config"""
    
    # Super minimal configuration for fast training
    model_config = ModelConfig(
        embedding_dim=128,          # Much smaller
        num_layers=2,               # Much fewer layers  
        num_heads=4,                # Fewer attention heads
        ff_dim=256,                 # Smaller feed-forward
        max_sequence_length=256,    # Shorter sequences
        dropout_rate=0.1
    )
    
    training_config = TrainingConfig(
        batch_size=8,               # Small batch for fast iteration
        learning_rate=0.001,        # Higher learning rate for faster convergence
        max_epochs=3,               # Just 3 epochs for testing
        validation_split=0.0,       # No validation for speed
        checkpoint_dir="quick_models"
    )
    
    data_config = DataConfig(
        max_sequence_length=256,    # Match model config
        min_sequence_length=32
    )
    
    # Create directories
    Path("quick_models").mkdir(exist_ok=True)
    
    logger.info(f"Quick training: {model_config.num_layers} layers, {model_config.embedding_dim}D, batch_size={training_config.batch_size}")
    
    # Load data
    loader = MIDILoader(data_config)
    processor = DataProcessor(data_config)
    
    data_dir = Path("data")
    midi_files = list(data_dir.glob("**/*.mid"))
    logger.info(f"Found {len(midi_files)} MIDI files")
    
    # Process files (limit to first 3 for speed)
    all_events = []
    for midi_file in midi_files[:3]:  # Only use first 3 files for speed
        try:
            events = loader.load_file(midi_file)
            if events:
                all_events.append(events)
                logger.info(f"Loaded {midi_file.name}: {len(events)} events")
        except Exception as e:
            logger.warning(f"Error processing {midi_file}: {str(e)}")
    
    if not all_events:
        logger.error("No data loaded!")
        return
    
    # Create dataset
    dataset = processor.prepare_dataset(
        all_events,
        batch_size=training_config.batch_size,
        shuffle=True
    )
    
    # Count batches for verification
    batch_count = sum(1 for _ in dataset)
    logger.info(f"Created dataset with {batch_count} batches")
    
    # Create model
    model = MelodiaModel(model_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config
    )
    
    # Train
    logger.info("Starting quick training...")
    try:
        trainer.train(train_dataset=dataset)
        logger.info("✅ Quick training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 