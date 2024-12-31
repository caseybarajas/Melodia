# melodia/scripts/train.py

import argparse
import logging
from pathlib import Path
import tensorflow as tf
from typing import Optional, Tuple

from melodia.config import TrainingConfig, ModelConfig, DataConfig
from melodia.data.loader import MIDILoader
from melodia.data.processor import DataProcessor
from melodia.model.architecture import MelodiaModel
from melodia.training.trainer import Trainer
from melodia.generation.generator import MusicGenerator
from melodia.evaluation.metrics import MusicEvaluator

logger = logging.getLogger(__name__)

def setup_logging(log_dir: Path):
    """Set up logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Melodia model')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training MIDI files')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--validation_split', type=float, default=0.1)
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # Generation parameters
    parser.add_argument('--generate_samples', action='store_true',
                       help='Generate samples during training')
    parser.add_argument('--num_samples', type=int, default=5)
    
    return parser.parse_args()

def prepare_data(
    data_dir: Path,
    config: TrainingConfig
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
    """Prepare training and validation datasets"""
    logger.info("Preparing datasets...")
    
    # Create data config
    data_config = DataConfig(
        max_sequence_length=512,
        min_sequence_length=64,
        valid_time_signatures=[(4, 4), (3, 4), (6, 8)]
    )
    
    # Load MIDI files
    loader = MIDILoader(data_config)
    processor = DataProcessor(config)
    
    midi_files = list(data_dir.glob('**/*.mid'))
    if not midi_files:
        raise ValueError(f"No MIDI files found in {data_dir}")
    
    # Process all files
    all_events = []
    for midi_file in midi_files:
        try:
            events = loader.read_midi(midi_file)
            if events:
                all_events.extend(events)
        except Exception as e:
            logger.warning(f"Error processing {midi_file}: {str(e)}")
    
    # Create datasets
    if config.validation_split > 0:
        split_idx = int(len(all_events) * (1 - config.validation_split))
        train_events = all_events[:split_idx]
        val_events = all_events[split_idx:]
        
        train_dataset = processor.prepare_dataset(
            train_events,
            batch_size=config.batch_size,
            shuffle=True
        )
        val_dataset = processor.prepare_dataset(
            val_events,
            batch_size=config.batch_size,
            shuffle=False
        )
        return train_dataset, val_dataset
    
    train_dataset = processor.prepare_dataset(
        all_events,
        batch_size=config.batch_size,
        shuffle=True
    )
    return train_dataset, None

def main():
    """Main training function"""
    args = parse_args()
    
    # Create directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(model_dir / 'logs')
    
    # Create configurations
    model_config = ModelConfig(
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        validation_split=args.validation_split,
        checkpoint_dir=str(model_dir / 'checkpoints')
    )
    
    # Prepare data
    train_dataset, val_dataset = prepare_data(
        Path(args.data_dir),
        training_config
    )
    
    # Create model
    model = MelodiaModel(model_config)
    
    # Create generator and evaluator if needed
    generator = None
    evaluator = None
    if args.generate_samples:
        generator = MusicGenerator(model, training_config)
        evaluator = MusicEvaluator()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        generator=generator,
        evaluator=evaluator
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        trainer.train(
            train_dataset=train_dataset,
            validation_dataset=val_dataset
        )
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
    finally:
        # Save final model state
        trainer.save_training_state()

if __name__ == '__main__':
    main()