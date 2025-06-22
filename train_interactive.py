#!/usr/bin/env python3
"""
Interactive Melodia Training Script
Run with: python train_interactive.py
"""

import os
import sys
import logging
from pathlib import Path
import tensorflow as tf
from typing import Optional, Tuple

# Add melodia to path
sys.path.insert(0, str(Path(__file__).parent))

from melodia.config import TrainingConfig, ModelConfig, DataConfig
from melodia.data.loader import MIDILoader
from melodia.data.processor import DataProcessor
from melodia.model.architecture import MelodiaModel
from melodia.training.trainer import Trainer

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

def check_gpu():
    """Check and configure GPU"""
    print("🔍 Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Configure GPU memory growth to avoid allocation issues
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth configured")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration warning: {e}")
        
        return True
    else:
        print("❌ No GPU found - training will be MUCH slower on CPU")
        print("💡 To install GPU support:")
        print("   pip install tensorflow[and-cuda]  # For CUDA GPUs")
        return False

def get_user_input(prompt: str, default: str, input_type: type = str):
    """Get user input with default value"""
    user_input = input(f"{prompt} [{default}]: ").strip()
    if not user_input:
        return input_type(default)
    try:
        return input_type(user_input)
    except ValueError:
        print(f"❌ Invalid input, using default: {default}")
        return input_type(default)

def interactive_config():
    """Get configuration from user interactively"""
    print("\n🎵 ============================================ 🎵")
    print("    MELODIA INTERACTIVE TRAINING SETUP")
    print("🎵 ============================================ 🎵\n")
    
    # Data configuration
    print("📁 DATA CONFIGURATION")
    print("-" * 30)
    data_dir = get_user_input("Training data directory", "data", str)
    if not Path(data_dir).exists():
        print(f"❌ Directory {data_dir} doesn't exist!")
        return None
    
    # Check available MIDI files
    midi_files = list(Path(data_dir).glob("**/*.mid"))
    print(f"✅ Found {len(midi_files)} MIDI files")
    if len(midi_files) == 0:
        print("❌ No MIDI files found!")
        return None
    
    # Model configuration
    print("\n🧠 MODEL CONFIGURATION")
    print("-" * 30)
    
    # Detect if GPU is available for model size recommendations
    has_gpu = check_gpu()
    
    if has_gpu:
        print("🚀 GPU detected - you can use larger models!")
        default_embedding = "256"
        default_layers = "4"
        default_heads = "8"
    else:
        print("🐌 CPU only - using smaller model for reasonable speed")
        default_embedding = "128"
        default_layers = "2"
        default_heads = "4"
    
    embedding_dim = get_user_input("Embedding dimension", default_embedding, int)
    num_layers = get_user_input("Number of transformer layers", default_layers, int)
    num_heads = get_user_input("Number of attention heads", default_heads, int)
    max_seq_len = get_user_input("Max sequence length", "512", int)
    
    # Training configuration
    print("\n📚 TRAINING CONFIGURATION")
    print("-" * 30)
    
    if has_gpu:
        default_batch = "16"
        default_epochs = "50"
    else:
        default_batch = "8"
        default_epochs = "10"
    
    batch_size = get_user_input("Batch size", default_batch, int)
    epochs = get_user_input("Number of epochs", default_epochs, int)
    learning_rate = get_user_input("Learning rate", "0.001", float)
    validation_split = get_user_input("Validation split (0.0-1.0)", "0.1", float)
    
    # Output configuration
    print("\n💾 OUTPUT CONFIGURATION")
    print("-" * 30)
    model_dir = get_user_input("Model output directory", "models", str)
    
    # Create configs
    model_config = ModelConfig(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_sequence_length=max_seq_len,
        ff_dim=embedding_dim * 4
    )
    
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=epochs,
        validation_split=validation_split,
        checkpoint_dir=str(Path(model_dir) / 'checkpoints')
    )
    
    data_config = DataConfig(
        max_sequence_length=max_seq_len,
        min_sequence_length=max_seq_len // 16
    )
    
    return {
        'data_dir': Path(data_dir),
        'model_dir': Path(model_dir),
        'model_config': model_config,
        'training_config': training_config,
        'data_config': data_config,
        'has_gpu': has_gpu
    }

def optimize_performance(has_gpu: bool):
    """Apply performance optimizations"""
    print("\n⚡ APPLYING PERFORMANCE OPTIMIZATIONS")
    print("-" * 40)
    
    if has_gpu:
        print("🚀 GPU optimizations:")
        print("   ✅ Mixed precision training")
        print("   ✅ XLA compilation")
        print("   ✅ GPU memory growth")
        
        # Enable mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Enable XLA
        tf.config.optimizer.set_jit(True)
        
    else:
        print("🐌 CPU optimizations:")
        print("   ✅ Reduced model complexity")
        print("   ✅ Smaller batch sizes")
        print("   ✅ Optimized data pipeline")
        
        # CPU-specific optimizations
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)

def prepare_data(data_dir: Path, config: TrainingConfig, data_config: DataConfig) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
    """Prepare training and validation datasets with optimizations"""
    print("\n📊 PREPARING DATASETS")
    print("-" * 30)
    
    # Load MIDI files
    loader = MIDILoader(data_config)
    processor = DataProcessor(data_config)
    
    midi_files = list(data_dir.glob('**/*.mid'))
    print(f"📁 Processing {len(midi_files)} MIDI files...")
    
    all_event_sequences = []
    for i, midi_file in enumerate(midi_files):
        try:
            events = loader.load_file(midi_file)
            if events:
                all_event_sequences.append(events)
                print(f"   ✅ {midi_file.name}: {len(events)} events")
        except Exception as e:
            print(f"   ❌ {midi_file.name}: {str(e)}")
    
    if not all_event_sequences:
        raise ValueError("No valid MIDI files found or processed")
    
    print(f"✅ Successfully processed {len(all_event_sequences)} files")
    
    # Create datasets with optimization
    if config.validation_split > 0:
        split_idx = int(len(all_event_sequences) * (1 - config.validation_split))
        train_sequences = all_event_sequences[:split_idx]
        val_sequences = all_event_sequences[split_idx:]
        
        print(f"📊 Train sequences: {len(train_sequences)}")
        print(f"📊 Validation sequences: {len(val_sequences)}")
        
        train_dataset = processor.prepare_dataset(
            train_sequences,
            batch_size=config.batch_size,
            shuffle=True
        )
        val_dataset = processor.prepare_dataset(
            val_sequences,
            batch_size=config.batch_size,
            shuffle=False
        )
        return train_dataset, val_dataset
    
    train_dataset = processor.prepare_dataset(
        all_event_sequences,
        batch_size=config.batch_size,
        shuffle=True
    )
    return train_dataset, None

def main():
    """Main interactive training function"""
    print("🎵 Welcome to Melodia Interactive Training! 🎵\n")
    
    # Get configuration from user
    config = interactive_config()
    if config is None:
        print("❌ Configuration failed. Exiting.")
        return
    
    # Apply performance optimizations
    optimize_performance(config['has_gpu'])
    
    # Create directories
    config['model_dir'].mkdir(parents=True, exist_ok=True)
    setup_logging(config['model_dir'] / 'logs')
    
    # Show final configuration
    print(f"\n🎯 FINAL CONFIGURATION")
    print("-" * 30)
    print(f"📁 Data: {config['data_dir']}")
    print(f"💾 Output: {config['model_dir']}")
    print(f"🧠 Model: {config['model_config'].num_layers} layers, {config['model_config'].embedding_dim}D")
    print(f"📚 Training: {config['training_config'].max_epochs} epochs, batch size {config['training_config'].batch_size}")
    print(f"⚡ Device: {'GPU' if config['has_gpu'] else 'CPU'}")
    
    print(f"\n🚀 ESTIMATED TRAINING TIME:")
    if config['has_gpu']:
        estimated_time = config['training_config'].max_epochs * 2  # ~2 min per epoch on GPU
        print(f"   GPU: ~{estimated_time} minutes ({estimated_time/60:.1f} hours)")
    else:
        estimated_time = config['training_config'].max_epochs * 20  # ~20 min per epoch on CPU
        print(f"   CPU: ~{estimated_time} minutes ({estimated_time/60:.1f} hours)")
        print("   💡 Consider reducing epochs or getting GPU support!")
    
    # Confirm start
    confirm = input(f"\n▶️  Start training? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("❌ Training cancelled.")
        return
    
    print("\n🚀 STARTING TRAINING...")
    print("=" * 50)
    
    try:
        # Prepare data
        train_dataset, val_dataset = prepare_data(
            config['data_dir'],
            config['training_config'],
            config['data_config']
        )
        
        # Create model
        model = MelodiaModel(config['model_config'])
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=config['training_config']
        )
        
        # Train model
        history = trainer.train(
            train_dataset=train_dataset,
            validation_dataset=val_dataset
        )
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Model saved to: {config['model_dir']}")
        print(f"📈 Training history: {len(history)} epochs completed")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 