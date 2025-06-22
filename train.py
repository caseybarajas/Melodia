#!/usr/bin/env python3
"""
Interactive Melodia Training Script - PyTorch Edition
Run with: python train.py
"""

import os
import sys
import logging
from pathlib import Path
import torch
import torch.cuda
from typing import Optional, Tuple

# Add melodia to path
sys.path.insert(0, str(Path(__file__).parent))

from melodia.config import TrainingConfig, ModelConfig, DataConfig
from melodia.data.loader import MIDILoader
from melodia.data.processor import DataProcessor, MelodiaDataset
from melodia.model.architecture import MelodiaModel
from melodia.training.trainer import Trainer
from torch.utils.data import DataLoader

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
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")
        
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print("🚀 GPU detected - you can use larger models!")
        return True
    else:
        print("❌ No GPU found - training will be MUCH slower on CPU")
        print("💡 To install GPU support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
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
        print("   ✅ CUDA acceleration")
        print("   ✅ cuDNN benchmark")
        print("   ✅ GPU memory optimization")
        
        # Enable cuDNN benchmark for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    else:
        print("🐌 CPU optimizations:")
        print("   ✅ Reduced model complexity")
        print("   ✅ Smaller batch sizes") 
        print("   ✅ Optimized threading")
        
        # CPU-specific optimizations
        torch.set_num_threads(torch.get_num_threads())

def prepare_data(data_dir: Path, config: TrainingConfig, data_config: DataConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Prepare training and validation datasets with optimizations"""
    print("\n📊 PREPARING DATASETS")
    print("-" * 30)
    
    # Load MIDI files
    print("📁 Processing MIDI files...")
    loader = MIDILoader(data_config)
    
    events_list = []
    for midi_file in data_dir.glob("**/*.mid"):
        try:
            events = loader.load_midi(midi_file)
            if events:
                events_list.append(events)
                print(f"   ✅ {midi_file.name}: {len(events)} events")
        except Exception as e:
            print(f"   ❌ {midi_file.name}: {str(e)}")
    
    if not events_list:
        raise ValueError("No MIDI files could be loaded")
    
    print(f"✅ Successfully processed {len(events_list)} files")
    
    # Process events
    data_processor = DataProcessor(data_config)
    processed_events = data_processor.process_events(events_list, augment=True)
    
    # Create dataset
    full_dataloader = data_processor.prepare_dataset(
        processed_events,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Convert to list to enable splitting
    all_data = [(inputs, targets) for inputs, targets in full_dataloader.dataset]
    
    # Split into training and validation
    dataset_size = len(all_data)
    train_size = int(dataset_size * (1 - config.validation_split))
    val_size = dataset_size - train_size
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:] if val_size > 0 else None
    
    # Create separate datasets
    train_dataset = MelodiaDataset(
        torch.stack([item[0] for item in train_data]).numpy(),
        torch.stack([item[1] for item in train_data]).numpy()
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    val_dataloader = None
    if val_data:
        val_dataset = MelodiaDataset(
            torch.stack([item[0] for item in val_data]).numpy(),
            torch.stack([item[1] for item in val_data]).numpy()
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    
    print(f"📊 Train sequences: {train_size}")
    if val_dataloader:
        print(f"📊 Validation sequences: {val_size}")
    
    return train_dataloader, val_dataloader, data_processor

def estimate_training_time(has_gpu: bool, epochs: int):
    """Estimate training time"""
    if has_gpu:
        time_per_epoch = 2  # minutes
        device = "GPU"
    else:
        time_per_epoch = 20  # minutes
        device = "CPU"
    
    total_minutes = epochs * time_per_epoch
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    print(f"\n🚀 ESTIMATED TRAINING TIME:")
    print(f"   {device}: ~{total_minutes} minutes ({hours:.1f} hours)")
    if not has_gpu:
        print("   💡 Consider reducing epochs or getting GPU support!")

def main():
    """Main training function"""
    print("🎵 Welcome to Melodia Interactive Training! 🎵\n")
    
    # Get configuration
    config = interactive_config()
    if config is None:
        print("❌ Configuration failed!")
        return
    
    # Set up logging
    setup_logging(config['model_dir'] / 'logs')
    
    # Apply optimizations
    optimize_performance(config['has_gpu'])
    
    # Show final configuration
    print("\n🎯 FINAL CONFIGURATION")
    print("-" * 30)
    print(f"📁 Data: {config['data_dir']}")
    print(f"💾 Output: {config['model_dir']}")
    print(f"🧠 Model: {config['model_config'].num_layers} layers, {config['model_config'].embedding_dim}D")
    print(f"📚 Training: {config['training_config'].max_epochs} epochs, batch size {config['training_config'].batch_size}")
    print(f"⚡ Device: {'GPU' if config['has_gpu'] else 'CPU'}")
    
    # Estimate training time
    estimate_training_time(config['has_gpu'], config['training_config'].max_epochs)
    
    # Confirm start
    start_training = get_user_input("\n▶️  Start training? [Y/n]", "Y", str).lower()
    if start_training not in ['y', 'yes', '']:
        print("❌ Training cancelled")
        return
    
    try:
        # Prepare data
        train_dataloader, val_dataloader, data_processor = prepare_data(
            config['data_dir'],
            config['training_config'],
            config['data_config']
        )
        
        # Update vocab size in model config
        config['model_config'].vocab_size = data_processor.tokenizer.vocab_size
        
        # Create trainer
        print("\n🧠 CREATING MODEL")
        print("-" * 30)
        trainer = Trainer(
            model_config=config['model_config'],
            training_config=config['training_config'],
            data_processor=data_processor,
            model_dir=str(config['model_dir'])
        )
        print(f"✅ Model created with {config['model_config'].embedding_dim}D embeddings, {config['model_config'].num_layers} layers")
        print(f"✅ Using device: {trainer.pytorch_trainer.get_device_info()}")
        
        # Train model
        print("\n🚀 STARTING TRAINING...")
        print("=" * 50)
        
        history = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Model saved to: {config['model_dir']}")
        print(f"📈 Training history: {len(history['train_loss'])} epochs completed")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 