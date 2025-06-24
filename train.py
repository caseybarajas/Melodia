#!/usr/bin/env python3
"""
Improved Melodia Training Script - PyTorch Edition
Run with: python train.py

🎵 WHAT THIS SCRIPT DOES:
This script trains an AI model to generate music by learning patterns from MIDI files.
Think of it like teaching a computer to compose music by showing it many examples!

📚 LEARNING PROCESS:
1. Load your MIDI files (the "sheet music" examples)
2. Convert them into tokens (like words in a sentence, but for music)
3. Create a neural network (the "brain" that learns patterns)
4. Train the model (show it examples over and over until it learns)
5. Save the trained model (so you can generate music later!)
"""

import os
import sys
import logging
from pathlib import Path
import torch
import torch.cuda
from typing import Optional, Tuple
import json

# Add melodia to path
sys.path.insert(0, str(Path(__file__).parent))

from melodia.config import TrainingConfig, ModelConfig, DataConfig
from melodia.data.loader import MIDILoader
from melodia.data.processor import DataProcessor, MelodiaDataset
from melodia.model.architecture import MelodiaModel
from melodia.training.trainer import Trainer
from torch.utils.data import DataLoader

def setup_logging(log_dir: Path):
    """Set up logging configuration
    
    🤔 WHAT THIS DOES:
    Creates log files to track what happens during training.
    Like keeping a diary of the training process!
    """
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
    """Check and configure GPU
    
    🤔 WHAT THIS DOES:
    Checks if you have a graphics card (GPU) that can speed up training.
    GPUs are like having 1000+ workers instead of 1 for math calculations!
    
    🎯 WHY IT MATTERS:
    - With GPU: Training might take 30 minutes
    - Without GPU: Same training might take 10+ hours
    """
    print("🔍 Checking GPU availability...")
    print("   (Graphics cards make training MUCH faster!)")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            # Get memory info
            if torch.cuda.is_available():
                memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f}GB VRAM)")
        
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print("🚀 GPU detected - you can use larger models!")
        return True, memory_gb
    else:
        print("❌ No GPU found - training will be MUCH slower on CPU")
        print("💡 To install GPU support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False, 0

def get_memory_preset(vram_gb: float):
    """Get recommended settings based on GPU memory
    
    🤔 WHAT THIS DOES:
    Chooses model size based on your graphics card's memory.
    Like choosing a car size based on your garage space!
    
    📊 MEMORY PRESETS:
    - 4GB or less: Small model (won't run out of memory)
    - 6-8GB: Medium model (good balance)
    - 12GB+: Large model (best quality)
    """
    if vram_gb >= 12:
        return {
            'name': 'Large (12GB+)',
            'embedding_dim': 512,
            'num_layers': 12,
            'num_heads': 16,
            'max_seq_len': 1024,
            'batch_size': 24,
            'description': 'Best quality, slowest training'
        }
    elif vram_gb >= 6:
        return {
            'name': 'Medium (6-8GB)',
            'embedding_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'max_seq_len': 512,
            'batch_size': 16,
            'description': 'Good balance of quality and speed'
        }
    else:
        return {
            'name': 'Small (4GB or less)',
            'embedding_dim': 128,
            'num_layers': 4,
            'num_heads': 4,
            'max_seq_len': 256,
            'batch_size': 8,
            'description': 'Fits in smaller memory, basic quality'
        }

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
    """Get configuration from user interactively
    
    🤔 WHAT THIS DOES:
    Asks you questions to set up the training process.
    Like filling out a form before starting a class!
    """
    print("\n🎵 ============================================ 🎵")
    print("    MELODIA IMPROVED TRAINING SETUP")
    print("🎵 ============================================ 🎵\n")
    
    # Data configuration
    print("📁 DATA CONFIGURATION")
    print("-" * 30)
    print("🤔 WHAT THIS DOES: Tell us where your MIDI files are")
    print("   (These are the 'examples' the AI will learn from)")
    
    data_dir = get_user_input("Training data directory", "data", str)
    if not Path(data_dir).exists():
        print(f"❌ Directory {data_dir} doesn't exist!")
        return None
    
    # Check available MIDI files
    midi_files = list(Path(data_dir).glob("**/*.mid"))
    print(f"✅ Found {len(midi_files)} MIDI files")
    print(f"   (The AI will learn musical patterns from these {len(midi_files)} songs)")
    
    if len(midi_files) == 0:
        print("❌ No MIDI files found!")
        return None
    
    # Model configuration
    print("\n🧠 MODEL CONFIGURATION")
    print("-" * 30)
    print("🤔 WHAT THIS DOES: Design the 'brain' of your AI")
    print("   (Bigger brain = better music, but needs more memory)")
    
    # Detect if GPU is available for model size recommendations
    has_gpu, vram_gb = check_gpu()
    
    # Memory optimization options
    print(f"\n💾 MEMORY OPTIMIZATION")
    print("-" * 30)
    
    if has_gpu:
        preset = get_memory_preset(vram_gb)
        print(f"🎯 RECOMMENDED FOR YOUR GPU ({vram_gb:.1f}GB):")
        print(f"   Preset: {preset['name']}")
        print(f"   {preset['description']}")
        
        print(f"\n📊 Available Memory Presets:")
        print(f"   1. Small (4GB):   Fast, basic quality")
        print(f"   2. Medium (6-8GB): Balanced (RECOMMENDED)")
        print(f"   3. Large (12GB+):  Slow, best quality")
        print(f"   4. Custom:         Set your own values")
        
        choice = get_user_input("Choose preset (1-4)", "2", int)
        
        if choice == 1:
            preset = get_memory_preset(4)
        elif choice == 2:
            preset = get_memory_preset(6)
        elif choice == 3:
            preset = get_memory_preset(12)
        elif choice == 4:
            preset = None
        else:
            print("Invalid choice, using Medium preset")
            preset = get_memory_preset(6)
        
        if preset:
            print(f"\n✅ Using {preset['name']} preset:")
            embedding_dim = preset['embedding_dim']
            num_layers = preset['num_layers'] 
            num_heads = preset['num_heads']
            max_seq_len = preset['max_seq_len']
            batch_size = preset['batch_size']
            
            print(f"   🧠 Embedding size: {embedding_dim} (how detailed each note is)")
            print(f"   🔗 Layers: {num_layers} (how deep the AI thinks)")
            print(f"   👁️  Attention heads: {num_heads} (how many things it focuses on)")
            print(f"   📏 Max sequence: {max_seq_len} (longest musical phrase)")
            print(f"   📦 Batch size: {batch_size} (how many examples at once)")
        else:
            print(f"\n🛠️  CUSTOM CONFIGURATION:")
            print("   💡 TIP: Start small if unsure!")
            embedding_dim = get_user_input("Embedding dimension (128-512)", "256", int)
            num_layers = get_user_input("Number of layers (4-12)", "6", int)
            num_heads = get_user_input("Number of attention heads (4-16)", "8", int)
            max_seq_len = get_user_input("Max sequence length (256-1024)", "512", int)
            batch_size = get_user_input("Batch size (8-32)", "16", int)
    else:
        print("🐌 CPU ONLY - Using small model for reasonable speed")
        print("   (Without a GPU, we need to keep things simple)")
        embedding_dim = 128
        num_layers = 4
        num_heads = 4
        max_seq_len = 256
        batch_size = 4
    
    # Training configuration
    print(f"\n📚 TRAINING CONFIGURATION")
    print("-" * 30)
    print("🤔 WHAT THIS DOES: Set how long and how intensively to train")
    print("   (Like setting how many hours to practice piano)")
    
    if has_gpu:
        default_epochs = "50"
        print(f"   💡 With GPU: Can train for more epochs (rounds of learning)")
    else:
        default_epochs = "20" 
        print(f"   💡 With CPU: Keep epochs lower for reasonable time")
    
    epochs = get_user_input(f"Number of epochs", default_epochs, int)
    print(f"   📖 Each epoch = seeing all your MIDI files once")
    print(f"   🔄 {epochs} epochs = going through all files {epochs} times")
    
    learning_rate = get_user_input("Learning rate (0.0001-0.01)", "0.001", float)
    print(f"   🎯 Learning rate: How big steps the AI takes while learning")
    print(f"   📊 0.001 = careful steps (recommended)")
    
    validation_split = get_user_input("Validation split (0.0-1.0)", "0.1", float)
    print(f"   🧪 Validation: Testing on {validation_split*100:.0f}% of data")
    print(f"   📈 This helps us see if the AI is actually learning")
    
    # Output configuration
    print(f"\n💾 OUTPUT CONFIGURATION")
    print("-" * 30)
    print("🤔 WHAT THIS DOES: Where to save your trained AI model")
    
    model_dir = get_user_input("Model output directory", "models", str)
    
    # Create configs with proper vocabulary size handling
    data_config = DataConfig(
        max_sequence_length=max_seq_len,
        min_sequence_length=max_seq_len // 8
    )
    
    # Create a temporary processor to get vocab size
    print(f"\n📚 ANALYZING YOUR MUSIC...")
    print("   (Converting MIDI files to understand the musical vocabulary)")
    temp_processor = DataProcessor(data_config)
    vocab_size = temp_processor.tokenizer.vocab_size
    print(f"📚 Vocabulary size: {vocab_size}")
    print(f"   (Your music uses {vocab_size} different musical 'words')")
    
    model_config = ModelConfig(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_sequence_length=max_seq_len,
        ff_dim=embedding_dim * 4,
        vocab_size=vocab_size  # Set correct vocab size
    )
    
    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=epochs,
        validation_split=validation_split,
        checkpoint_dir=str(Path(model_dir) / 'checkpoints')
    )
    
    return {
        'data_dir': Path(data_dir),
        'model_dir': Path(model_dir),
        'model_config': model_config,
        'training_config': training_config,
        'data_config': data_config,
        'has_gpu': has_gpu,
        'vram_gb': vram_gb
    }

def optimize_performance(has_gpu: bool, vram_gb: float):
    """Apply performance optimizations
    
    🤔 WHAT THIS DOES:
    Configures your computer to run as efficiently as possible.
    Like tuning a race car before a race!
    """
    print(f"\n⚡ APPLYING PERFORMANCE OPTIMIZATIONS")
    print("-" * 40)
    
    if has_gpu:
        print(f"🚀 GPU optimizations for {vram_gb:.1f}GB GPU:")
        print(f"   ✅ CUDA acceleration (using GPU instead of CPU)")
        print(f"   ✅ cuDNN benchmark (finding fastest algorithms)")
        
        # Memory management based on GPU size
        if vram_gb <= 8:
            print(f"   ✅ Conservative memory usage (8GB or less)")
            memory_fraction = 0.85  # Use 85% to leave room for system
        else:
            print(f"   ✅ Optimized memory usage (>8GB)")
            memory_fraction = 0.9
            
        print(f"   💾 Using {memory_fraction*100:.0f}% of GPU memory")
        
        # Enable cuDNN benchmark for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory allocation strategy
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Memory optimization settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
    else:
        print(f"🐌 CPU optimizations:")
        print(f"   ✅ Reduced model complexity")
        print(f"   ✅ Smaller batch sizes") 
        print(f"   ✅ Optimized threading")
        
        # CPU-specific optimizations
        torch.set_num_threads(min(8, torch.get_num_threads()))

def estimate_memory_usage(model_config: ModelConfig, batch_size: int):
    """Estimate memory usage
    
    🤔 WHAT THIS DOES:
    Calculates how much graphics card memory the model will need.
    Like measuring if furniture will fit in your room!
    """
    # Rough estimation of memory usage
    vocab_size = model_config.vocab_size
    embedding_dim = model_config.embedding_dim
    seq_len = model_config.max_sequence_length
    num_layers = model_config.num_layers
    
    # Model parameters memory (in bytes)
    embedding_params = vocab_size * embedding_dim * 4  # 4 bytes per float32
    attention_params = num_layers * embedding_dim * embedding_dim * 4 * 4  # Q,K,V,O projections
    ffn_params = num_layers * embedding_dim * (embedding_dim * 4) * 2 * 4  # Two linear layers
    
    model_memory_mb = (embedding_params + attention_params + ffn_params) / (1024 * 1024)
    
    # Activation memory (depends on batch size and sequence length)
    activation_memory_mb = batch_size * seq_len * embedding_dim * num_layers * 4 / (1024 * 1024)
    
    # Add some overhead (optimizer states, gradients, etc.)
    total_memory_mb = (model_memory_mb + activation_memory_mb) * 3  # 3x factor for safety
    
    return total_memory_mb

def prepare_data(data_dir: Path, config: TrainingConfig, data_config: DataConfig) -> Tuple[DataLoader, Optional[DataLoader], DataProcessor]:
    """Prepare training and validation datasets with improved handling
    
    🤔 WHAT THIS DOES:
    Converts your MIDI files into a format the AI can understand.
    Like translating sheet music into a language the computer knows!
    
    📚 PROCESS:
    1. Load each MIDI file (your songs)
    2. Convert notes/timing to tokens (like words)
    3. Split into training vs testing data
    4. Package into batches for efficient learning
    """
    print("\n📊 PREPARING DATASETS")
    print("-" * 30)
    print("🤔 Converting MIDI files to AI-readable format...")
    
    # Load MIDI files
    print("📁 Processing MIDI files...")
    loader = MIDILoader(data_config)
    
    events_list = []
    for midi_file in data_dir.glob("**/*.mid"):
        try:
            events = loader.load_midi(midi_file)
            if events and len(events) > 5:  # Only include files with sufficient content
                events_list.append(events)
                print(f"   ✅ {midi_file.name}: {len(events)} events")
        except Exception as e:
            print(f"   ⚠️  {midi_file.name}: {str(e)}")
    
    if not events_list:
        raise ValueError("No MIDI files could be loaded")
    
    print(f"✅ Successfully processed {len(events_list)} files")
    
    # Process events
    data_processor = DataProcessor(data_config)
    
    # Update vocab size in data_processor to match the one we calculated
    print(f"📚 Final vocabulary size: {data_processor.tokenizer.vocab_size}")
    
    processed_events = data_processor.process_events(events_list, augment=True)
    
    if not processed_events:
        raise ValueError("No sequences generated after processing")
    
    # Create dataset
    dataloader = data_processor.prepare_dataset(
        processed_events,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Get all data for splitting
    all_data = list(dataloader.dataset)
    
    if len(all_data) == 0:
        raise ValueError("No training data generated")
    
    # Split into training and validation
    dataset_size = len(all_data)
    train_size = int(dataset_size * (1 - config.validation_split))
    
    print(f"📊 Total examples: {dataset_size}")
    print(f"📊 Training examples: {train_size}")
    print(f"📊 Validation examples: {dataset_size - train_size}")
    
    # Create train dataset
    train_data = all_data[:train_size]
    train_inputs = torch.stack([item[0] for item in train_data])
    train_targets = torch.stack([item[1] for item in train_data])
    train_dataset = MelodiaDataset(train_inputs.numpy(), train_targets.numpy())
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    # Create validation dataset if we have enough data
    val_dataloader = None
    if dataset_size - train_size > 0:
        val_data = all_data[train_size:]
        val_inputs = torch.stack([item[0] for item in val_data])
        val_targets = torch.stack([item[1] for item in val_data])
        val_dataset = MelodiaDataset(val_inputs.numpy(), val_targets.numpy())
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_dataloader, val_dataloader, data_processor

def estimate_training_time(has_gpu: bool, epochs: int, num_batches: int, vram_gb: float = 0):
    """Estimate training time
    
    🤔 WHAT THIS DOES:
    Predicts how long training will take.
    Like estimating how long a road trip will be!
    """
    if has_gpu:
        if vram_gb >= 12:
            time_per_batch = 0.3  # High-end GPU
        elif vram_gb >= 8:
            time_per_batch = 0.5  # Mid-range GPU  
        else:
            time_per_batch = 0.8  # Entry-level GPU
        device = f"GPU ({vram_gb:.1f}GB)"
    else:
        time_per_batch = 3.0  # seconds
        device = "CPU"
    
    total_seconds = epochs * num_batches * time_per_batch
    total_minutes = total_seconds / 60
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    print(f"\n🚀 ESTIMATED TRAINING TIME:")
    print(f"   {device}: ~{total_minutes:.1f} minutes ({hours:.1f} hours)")
    print(f"   ⏱️  Per epoch: ~{(total_minutes/epochs):.1f} minutes")
    
    if not has_gpu and total_minutes > 120:
        print("   💡 Consider reducing epochs or getting GPU support!")
    elif total_minutes > 180:  # 3 hours
        print("   ⚠️  This will take a while - consider reducing model size!")

def main():
    """Main training function
    
    🤔 WHAT THIS DOES:
    This is the main control center that coordinates everything.
    Like a conductor directing an orchestra!
    """
    print("🎵 Welcome to Improved Melodia Training! 🎵\n")
    print("🎯 This will train an AI to compose music like your MIDI files!")
    
    # Get configuration
    config = interactive_config()
    if config is None:
        print("❌ Configuration failed!")
        return
    
    # Create output directory
    config['model_dir'].mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(config['model_dir'] / 'logs')
    
    # Apply optimizations
    optimize_performance(config['has_gpu'], config['vram_gb'])
    
    # Estimate memory usage
    estimated_memory = estimate_memory_usage(config['model_config'], config['training_config'].batch_size)
    
    # Show final configuration
    print("\n🎯 FINAL CONFIGURATION")
    print("-" * 30)
    print(f"📁 Data: {config['data_dir']}")
    print(f"💾 Output: {config['model_dir']}")
    print(f"🧠 Model: {config['model_config'].num_layers} layers, {config['model_config'].embedding_dim}D")
    print(f"📚 Vocab size: {config['model_config'].vocab_size}")
    print(f"📊 Training: {config['training_config'].max_epochs} epochs, batch size {config['training_config'].batch_size}")
    print(f"⚡ Device: {'GPU' if config['has_gpu'] else 'CPU'}")
    
    # Memory warning
    if config['has_gpu']:
        available_memory = config['vram_gb'] * 1024  # Convert to MB
        print(f"💾 Estimated memory usage: {estimated_memory:.0f}MB")
        print(f"💾 Available GPU memory: {available_memory:.0f}MB")
        
        if estimated_memory > available_memory * 0.9:
            print("⚠️  WARNING: Model might be too large for your GPU!")
            print("   💡 Consider using a smaller preset or reducing batch size")
            continue_anyway = get_user_input("Continue anyway? [y/N]", "N", str).lower()
            if continue_anyway not in ['y', 'yes']:
                print("❌ Training cancelled - try a smaller model")
                return
    
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
        
        # Estimate training time
        num_batches = len(train_dataloader)
        estimate_training_time(config['has_gpu'], config['training_config'].max_epochs, num_batches, config['vram_gb'])
        
        print(f"\n📊 Training batches per epoch: {num_batches}")
        if val_dataloader:
            print(f"📊 Validation batches per epoch: {len(val_dataloader)}")
        
        # Create trainer
        print("\n🧠 CREATING TRAINER")
        print("-" * 30)
        print("🤔 Building the AI 'brain' that will learn to compose music...")
        
        trainer = Trainer(
            model_config=config['model_config'],
            training_config=config['training_config'],
            data_processor=data_processor,
            model_dir=str(config['model_dir'])
        )
        print(f"✅ Trainer created successfully")
        
        model_params = sum(p.numel() for p in trainer.pytorch_trainer.model.parameters())
        print(f"✅ Model parameters: {model_params:,}")
        print(f"   (That's {model_params:,} individual 'weights' the AI will adjust!)")
        
        # Train model
        print("\n🚀 STARTING TRAINING...")
        print("=" * 50)
        print("🎵 The AI will now learn patterns from your music!")
        print("💡 Watch the loss numbers - lower = better learning")
        
        history = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Model saved to: {config['model_dir']}")
        if history and 'train_loss' in history:
            print(f"📈 Final training loss: {history['train_loss'][-1]:.4f}")
            if 'val_loss' in history and history['val_loss']:
                print(f"📈 Final validation loss: {history['val_loss'][-1]:.4f}")
        
        print("\n💡 Next steps:")
        print(f"   🎵 Generate music with: python generate.py")
        print(f"   📊 Check training logs in: {config['model_dir']}/logs/")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 