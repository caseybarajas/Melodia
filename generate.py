#!/usr/bin/env python3
"""
Interactive Melodia Generation Script - PyTorch Edition
Run with: python generate.py
"""

import os
import sys
import logging
from pathlib import Path
import json
import torch
from typing import Optional, Dict, List

# Add melodia to path
sys.path.insert(0, str(Path(__file__).parent))

from melodia.config import GenerationConfig, ModelConfig
from melodia.model.architecture import MelodiaModel
from melodia.generation.generator import MusicGenerator
from melodia.generation.controls import GenerationControls, MusicalStyle
from melodia.evaluation.metrics import MusicEvaluator
from melodia.utils.midi import MIDIProcessor
from melodia.data.processor import EventTokenizer, DataConfig

def setup_logging(log_dir: Path):
    """Set up logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'generation.log'),
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
        print("🚀 GPU detected - generation will be faster!")
        return True
    else:
        print("❌ No GPU found - generation will be slower on CPU")
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

def get_available_models() -> List[Path]:
    """Find available trained models"""
    model_dirs = []
    
    # Check common model directories
    search_paths = [Path("models"), Path("checkpoints"), Path(".")]
    
    for search_path in search_paths:
        if search_path.exists():
            # Look for melodia_model.pt files
            for model_file in search_path.glob("**/melodia_model.pt"):
                model_dirs.append(model_file.parent)
            
            # Look for checkpoint directories
            for checkpoint_dir in search_path.glob("**/checkpoints"):
                if any(checkpoint_dir.glob("*.pt")):
                    model_dirs.append(checkpoint_dir.parent)
    
    return list(set(model_dirs))  # Remove duplicates

def select_model() -> Optional[Path]:
    """Interactive model selection"""
    print("📂 MODEL SELECTION")
    print("-" * 30)
    
    available_models = get_available_models()
    
    if not available_models:
        print("❌ No trained models found!")
        print("💡 Please train a model first using: python train.py")
        return None
    
    print(f"✅ Found {len(available_models)} trained model(s):")
    for i, model_dir in enumerate(available_models, 1):
        # Check if training info exists
        training_info_path = model_dir / 'training_info.json'
        if training_info_path.exists():
            with open(training_info_path, 'r') as f:
                info = json.load(f)
            epochs = info.get('epoch', 'Unknown')
            print(f"   {i}. {model_dir} (Epoch: {epochs})")
        else:
            print(f"   {i}. {model_dir}")
    
    if len(available_models) == 1:
        print(f"🎯 Using the only available model: {available_models[0]}")
        return available_models[0]
    
    while True:
        try:
            choice = get_user_input(f"Select model (1-{len(available_models)})", "1", int)
            if 1 <= choice <= len(available_models):
                selected = available_models[choice - 1]
                print(f"✅ Selected: {selected}")
                return selected
            else:
                print(f"❌ Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("❌ Please enter a valid number")

def interactive_config():
    """Get configuration from user interactively"""
    print("\n🎵 ============================================ 🎵")
    print("    MELODIA INTERACTIVE MUSIC GENERATION")
    print("🎵 ============================================ 🎵\n")
    
    # Model selection
    model_dir = select_model()
    if not model_dir:
        return None
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Generation parameters
    print("\n🎼 GENERATION PARAMETERS")
    print("-" * 30)
    
    num_samples = get_user_input("Number of samples to generate", "3", int)
    max_length = get_user_input("Maximum sequence length", "512", int)
    temperature = get_user_input("Temperature (0.1-2.0, higher = more creative)", "1.0", float)
    
    # Musical style
    print("\n🎨 MUSICAL STYLE")
    print("-" * 30)
    styles = [style.value for style in MusicalStyle]
    print("Available styles:")
    for i, style in enumerate(styles, 1):
        print(f"   {i}. {style.capitalize()}")
    
    style_choice = get_user_input(f"Select style (1-{len(styles)}) or press Enter for auto", "", str)
    selected_style = None
    if style_choice and style_choice.isdigit():
        choice_idx = int(style_choice) - 1
        if 0 <= choice_idx < len(styles):
            selected_style = styles[choice_idx]
            print(f"✅ Selected style: {selected_style.capitalize()}")
    
    # Musical controls
    print("\n🎹 MUSICAL CONTROLS")
    print("-" * 30)
    
    key = get_user_input("Key (e.g., C, F#, Bb) or press Enter for auto", "", str)
    tempo = get_user_input("Tempo (BPM)", "120", int)
    time_signature = get_user_input("Time signature (e.g., 4/4, 3/4)", "4/4", str)
    
    # Output configuration
    print("\n💾 OUTPUT CONFIGURATION")
    print("-" * 30)
    output_dir = get_user_input("Output directory", "outputs", str)
    evaluate = get_user_input("Evaluate generated music? (y/n)", "y", str).lower() in ['y', 'yes']
    
    return {
        'model_dir': model_dir,
        'output_dir': Path(output_dir),
        'num_samples': num_samples,
        'max_length': max_length,
        'temperature': temperature,
        'style': selected_style,
        'key': key if key else None,
        'tempo': tempo,
        'time_signature': time_signature,
        'evaluate': evaluate,
        'has_gpu': has_gpu
    }

def load_model(
    model_dir: Path,
    device: torch.device
) -> tuple[MelodiaModel, ModelConfig, EventTokenizer]:
    """Load trained model from checkpoint"""
    print(f"📥 Loading model from {model_dir}...")
    
    # Load training info which contains model config
    training_info_path = model_dir / 'training_info.json'
    if training_info_path.exists():
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
        model_config_dict = training_info.get('model_config', {})
        print(f"✅ Found training info: Epoch {training_info.get('epoch', 'Unknown')}")
    else:
        # Fallback to default config
        model_config_dict = {}
        print("⚠️  No training info found, using default config")
    
    model_config = ModelConfig(**model_config_dict)
    model = MelodiaModel(model_config)
    
    # Load model weights
    model_path = model_dir / 'melodia_model.pt'
    if model_path.exists():
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # New format with metadata
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Old format - just the state dict
            model_state_dict = checkpoint
        
        model.load_state_dict(model_state_dict)
        print(f"✅ Loaded model weights from {model_path}")
    else:
        raise ValueError(f"No model found at {model_path}")
    
    # Create tokenizer - for now use default config
    data_config = DataConfig()
    tokenizer = EventTokenizer(data_config)
    
    # Update vocab size if available
    if 'vocab_size' in training_info:
        tokenizer.vocab_size = training_info['vocab_size']
        print(f"✅ Vocabulary size: {tokenizer.vocab_size}")
    
    return model, model_config, tokenizer

def create_generation_controls(config: dict, generation_config: GenerationConfig) -> GenerationControls:
    """Create generation controls from user config"""
    controls = GenerationControls(generation_config, style=config['style'])
    
    # Set key if provided
    if config['key']:
        controls.harmonic.key = config['key']
        print(f"🎵 Set key: {config['key']}")
    
    # Set tempo
    controls.rhythmic.tempo = config['tempo']
    print(f"🎼 Set tempo: {config['tempo']} BPM")
    
    # Set time signature if provided
    if config['time_signature']:
        try:
            num, den = map(int, config['time_signature'].split('/'))
            controls.rhythmic.time_signature = (num, den)
            print(f"🎯 Set time signature: {config['time_signature']}")
        except Exception as e:
            print(f"⚠️  Invalid time signature format: {str(e)}")
    
    return controls

def estimate_generation_time(has_gpu: bool, num_samples: int, max_length: int):
    """Estimate generation time"""
    if has_gpu:
        time_per_sample = max_length / 100  # seconds
        device = "GPU"
    else:
        time_per_sample = max_length / 20   # seconds
        device = "CPU"
    
    total_seconds = num_samples * time_per_sample
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    print(f"\n🚀 ESTIMATED GENERATION TIME:")
    print(f"   {device}: ~{int(minutes)}m {int(seconds)}s for {num_samples} samples")

def main():
    """Main generation function"""
    print("🎵 Welcome to Melodia Interactive Music Generation! 🎵\n")
    
    # Get configuration
    config = interactive_config()
    if config is None:
        print("❌ Configuration failed!")
        return
    
    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(config['output_dir'] / 'logs')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Show final configuration
    print("\n🎯 FINAL CONFIGURATION")
    print("-" * 30)
    print(f"📂 Model: {config['model_dir']}")
    print(f"💾 Output: {config['output_dir']}")
    print(f"🎼 Samples: {config['num_samples']}")
    print(f"📏 Max length: {config['max_length']}")
    print(f"🌡️  Temperature: {config['temperature']}")
    if config['style']:
        print(f"🎨 Style: {config['style'].capitalize()}")
    if config['key']:
        print(f"🎵 Key: {config['key']}")
    print(f"⚡ Device: {'GPU' if config['has_gpu'] else 'CPU'}")
    
    # Estimate generation time
    estimate_generation_time(config['has_gpu'], config['num_samples'], config['max_length'])
    
    # Confirm start
    start_generation = get_user_input("\n▶️  Start generation? [Y/n]", "Y", str).lower()
    if start_generation not in ['y', 'yes', '']:
        print("❌ Generation cancelled")
        return
    
    try:
        # Load model
        model, model_config, tokenizer = load_model(config['model_dir'], device)
        model.to(device)
        
        # Create generation config
        generation_config = GenerationConfig(
            temperature=config['temperature'],
            max_length=config['max_length']
        )
        
        # Create controls
        controls = create_generation_controls(config, generation_config)
        
        # Create generator
        print("\n🧠 CREATING GENERATOR")
        print("-" * 30)
        generator = MusicGenerator(
            model=model,
            tokenizer=tokenizer,
            config=generation_config,
            device=device
        )
        print(f"✅ Generator ready on {device}")
        
        # Create evaluator if needed
        evaluator = MusicEvaluator() if config['evaluate'] else None
        if evaluator:
            print("✅ Music evaluator ready")
        
        # Generate samples
        print("\n🎼 GENERATING MUSIC...")
        print("=" * 50)
        
        for i in range(config['num_samples']):
            try:
                print(f"\n🎵 Generating sample {i+1}/{config['num_samples']}...")
                
                # Generate music
                events = generator.generate(
                    max_length=config['max_length']
                )
                
                # Save as MIDI
                output_path = config['output_dir'] / f"sample_{i+1}.mid"
                midi_processor = MIDIProcessor()
                midi_processor.write_midi(events, output_path)
                print(f"   ✅ Saved MIDI: {output_path}")
                
                # Evaluate if requested
                if evaluator:
                    try:
                        metrics = evaluator.evaluate(events)
                        metrics_path = config['output_dir'] / f"sample_{i+1}_metrics.json"
                        with open(metrics_path, 'w') as f:
                            json.dump(metrics, f, indent=2)
                        print(f"   ✅ Saved metrics: {metrics_path}")
                        
                        # Show key metrics
                        if 'avg_pitch' in metrics:
                            avg_pitch = metrics['avg_pitch']
                            print(f"   📊 Avg pitch: {avg_pitch:.1f}")
                        if 'note_density' in metrics:
                            density = metrics['note_density']
                            print(f"   📊 Note density: {density:.2f}")
                    except Exception as eval_error:
                        print(f"   ⚠️  Evaluation failed: {str(eval_error)}")
                        print("   ✅ MIDI file still saved successfully")
            
            except Exception as e:
                print(f"   ❌ Error generating sample {i+1}: {str(e)}")
        
        print("\n🎉 GENERATION COMPLETED SUCCESSFULLY!")
        print(f"📁 All files saved to: {config['output_dir']}")
        print(f"🎵 Generated {config['num_samples']} musical samples")
        
        if config['evaluate']:
            print("📊 Evaluation metrics saved alongside each sample")
        
        print("\n💡 You can now:")
        print(f"   🎹 Play the MIDI files in {config['output_dir']}")
        print("   🔄 Run generation again with different settings")
        print("   ⚙️  Adjust model parameters and re-train")
        
    except KeyboardInterrupt:
        print("\n🛑 Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()