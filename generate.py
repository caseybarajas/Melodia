# melodia/scripts/generate.py

import argparse
import logging
from pathlib import Path
import json
from typing import Optional, Dict

from melodia.config import GenerationConfig, ModelConfig
from melodia.model.architecture import MelodiaModel
from melodia.generation.generator import MusicGenerator
from melodia.generation.controls import GenerationControls, MusicalStyle
from melodia.evaluation.metrics import MusicEvaluator
from melodia.utils.midi import MIDIProcessor

logger = logging.getLogger(__name__)

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate music with Melodia')
    
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save generated music')
    
    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--style', type=str, choices=[s.value for s in MusicalStyle],
                       help='Musical style')
    
    # Control parameters
    parser.add_argument('--key', type=str,
                       help='Key (e.g., C, F#, Bb)')
    parser.add_argument('--tempo', type=int,
                       help='Tempo in BPM')
    parser.add_argument('--time_signature', type=str,
                       help='Time signature (e.g., 4/4, 3/4)')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate generated samples')
    
    return parser.parse_args()

def load_model(
    model_dir: Path,
    config: Optional[Dict] = None
) -> tuple[MelodiaModel, ModelConfig]:
    """Load trained model from checkpoint"""
    # Load configuration
    if config is None:
        config_path = model_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
    
    model_config = ModelConfig(**config.get('model', {}))
    model = MelodiaModel(model_config)
    
    # Load weights
    checkpoint_dir = model_dir / 'checkpoints'
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        logger.info(f"Loaded weights from {latest_checkpoint}")
    else:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")
    
    return model, model_config

def create_generation_controls(args, config: GenerationConfig) -> GenerationControls:
    """Create generation controls from arguments"""
    controls = GenerationControls(config, style=args.style)
    
    # Set key if provided
    if args.key:
        controls.harmonic.key = args.key
    
    # Set tempo if provided
    if args.tempo:
        controls.rhythmic.tempo = args.tempo
    
    # Set time signature if provided
    if args.time_signature:
        try:
            num, den = map(int, args.time_signature.split('/'))
            controls.rhythmic.time_signature = (num, den)
        except Exception as e:
            logger.warning(f"Invalid time signature format: {str(e)}")
    
    return controls

def main():
    """Main generation function"""
    args = parse_args()
    
    # Create directories
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir / 'logs')
    
    try:
        # Load model
        model, model_config = load_model(model_dir)
        
        # Create generation config
        generation_config = GenerationConfig(
            temperature=args.temperature,
            max_length=args.max_length
        )
        
        # Create controls
        controls = create_generation_controls(args, generation_config)
        
        # Create generator
        generator = MusicGenerator(
            model=model,
            config=generation_config,
            controls=controls
        )
        
        # Create evaluator if needed
        evaluator = MusicEvaluator() if args.evaluate else None
        
        # Generate samples
        logger.info(f"Generating {args.num_samples} samples...")
        
        for i in range(args.num_samples):
            try:
                # Generate music
                events = generator.generate(
                    max_length=args.max_length
                )
                
                # Save as MIDI
                output_path = output_dir / f"sample_{i+1}.mid"
                midi_processor = MIDIProcessor()
                midi_processor.write_midi(events, output_path)
                logger.info(f"Saved sample to {output_path}")
                
                # Evaluate if requested
                if evaluator:
                    metrics = evaluator.evaluate(events)
                    metrics_path = output_dir / f"sample_{i+1}_metrics.json"
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    logger.info(f"Saved metrics to {metrics_path}")
            
            except Exception as e:
                logger.error(f"Error generating sample {i+1}: {str(e)}")
        
        logger.info("Generation completed successfully")
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")

if __name__ == '__main__':
    main()