# /melodia/config.py

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
from pathlib import Path

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 2048
    dropout_rate: float = 0.1
    max_sequence_length: int = 1024
    vocab_size: int = 512  # Will be updated based on data

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 0.0001
    max_epochs: int = 200
    warmup_steps: int = 4000
    gradient_clip_val: float = 1.0
    weight_decay: float = 0.01
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"

@dataclass
class DataConfig:
    """Data processing configuration"""
    max_sequence_length: int = 1024
    min_sequence_length: int = 64
    sampling_rate: int = 44100
    hop_length: int = 512
    n_mels: int = 128
    valid_time_signatures: List[Tuple[int, int]] = (
        (4, 4), (3, 4), (6, 8), (2, 4), (9, 8)
    )

@dataclass
class GenerationConfig:
    """Music generation configuration"""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    max_length: int = 1024
    num_return_sequences: int = 1

@dataclass
class MelodiaConfig:
    """Main configuration class"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    generation: GenerationConfig = GenerationConfig()
    
    # System paths
    project_root: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "outputs"
    log_dir: Path = project_root / "logs"
    
    def __post_init__(self):
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        Path(self.training.checkpoint_dir).mkdir(exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MelodiaConfig':
        """Create config from dictionary"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        generation_config = GenerationConfig(**config_dict.get('generation', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            generation=generation_config
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'generation': self.generation.__dict__
        }

# Default configuration instance
config = MelodiaConfig()