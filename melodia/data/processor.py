# melodia/data/processor.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import tensorflow as tf
from collections import defaultdict
from ..config import DataConfig
from .loader import MusicEvent
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class EventTokenizer:
    """Converts musical events to and from tokens"""
    
    # Special tokens
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    UNK_TOKEN = 3
    
    # Token type ranges (using prime numbers to avoid collisions)
    NOTE_START = 10
    VELOCITY_START = 127
    DURATION_START = 227
    TEMPO_START = 327
    TIME_SIG_START = 427
    KEY_SIG_START = 527
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Initialize vocabulary
        self.token_to_event = {
            self.PAD_TOKEN: '<pad>',
            self.BOS_TOKEN: '<bos>',
            self.EOS_TOKEN: '<eos>',
            self.UNK_TOKEN: '<unk>'
        }
        self.event_to_token = {v: k for k, v in self.token_to_event.items()}
        
        # Initialize mappings
        self._init_note_tokens()
        self._init_velocity_tokens()
        self._init_duration_tokens()
        self._init_tempo_tokens()
        self._init_time_sig_tokens()
        self._init_key_sig_tokens()
        
        self.vocab_size = len(self.token_to_event)
    
    def _init_note_tokens(self):
        """Initialize MIDI note tokens (0-127)"""
        for note in range(128):
            token = self.NOTE_START + note
            self.token_to_event[token] = f'NOTE_{note}'
            self.event_to_token[f'NOTE_{note}'] = token
    
    def _init_velocity_tokens(self):
        """Initialize velocity tokens (quantized to 32 levels)"""
        for vel in range(0, 128, 4):
            token = self.VELOCITY_START + (vel // 4)
            self.token_to_event[token] = f'VEL_{vel}'
            self.event_to_token[f'VEL_{vel}'] = token
    
    def _init_duration_tokens(self):
        """Initialize duration tokens (quantized logarithmically)"""
        durations = [
            0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 
            1.5, 2.0, 3.0, 4.0, 6.0, 8.0
        ]
        for i, dur in enumerate(durations):
            token = self.DURATION_START + i
            self.token_to_event[token] = f'DUR_{dur}'
            self.event_to_token[f'DUR_{dur}'] = token
    
    def _init_tempo_tokens(self):
        """Initialize tempo tokens (quantized to ranges)"""
        tempo_ranges = [
            (20, 40), (40, 60), (60, 80), (80, 100),
            (100, 120), (120, 140), (140, 160), (160, 200)
        ]
        for i, (low, high) in enumerate(tempo_ranges):
            token = self.TEMPO_START + i
            self.token_to_event[token] = f'TEMPO_{low}_{high}'
            self.event_to_token[f'TEMPO_{low}_{high}'] = token
    
    def _init_time_sig_tokens(self):
        """Initialize time signature tokens"""
        for i, (num, den) in enumerate(self.config.valid_time_signatures):
            token = self.TIME_SIG_START + i
            self.token_to_event[token] = f'TIME_{num}_{den}'
            self.event_to_token[f'TIME_{num}_{den}'] = token
    
    def _init_key_sig_tokens(self):
        """Initialize key signature tokens"""
        keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
               'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
        for i, key_name in enumerate(keys):
            for mode in ['major', 'minor']:
                token = self.KEY_SIG_START + (i * 2) + (0 if mode == 'major' else 1)
                key_str = f'{key_name}_{mode}'
                self.token_to_event[token] = f'KEY_{key_str}'
                self.event_to_token[f'KEY_{key_str}'] = token

    def save_vocabulary(self, path: Union[str, Path]):
        """Save tokenizer vocabulary to file"""
        vocab_data = {
            'token_to_event': self.token_to_event,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'PAD_TOKEN': self.PAD_TOKEN,
                'BOS_TOKEN': self.BOS_TOKEN,
                'EOS_TOKEN': self.EOS_TOKEN,
                'UNK_TOKEN': self.UNK_TOKEN
            }
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

    @classmethod
    def load_vocabulary(cls, path: Union[str, Path], config: DataConfig) -> 'EventTokenizer':
        """Load tokenizer vocabulary from file"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(config)
        tokenizer.token_to_event = {int(k): v for k, v in vocab_data['token_to_event'].items()}
        tokenizer.event_to_token = {v: int(k) for k, v in tokenizer.token_to_event.items()}
        tokenizer.vocab_size = vocab_data['vocab_size']
        return tokenizer

class DataProcessor:
    """Handles data processing, augmentation, and batching for training"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = EventTokenizer(config)
    
    def process_events(
        self,
        events_list: List[List[MusicEvent]],
        augment: bool = True
    ) -> List[List[MusicEvent]]:
        """Process and optionally augment event sequences"""
        processed_events = []
        
        for events in events_list:
            # Basic processing
            processed = self._process_single_sequence(events)
            processed_events.append(processed)
            
            if augment:
                # Transposition augmentation
                for semitones in [-2, -1, 1, 2]:
                    transposed = self._transpose_sequence(processed, semitones)
                    processed_events.append(transposed)
                
                # Tempo augmentation
                for tempo_factor in [0.9, 1.1]:
                    tempo_varied = self._vary_tempo(processed, tempo_factor)
                    processed_events.append(tempo_varied)
        
        return processed_events

    def _process_single_sequence(self, events: List[MusicEvent]) -> List[MusicEvent]:
        """Process a single sequence of events"""
        # Sort by time
        events = sorted(events, key=lambda x: x.time)
        
        # Quantize times and durations
        quantized = []
        for event in events:
            event_copy = MusicEvent(
                type=event.type,
                time=self._quantize_time(event.time),
                duration=self._quantize_time(event.duration) if event.duration else None,
                pitch=event.pitch,
                velocity=event.velocity if event.velocity else None,
                tempo=event.tempo if event.tempo else None,
                numerator=event.numerator if event.numerator else None,
                denominator=event.denominator if event.denominator else None,
                key=event.key if event.key else None
            )
            quantized.append(event_copy)
        
        return quantized

    def _quantize_time(self, time: Optional[float]) -> Optional[float]:
        """Quantize time values to the nearest subdivision"""
        if time is None:
            return None
        
        subdivision = 0.125  # 32nd note
        return round(time / subdivision) * subdivision

    def prepare_dataset(
        self,
        events_list: List[List[MusicEvent]],
        batch_size: int = 32,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """Prepare a TensorFlow dataset for training"""
        # Convert events to sequences
        sequences = []
        for events in events_list:
            tokens = self.tokenizer.events_to_tokens(events)
            if self.config.min_sequence_length <= len(tokens) <= self.config.max_sequence_length:
                sequences.append(tokens)
        
        # Create input/target pairs
        inputs = []
        targets = []
        
        for sequence in sequences:
            # Create sliding windows
            for i in range(0, len(sequence) - self.config.min_sequence_length):
                end = min(i + self.config.max_sequence_length, len(sequence))
                
                input_seq = sequence[i:end-1]
                target_seq = sequence[i+1:end]
                
                # Pad sequences if necessary
                if len(input_seq) < self.config.max_sequence_length:
                    pad_length = self.config.max_sequence_length - len(input_seq)
                    input_seq = input_seq + [self.tokenizer.PAD_TOKEN] * pad_length
                    target_seq = target_seq + [self.tokenizer.PAD_TOKEN] * pad_length
                
                inputs.append(input_seq)
                targets.append(target_seq)
        
        # Convert to numpy arrays
        inputs = np.array(inputs, dtype=np.int32)
        targets = np.array(targets, dtype=np.int32)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def _transpose_sequence(
        self,
        events: List[MusicEvent],
        semitones: int
    ) -> List[MusicEvent]:
        """Transpose a sequence by a number of semitones"""
        transposed = []
        
        for event in events:
            event_copy = MusicEvent(
                type=event.type,
                time=event.time,
                duration=event.duration,
                pitch=None,
                velocity=event.velocity,
                tempo=event.tempo,
                numerator=event.numerator,
                denominator=event.denominator,
                key=event.key
            )
            
            # Transpose pitch if present
            if event.type == 'note':
                if isinstance(event.pitch, int):
                    new_pitch = event.pitch + semitones
                    if 0 <= new_pitch <= 127:
                        event_copy.pitch = new_pitch
                        transposed.append(event_copy)
                elif isinstance(event.pitch, list):
                    new_pitches = [p + semitones for p in event.pitch]
                    if all(0 <= p <= 127 for p in new_pitches):
                        event_copy.pitch = new_pitches
                        transposed.append(event_copy)
            else:
                transposed.append(event_copy)
        
        return transposed

    def _vary_tempo(
        self,
        events: List[MusicEvent],
        tempo_factor: float
    ) -> List[MusicEvent]:
        """Vary the tempo of a sequence"""
        scaled = []
        current_time = 0.0
        
        for event in events:
            event_copy = MusicEvent(
                type=event.type,
                time=current_time,
                duration=event.duration / tempo_factor if event.duration else None,
                pitch=event.pitch,
                velocity=event.velocity,
                tempo=event.tempo * tempo_factor if event.tempo else None,
                numerator=event.numerator,
                denominator=event.denominator,
                key=event.key
            )
            
            scaled.append(event_copy)
            if event.duration:
                current_time += event.duration / tempo_factor
        
        return scaled

    def save_processor(self, path: Union[str, Path]):
        """Save the data processor state"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer vocabulary
        self.tokenizer.save_vocabulary(save_path / 'vocabulary.json')
        
        # Save config
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def load_processor(cls, path: Union[str, Path]) -> 'DataProcessor':
        """Load a saved data processor"""
        load_path = Path(path)
        
        # Load config
        with open(load_path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        config = DataConfig(**config_dict)
        
        # Create processor and load tokenizer
        processor = cls(config)
        processor.tokenizer = EventTokenizer.load_vocabulary(
            load_path / 'vocabulary.json',
            config
        )
        
        return processor