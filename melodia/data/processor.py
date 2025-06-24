# melodia/data/processor.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from ..config import DataConfig
from .loader import MusicEvent
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class MelodiaDataset(Dataset):
    """PyTorch Dataset for music sequences"""
    
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.from_numpy(inputs).long()
        self.targets = torch.from_numpy(targets).long()
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class EventTokenizer:
    """Enhanced tokenizer with better musical representation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build a comprehensive musical vocabulary"""
        vocab_id = 0
        
        # Special tokens
        special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        for token in special_tokens:
            self.vocab[token] = vocab_id
            self.reverse_vocab[vocab_id] = token
            vocab_id += 1
        
        # Note events (pitch + velocity + duration as single tokens for better context)
        for pitch in range(128):
            for vel_bucket in range(8):  # 8 velocity buckets (0-15, 16-31, ..., 112-127)
                for dur_bucket in range(12):  # 12 duration buckets
                    token = f'NOTE_{pitch}_{vel_bucket}_{dur_bucket}'
                    self.vocab[token] = vocab_id
                    self.reverse_vocab[vocab_id] = token
                    vocab_id += 1
        
        # Rest/silence tokens with duration
        for dur_bucket in range(12):
            token = f'REST_{dur_bucket}'
            self.vocab[token] = vocab_id
            self.reverse_vocab[vocab_id] = token
            vocab_id += 1
        
        # Chord tokens (root + type)
        chord_types = ['maj', 'min', 'dim', 'aug', '7', 'maj7', 'min7', 'dim7']
        for root in range(12):
            for chord_type in chord_types:
                token = f'CHORD_{root}_{chord_type}'
                self.vocab[token] = vocab_id
                self.reverse_vocab[vocab_id] = token
                vocab_id += 1
        
        # Time signature tokens
        for num, den in [(4, 4), (3, 4), (2, 4), (6, 8), (9, 8), (12, 8)]:
            token = f'TIME_SIG_{num}_{den}'
            self.vocab[token] = vocab_id
            self.reverse_vocab[vocab_id] = token
            vocab_id += 1
        
        # Key signature tokens
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for key in keys:
            for mode in ['maj', 'min']:
                token = f'KEY_{key}_{mode}'
                self.vocab[token] = vocab_id
                self.reverse_vocab[vocab_id] = token
                vocab_id += 1
        
        # Tempo tokens (in ranges)
        tempo_ranges = ['slow', 'medium', 'fast', 'very_fast']
        for tempo in tempo_ranges:
            token = f'TEMPO_{tempo}'
            self.vocab[token] = vocab_id
            self.reverse_vocab[vocab_id] = token
            vocab_id += 1
        
        # Musical structure tokens
        structure_tokens = ['PHRASE_START', 'PHRASE_END', 'SECTION_START', 'SECTION_END']
        for token in structure_tokens:
            self.vocab[token] = vocab_id
            self.reverse_vocab[vocab_id] = token
            vocab_id += 1
        
        self.vocab_size = len(self.vocab)
        
        # Create convenience mappings
        self.pad_token = self.vocab['<PAD>']
        self.bos_token = self.vocab['<BOS>']
        self.eos_token = self.vocab['<EOS>']
        self.unk_token = self.vocab['<UNK>']
        
        # Compatibility with generator (uppercase attributes)
        self.PAD_TOKEN = self.pad_token
        self.BOS_TOKEN = self.bos_token
        self.EOS_TOKEN = self.eos_token
        self.UNK_TOKEN = self.unk_token
    
    def _get_velocity_bucket(self, velocity: int) -> int:
        """Convert velocity to bucket (0-7)"""
        return min(7, max(0, velocity // 16))
    
    def _get_duration_bucket(self, duration: float) -> int:
        """Convert duration to bucket (0-11)"""
        # Duration buckets: 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0+
        duration_thresholds = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
        for i, threshold in enumerate(duration_thresholds):
            if duration <= threshold:
                return i
        return len(duration_thresholds) - 1
    
    def _detect_chord_type(self, pitches: List[int]) -> str:
        """Detect chord type from pitches"""
        if len(pitches) < 2:
            return 'maj'  # Default
        
        # Sort and get intervals
        sorted_pitches = sorted(pitches)
        intervals = [(sorted_pitches[i] - sorted_pitches[0]) % 12 for i in range(len(sorted_pitches))]
        intervals = sorted(set(intervals))
        
        # Basic chord detection
        if intervals == [0, 3, 6]:
            return 'dim'
        elif intervals == [0, 4, 8]:
            return 'aug'
        elif 3 in intervals and 7 in intervals:
            return 'min'
        elif 4 in intervals and 7 in intervals:
            return 'maj'
        elif 10 in intervals:
            return '7'
        else:
            return 'maj'  # Default
    
    def encode_events(self, events: List[MusicEvent]) -> List[int]:
        """Convert musical events to tokens with improved structure"""
        tokens = [self.bos_token]
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x.time)
        
        # Group events by time to handle simultaneity
        time_groups = defaultdict(list)
        for event in sorted_events:
            quantized_time = round(event.time * 8) / 8  # Quantize to 32nd notes
            time_groups[quantized_time].append(event)
        
        # Process events in temporal order
        last_time = 0.0
        phrase_length = 0
        
        for time_point in sorted(time_groups.keys()):
            # Add rest if there's a gap
            if time_point > last_time:
                rest_duration = time_point - last_time
                dur_bucket = self._get_duration_bucket(rest_duration)
                rest_token = f'REST_{dur_bucket}'
                if rest_token in self.vocab:
                    tokens.append(self.vocab[rest_token])
            
            # Process simultaneous events
            group = time_groups[time_point]
            
            # Separate notes and chords
            notes = [e for e in group if e.type == 'note']
            chords = [e for e in group if e.type == 'chord']
            other = [e for e in group if e.type not in ['note', 'chord']]
            
            # Add chord tokens first
            for chord_event in chords:
                if isinstance(chord_event.pitch, list) and len(chord_event.pitch) > 1:
                    root = chord_event.pitch[0] % 12
                    chord_type = self._detect_chord_type(chord_event.pitch)
                    chord_token = f'CHORD_{root}_{chord_type}'
                    if chord_token in self.vocab:
                        tokens.append(self.vocab[chord_token])
            
            # Add note tokens
            for note_event in notes:
                if note_event.pitch is not None and 0 <= note_event.pitch <= 127:
                    vel_bucket = self._get_velocity_bucket(note_event.velocity or 64)
                    dur_bucket = self._get_duration_bucket(note_event.duration or 1.0)
                    note_token = f'NOTE_{note_event.pitch}_{vel_bucket}_{dur_bucket}'
                    if note_token in self.vocab:
                        tokens.append(self.vocab[note_token])
            
            # Add structural tokens
            phrase_length += 1
            if phrase_length >= 16:  # Start new phrase every 16 events
                tokens.append(self.vocab['PHRASE_END'])
                tokens.append(self.vocab['PHRASE_START'])
                phrase_length = 0
            
            # Add other events (tempo, key, etc.)
            for event in other:
                if event.type == 'tempo' and event.tempo is not None:
                    if event.tempo < 80:
                        tempo_token = 'TEMPO_slow'
                    elif event.tempo < 120:
                        tempo_token = 'TEMPO_medium'
                    elif event.tempo < 160:
                        tempo_token = 'TEMPO_fast'
                    else:
                        tempo_token = 'TEMPO_very_fast'
                    
                    if tempo_token in self.vocab:
                        tokens.append(self.vocab[tempo_token])
                
                elif event.type == 'time_sig':
                    time_sig_token = f'TIME_SIG_{event.numerator}_{event.denominator}'
                    if time_sig_token in self.vocab:
                        tokens.append(self.vocab[time_sig_token])
            
            last_time = time_point
        
        tokens.append(self.eos_token)
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> List[MusicEvent]:
        """Convert tokens back to musical events"""
        events = []
        current_time = 0.0
        
        for token in tokens:
            if token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            
            token_str = self.reverse_vocab.get(token, '<UNK>')
            
            if token_str.startswith('NOTE_'):
                # Parse note token: NOTE_pitch_vel_dur
                parts = token_str.split('_')
                if len(parts) == 4:
                    pitch = int(parts[1])
                    vel_bucket = int(parts[2])
                    dur_bucket = int(parts[3])
                    
                    velocity = vel_bucket * 16 + 8  # Convert back from bucket
                    durations = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
                    duration = durations[min(dur_bucket, len(durations) - 1)]
                    
                    events.append(MusicEvent(
                        type='note',
                        time=current_time,
                        pitch=pitch,
                        velocity=velocity,
                        duration=duration
                    ))
            
            elif token_str.startswith('REST_'):
                # Parse rest token
                parts = token_str.split('_')
                if len(parts) == 2:
                    dur_bucket = int(parts[1])
                    durations = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
                    duration = durations[min(dur_bucket, len(durations) - 1)]
                    current_time += duration
            
            elif token_str.startswith('CHORD_'):
                # Parse chord token
                parts = token_str.split('_')
                if len(parts) == 3:
                    root = int(parts[1]) + 60  # Middle C octave
                    chord_type = parts[2]
                    
                    # Generate chord pitches
                    if chord_type == 'maj':
                        pitches = [root, root + 4, root + 7]
                    elif chord_type == 'min':
                        pitches = [root, root + 3, root + 7]
                    elif chord_type == 'dim':
                        pitches = [root, root + 3, root + 6]
                    elif chord_type == 'aug':
                        pitches = [root, root + 4, root + 8]
                    else:
                        pitches = [root, root + 4, root + 7]  # Default to major
                    
                    events.append(MusicEvent(
                        type='chord',
                        time=current_time,
                        pitch=pitches,
                        velocity=64,
                        duration=1.0
                    ))
            
            elif token_str.startswith('TEMPO_'):
                tempo_map = {
                    'TEMPO_slow': 70,
                    'TEMPO_medium': 100,
                    'TEMPO_fast': 140,
                    'TEMPO_very_fast': 180
                }
                tempo = tempo_map.get(token_str, 120)
                events.append(MusicEvent(
                    type='tempo',
                    time=current_time,
                    tempo=tempo
                ))
        
        return events
    
    def save_vocabulary(self, path: Union[str, Path]):
        """Save vocabulary to file"""
        vocab_data = {
            'vocab': self.vocab,
            'reverse_vocab': {str(k): v for k, v in self.reverse_vocab.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    @classmethod
    def load_vocabulary(cls, path: Union[str, Path], config: DataConfig) -> 'EventTokenizer':
        """Load tokenizer vocabulary from file"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(config)
        tokenizer.vocab = vocab_data['vocab']
        tokenizer.reverse_vocab = {int(k): v for k, v in vocab_data['reverse_vocab'].items()}
        tokenizer.vocab_size = vocab_data['vocab_size']
        
        # Re-create convenience mappings after loading
        tokenizer.pad_token = tokenizer.vocab['<PAD>']
        tokenizer.bos_token = tokenizer.vocab['<BOS>']
        tokenizer.eos_token = tokenizer.vocab['<EOS>']
        tokenizer.unk_token = tokenizer.vocab['<UNK>']
        
        # Compatibility with generator (uppercase attributes)
        tokenizer.PAD_TOKEN = tokenizer.pad_token
        tokenizer.BOS_TOKEN = tokenizer.bos_token
        tokenizer.EOS_TOKEN = tokenizer.eos_token
        tokenizer.UNK_TOKEN = tokenizer.unk_token
        
        return tokenizer

class DataProcessor:
    """Enhanced data processor with better musical understanding"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = EventTokenizer(config)
    
    def process_events(
        self,
        events_list: List[List[MusicEvent]],
        augment: bool = True
    ) -> List[List[MusicEvent]]:
        """Process and augment event sequences with better musical structure"""
        processed_events = []
        
        for events in events_list:
            # Filter and clean events
            cleaned = self._clean_events(events)
            if len(cleaned) < 10:  # Skip very short sequences
                continue
                
            # Basic processing
            processed = self._process_single_sequence(cleaned)
            processed_events.append(processed)
            
            if augment:
                # Musical transposition (within reasonable ranges)
                for semitones in [-5, -3, -2, -1, 1, 2, 3, 5]:
                    transposed = self._transpose_sequence(processed, semitones)
                    if transposed:  # Only add if transposition was successful
                        processed_events.append(transposed)
                
                # Rhythm variations
                for factor in [0.75, 0.9, 1.1, 1.25]:
                    tempo_varied = self._vary_tempo(processed, factor)
                    processed_events.append(tempo_varied)
                
                # Dynamics variations
                for vel_factor in [0.7, 0.85, 1.15, 1.3]:
                    dynamics_varied = self._vary_dynamics(processed, vel_factor)
                    processed_events.append(dynamics_varied)
        
        print(f"Processed {len(events_list)} original sequences into {len(processed_events)} training sequences")
        return processed_events
    
    def _clean_events(self, events: List[MusicEvent]) -> List[MusicEvent]:
        """Clean and filter events"""
        cleaned = []
        
        for event in events:
            # Filter out invalid events
            if event.type == 'note':
                if (event.pitch is not None and 
                    0 <= event.pitch <= 127 and 
                    event.duration is not None and 
                    0.0625 <= event.duration <= 8.0):  # Reasonable duration range
                    cleaned.append(event)
            elif event.type == 'chord':
                if (isinstance(event.pitch, list) and 
                    len(event.pitch) >= 2 and 
                    all(0 <= p <= 127 for p in event.pitch)):
                    cleaned.append(event)
            else:
                cleaned.append(event)
        
        return cleaned
    
    def _process_single_sequence(self, events: List[MusicEvent]) -> List[MusicEvent]:
        """Process a single sequence with better timing"""
        # Sort by time
        events = sorted(events, key=lambda x: x.time)
        
        # Normalize time to start at 0
        if events:
            start_time = events[0].time
            for event in events:
                event.time -= start_time
        
        # Quantize to musical grid (32nd notes)
        quantized = []
        for event in events:
            event_copy = MusicEvent(
                type=event.type,
                time=self._quantize_time(event.time),
                duration=self._quantize_time(event.duration) if event.duration else None,
                pitch=event.pitch,
                velocity=self._quantize_velocity(event.velocity) if event.velocity else None,
                tempo=event.tempo,
                numerator=event.numerator,
                denominator=event.denominator,
                key=event.key
            )
            quantized.append(event_copy)
        
        return quantized
    
    def _quantize_time(self, time: Optional[float]) -> Optional[float]:
        """Quantize time to musical grid (32nd notes)"""
        if time is None:
            return None
        subdivision = 0.125  # 32nd note
        return round(time / subdivision) * subdivision
    
    def _quantize_velocity(self, velocity: Optional[int]) -> Optional[int]:
        """Quantize velocity to reasonable ranges"""
        if velocity is None:
            return None
        # Ensure velocity is in valid MIDI range and somewhat musical
        velocity = max(20, min(127, velocity))  # Avoid too quiet notes
        return velocity

    def prepare_dataset(
        self,
        events_list: List[List[MusicEvent]],
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """Prepare dataset with better sequence handling"""
        # Convert events to token sequences
        sequences = []
        
        for events in events_list:
            tokens = self.tokenizer.encode_events(events)
            
            if len(tokens) > self.config.max_sequence_length:
                # Create overlapping windows for better continuity
                window_size = self.config.max_sequence_length
                step_size = window_size // 3  # 2/3 overlap
                
                for i in range(0, len(tokens) - self.config.min_sequence_length, step_size):
                    chunk = tokens[i:i + window_size]
                    if len(chunk) >= self.config.min_sequence_length:
                        sequences.append(chunk)
            elif len(tokens) >= self.config.min_sequence_length:
                sequences.append(tokens)
        
        print(f"Created {len(sequences)} training sequences")
        
        # Create input/target pairs
        inputs = []
        targets = []
        
        for sequence in sequences:
            # Ensure sequence has BOS/EOS tokens
            if sequence[0] != self.tokenizer.bos_token:
                sequence = [self.tokenizer.bos_token] + sequence
            if sequence[-1] != self.tokenizer.eos_token:
                sequence = sequence + [self.tokenizer.eos_token]
            
            # Create training examples with next-token prediction
            max_len = min(len(sequence), self.config.max_sequence_length)
            if max_len >= self.config.min_sequence_length:
                input_seq = sequence[:max_len-1]
                target_seq = sequence[1:max_len]
                
                # Pad to fixed length
                pad_length = self.config.max_sequence_length - 1 - len(input_seq)
                if pad_length > 0:
                    input_seq = input_seq + [self.tokenizer.pad_token] * pad_length
                    target_seq = target_seq + [self.tokenizer.pad_token] * pad_length
                
                inputs.append(input_seq)
                targets.append(target_seq)
        
        print(f"Generated {len(inputs)} training examples")
        
        # Convert to arrays
        inputs = np.array(inputs, dtype=np.int64)
        targets = np.array(targets, dtype=np.int64)
        
        # Create dataset and dataloader
        dataset = MelodiaDataset(inputs, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        return dataloader

    def _transpose_sequence(
        self,
        events: List[MusicEvent],
        semitones: int
    ) -> List[MusicEvent]:
        """Transpose sequence with validation"""
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
            
            if event.type == 'note' and isinstance(event.pitch, int):
                new_pitch = event.pitch + semitones
                if 21 <= new_pitch <= 108:  # Piano range
                    event_copy.pitch = new_pitch
                    transposed.append(event_copy)
            elif event.type == 'chord' and isinstance(event.pitch, list):
                new_pitches = [p + semitones for p in event.pitch]
                if all(21 <= p <= 108 for p in new_pitches):  # All in piano range
                    event_copy.pitch = new_pitches
                    transposed.append(event_copy)
            else:
                transposed.append(event_copy)
        
        return transposed if len(transposed) >= len(events) * 0.8 else []  # Keep if 80% notes preserved

    def _vary_tempo(
        self,
        events: List[MusicEvent],
        tempo_factor: float
    ) -> List[MusicEvent]:
        """Vary tempo while preserving musical structure"""
        scaled = []
        time_scale = 1.0 / tempo_factor
        
        for event in events:
            event_copy = MusicEvent(
                type=event.type,
                time=event.time * time_scale,
                duration=event.duration * time_scale if event.duration else None,
                pitch=event.pitch,
                velocity=event.velocity,
                tempo=event.tempo * tempo_factor if event.tempo else None,
                numerator=event.numerator,
                denominator=event.denominator,
                key=event.key
            )
            scaled.append(event_copy)
        
        return scaled
    
    def _vary_dynamics(
        self,
        events: List[MusicEvent],
        velocity_factor: float
    ) -> List[MusicEvent]:
        """Vary dynamics (velocity) of notes"""
        varied = []
        
        for event in events:
            event_copy = MusicEvent(
                type=event.type,
                time=event.time,
                duration=event.duration,
                pitch=event.pitch,
                velocity=None,
                tempo=event.tempo,
                numerator=event.numerator,
                denominator=event.denominator,
                key=event.key
            )
            
            if event.velocity is not None:
                new_velocity = int(event.velocity * velocity_factor)
                event_copy.velocity = max(20, min(127, new_velocity))
            else:
                event_copy.velocity = event.velocity
            
            varied.append(event_copy)
        
        return varied

    def save_processor(self, path: Union[str, Path]):
        """Save processor state"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer.save_vocabulary(save_path / 'vocabulary.json')
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def load_processor(cls, path: Union[str, Path]) -> 'DataProcessor':
        """Load saved processor"""
        load_path = Path(path)
        
        with open(load_path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        config = DataConfig(**config_dict)
        
        processor = cls(config)
        processor.tokenizer = EventTokenizer.load_vocabulary(
            load_path / 'vocabulary.json',
            config
        )
        
        return processor