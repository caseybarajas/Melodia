# melodia/data/tokenizer.py

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from ..config import DataConfig

logger = logging.getLogger(__name__)

@dataclass
class TokenRange:
    """Defines a range of token values for a specific event type"""
    start: int
    size: int
    
    @property
    def end(self) -> int:
        return self.start + self.size
    
    def contains(self, token: int) -> bool:
        return self.start <= token < self.end

class VocabularyConfig:
    """Configuration for token vocabulary ranges"""
    
    def __init__(self):
        # Special tokens
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.MASK = 4
        
        # Token ranges for different event types
        self.NOTE_PITCH = TokenRange(start=16, size=128)  # MIDI pitch range
        self.NOTE_VELOCITY = TokenRange(start=144, size=128)  # MIDI velocity range
        self.NOTE_DURATION = TokenRange(start=272, size=64)  # Quantized durations
        self.TIME_SHIFT = TokenRange(start=336, size=100)  # Time shifts
        self.CHORD = TokenRange(start=436, size=72)  # Common chord types
        self.TIME_SIGNATURE = TokenRange(start=508, size=16)  # Common time signatures
        self.KEY_SIGNATURE = TokenRange(start=524, size=30)  # Major/minor keys
        self.TEMPO = TokenRange(start=554, size=50)  # Quantized tempo values
        self.PROGRAM = TokenRange(start=604, size=128)  # MIDI program numbers
        
        # Additional tokens for special events
        self.SPECIAL_START = 732
        self.special_tokens = {
            'SECTION_START': self.SPECIAL_START,
            'SECTION_END': self.SPECIAL_START + 1,
            'PHRASE_START': self.SPECIAL_START + 2,
            'PHRASE_END': self.SPECIAL_START + 3
        }

class MusicTokenizer:
    """Handles conversion between musical events and tokens"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.vocab_config = VocabularyConfig()
        
        # Initialize mappings
        self._init_mappings()
        
        # Track vocabulary statistics
        self.token_counts = {}
        self.total_tokens = 0
    
    def _init_mappings(self):
        """Initialize token-to-event and event-to-token mappings"""
        self.token_to_event = {
            self.vocab_config.PAD: '<pad>',
            self.vocab_config.BOS: '<bos>',
            self.vocab_config.EOS: '<eos>',
            self.vocab_config.UNK: '<unk>',
            self.vocab_config.MASK: '<mask>'
        }
        
        # Note pitch mappings
        for pitch in range(128):
            token = self.vocab_config.NOTE_PITCH.start + pitch
            self.token_to_event[token] = f'PITCH_{pitch}'
        
        # Note velocity mappings (quantized)
        for vel in range(0, 128, 2):  # Quantize to 64 levels
            token = self.vocab_config.NOTE_VELOCITY.start + (vel // 2)
            self.token_to_event[token] = f'VEL_{vel}'
        
        # Duration mappings (logarithmic quantization)
        durations = self._get_quantized_durations()
        for i, dur in enumerate(durations):
            token = self.vocab_config.NOTE_DURATION.start + i
            self.token_to_event[token] = f'DUR_{dur:.3f}'
        
        # Time shift mappings
        shifts = self._get_time_shifts()
        for i, shift in enumerate(shifts):
            token = self.vocab_config.TIME_SHIFT.start + i
            self.token_to_event[token] = f'SHIFT_{shift:.3f}'
        
        # Chord mappings
        chord_types = ['maj', 'min', '7', 'maj7', 'min7', 'dim']
        roots = list(range(12))  # All pitch classes
        for i, root in enumerate(roots):
            for j, chord_type in enumerate(chord_types):
                token = self.vocab_config.CHORD.start + (i * len(chord_types) + j)
                self.token_to_event[token] = f'CHORD_{root}_{chord_type}'
        
        # Time signature mappings
        time_sigs = [(4, 4), (3, 4), (6, 8), (2, 4), (9, 8)]
        for i, (num, den) in enumerate(time_sigs):
            token = self.vocab_config.TIME_SIGNATURE.start + i
            self.token_to_event[token] = f'TIME_{num}_{den}'
        
        # Key signature mappings
        keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
               'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
        for i, key_name in enumerate(keys):
            for mode in ['major', 'minor']:
                token = self.vocab_config.KEY_SIGNATURE.start + (i * 2) + (0 if mode == 'major' else 1)
                self.token_to_event[token] = f'KEY_{key_name}_{mode}'
        
        # Tempo mappings (quantized)
        tempo_range = np.arange(30, 250, 5)  # 30-250 BPM in steps of 5
        for i, tempo in enumerate(tempo_range):
            token = self.vocab_config.TEMPO.start + i
            self.token_to_event[token] = f'TEMPO_{tempo}'
        
        # Program (instrument) mappings
        for program in range(128):
            token = self.vocab_config.PROGRAM.start + program
            self.token_to_event[token] = f'PROGRAM_{program}'
        
        # Special event mappings
        for event, token in self.vocab_config.special_tokens.items():
            self.token_to_event[token] = event
        
        # Create reverse mapping
        self.event_to_token = {v: k for k, v in self.token_to_event.items()}
    
    def _get_quantized_durations(self) -> List[float]:
        """Get logarithmically quantized note durations"""
        # Range from 32nd note to 4 bars
        min_duration = 0.125  # 32nd note
        max_duration = 16.0   # 4 bars
        num_durations = self.vocab_config.NOTE_DURATION.size
        
        return np.exp(
            np.linspace(
                np.log(min_duration),
                np.log(max_duration),
                num_durations
            )
        )
    
    def _get_time_shifts(self) -> List[float]:
        """Get quantized time shift values"""
        # Range from 32nd note to 1 bar
        min_shift = 0.125
        max_shift = 4.0
        num_shifts = self.vocab_config.TIME_SHIFT.size
        
        return np.linspace(min_shift, max_shift, num_shifts)
    
    def quantize_duration(self, duration: float) -> int:
        """Quantize a duration value to nearest token"""
        durations = self._get_quantized_durations()
        idx = np.abs(durations - duration).argmin()
        return self.vocab_config.NOTE_DURATION.start + idx
    
    def quantize_time_shift(self, shift: float) -> int:
        """Quantize a time shift value to nearest token"""
        shifts = self._get_time_shifts()
        idx = np.abs(shifts - shift).argmin()
        return self.vocab_config.TIME_SHIFT.start + idx
    
    def encode_events(self, events: List[Dict]) -> List[int]:
        """Convert a sequence of musical events to tokens"""
        tokens = [self.vocab_config.BOS]
        current_time = 0.0
        
        for event in events:
            # Handle time shift
            time_shift = event['time'] - current_time
            if time_shift > 0:
                shift_token = self.quantize_time_shift(time_shift)
                tokens.append(shift_token)
            
            # Encode event based on type
            if event['type'] == 'note':
                # Note pitch
                pitch_token = self.vocab_config.NOTE_PITCH.start + event['pitch']
                tokens.append(pitch_token)
                
                # Note velocity (quantized)
                vel_token = self.vocab_config.NOTE_VELOCITY.start + (event['velocity'] // 2)
                tokens.append(vel_token)
                
                # Note duration
                dur_token = self.quantize_duration(event['duration'])
                tokens.append(dur_token)
            
            elif event['type'] == 'chord':
                chord_token = self.event_to_token.get(
                    f"CHORD_{event['root']}_{event['quality']}",
                    self.vocab_config.UNK
                )
                tokens.append(chord_token)
            
            elif event['type'] == 'time_signature':
                time_sig_token = self.event_to_token.get(
                    f"TIME_{event['numerator']}_{event['denominator']}",
                    self.vocab_config.UNK
                )
                tokens.append(time_sig_token)
            
            elif event['type'] == 'key_signature':
                key_token = self.event_to_token.get(
                    f"KEY_{event['key']}_{event['mode']}",
                    self.vocab_config.UNK
                )
                tokens.append(key_token)
            
            elif event['type'] == 'tempo':
                tempo_val = round(event['tempo'] / 5) * 5  # Quantize to nearest 5 BPM
                tempo_token = self.event_to_token.get(
                    f"TEMPO_{tempo_val}",
                    self.vocab_config.UNK
                )
                tokens.append(tempo_token)
            
            elif event['type'] == 'program':
                program_token = self.vocab_config.PROGRAM.start + event['program']
                tokens.append(program_token)
            
            current_time = event['time']
        
        tokens.append(self.vocab_config.EOS)
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> List[Dict]:
        """Convert a sequence of tokens back to musical events"""
        events = []
        current_time = 0.0
        
        for token in tokens:
            if token == self.vocab_config.PAD:
                continue
            if token in [self.vocab_config.BOS, self.vocab_config.EOS]:
                continue
                
            event = self._decode_single_token(token, current_time)
            if event:
                if 'time_shift' in event:
                    current_time += event.pop('time_shift')
                else:
                    events.append(event)
        
        return events
    
    def _decode_single_token(self, token: int, current_time: float) -> Optional[Dict]:
        """Decode a single token to an event"""
        event_str = self.token_to_event.get(token, '<unk>')
        
        # Time shift
        if self.vocab_config.TIME_SHIFT.contains(token):
            idx = token - self.vocab_config.TIME_SHIFT.start
            shifts = self._get_time_shifts()
            return {'time_shift': shifts[idx]}
        
        # Note pitch
        if self.vocab_config.NOTE_PITCH.contains(token):
            pitch = token - self.vocab_config.NOTE_PITCH.start
            return {
                'type': 'note',
                'time': current_time,
                'pitch': pitch
            }
        
        # Parse other event types
        parts = event_str.split('_')
        event_type = parts[0].lower()
        
        if event_type == 'chord':
            return {
                'type': 'chord',
                'time': current_time,
                'root': int(parts[1]),
                'quality': parts[2]
            }
        elif event_type == 'time':
            return {
                'type': 'time_signature',
                'time': current_time,
                'numerator': int(parts[1]),
                'denominator': int(parts[2])
            }
        elif event_type == 'key':
            return {
                'type': 'key_signature',
                'time': current_time,
                'key': parts[1],
                'mode': parts[2]
            }
        elif event_type == 'tempo':
            return {
                'type': 'tempo',
                'time': current_time,
                'tempo': float(parts[1])
            }
        elif event_type == 'program':
            return {
                'type': 'program',
                'time': current_time,
                'program': int(parts[1])
            }
        
        return None
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer configuration and mappings"""
        save_data = {
            'token_to_event': self.token_to_event,
            'vocab_config': vars(self.vocab_config),
            'token_counts': self.token_counts,
            'total_tokens': self.total_tokens
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path], config: DataConfig) -> 'MusicTokenizer':
        """Load tokenizer from saved configuration"""
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        tokenizer = cls(config)
        tokenizer.token_to_event = {int(k): v for k, v in save_data['token_to_event'].items()}
        tokenizer.event_to_token = {v: int(k) for k, v in tokenizer.token_to_event.items()}
        tokenizer.token_counts = save_data['token_counts']
        tokenizer.total_tokens = save_data['total_tokens']
        
        return tokenizer