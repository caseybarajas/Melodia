# melodia/data/loader.py

import mido
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from music21 import converter, instrument, note, chord, stream, tempo, meter, key
from ..config import DataConfig

logger = logging.getLogger(__name__)

@dataclass
class MusicEvent:
    """Representation of a musical event"""
    type: str  # 'note', 'chord', 'tempo', 'time_sig', 'key_sig'
    time: float
    duration: Optional[float] = None
    pitch: Optional[Union[int, List[int]]] = None
    velocity: Optional[int] = None
    numerator: Optional[int] = None
    denominator: Optional[int] = None
    tempo: Optional[float] = None
    key: Optional[str] = None

class MIDILoader:
    """Loads and processes MIDI files into a standardized format"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration parameters"""
        if not isinstance(self.config.valid_time_signatures, (list, tuple)):
            raise ValueError("valid_time_signatures must be a list or tuple")
            
    def load_midi(self, file_path: Union[str, Path]) -> List[MusicEvent]:
        """Load a MIDI file and convert it to a list of music events"""
        try:
            score = converter.parse(str(file_path))
            return self._process_score(score)
        except Exception as e:
            logger.error(f"Error loading MIDI file {file_path}: {str(e)}")
            return []
    
    def load_file(self, file_path: Union[str, Path]) -> List[MusicEvent]:
        """Alias for load_midi for backward compatibility"""
        return self.load_midi(file_path)
    
    def _process_score(self, score: stream.Score) -> List[MusicEvent]:
        """Process a music21 Score object into a list of MusicEvent objects"""
        events = []
        
        # Extract global properties
        self._extract_metadata(score, events)
        
        # Process each part
        parts = instrument.partitionByInstrument(score)
        if parts:
            for part in parts.parts:
                events.extend(self._process_part(part))
        else:
            events.extend(self._process_part(score))
        
        # Sort events by time
        events.sort(key=lambda x: x.time)
        
        return events
    
    def _extract_metadata(self, score: stream.Score, events: List[MusicEvent]):
        """Extract global score metadata"""
        # Time signature
        ts = score.getTimeSignatures()[0] if score.getTimeSignatures() else meter.TimeSignature('4/4')
        if (ts.numerator, ts.denominator) in self.config.valid_time_signatures:
            events.append(MusicEvent(
                type='time_sig',
                time=0.0,
                numerator=ts.numerator,
                denominator=ts.denominator
            ))
        
        # Key signature
        ks = score.analyze('key')
        events.append(MusicEvent(
            type='key_sig',
            time=0.0,
            key=str(ks)
        ))
        
        # Tempo
        try:
            tmp = score.metronomeMarkBoundaries()
            if tmp:
                tempo_mark = tmp[0][1]
                tempo_value = tempo_mark.number if hasattr(tempo_mark, 'number') else float(tempo_mark)
                events.append(MusicEvent(
                    type='tempo',
                    time=0.0,
                    tempo=tempo_value
                ))
        except Exception:
            # Default tempo if extraction fails
            events.append(MusicEvent(
                type='tempo',
                time=0.0,
                tempo=120.0
            ))
    
    def _process_part(self, part: stream.Part) -> List[MusicEvent]:
        """Process a single part/voice of the score"""
        events = []
        
        for element in part.recurse():
            if isinstance(element, note.Note):
                events.append(MusicEvent(
                    type='note',
                    time=element.offset,
                    duration=element.duration.quarterLength,
                    pitch=element.pitch.midi,
                    velocity=element.volume.velocity if element.volume.velocity else 64
                ))
            elif isinstance(element, chord.Chord):
                events.append(MusicEvent(
                    type='chord',
                    time=element.offset,
                    duration=element.duration.quarterLength,
                    pitch=[p.midi for p in element.pitches],
                    velocity=element.volume.velocity if element.volume.velocity else 64
                ))
            elif isinstance(element, tempo.MetronomeMark):
                tempo_value = element.number if hasattr(element, 'number') else float(element)
                events.append(MusicEvent(
                    type='tempo',
                    time=element.offset,
                    tempo=tempo_value
                ))
            elif isinstance(element, meter.TimeSignature):
                if (element.numerator, element.denominator) in self.config.valid_time_signatures:
                    events.append(MusicEvent(
                        type='time_sig',
                        time=element.offset,
                        numerator=element.numerator,
                        denominator=element.denominator
                    ))
            elif isinstance(element, key.KeySignature):
                events.append(MusicEvent(
                    type='key_sig',
                    time=element.offset,
                    key=str(element)
                ))
        
        return events
    
    def save_events(self, events: List[MusicEvent], output_path: Union[str, Path]):
        """Save music events back to a MIDI file"""
        score = stream.Score()
        part = stream.Part()
        
        current_time = 0.0
        for event in events:
            if event.type in ['note', 'chord']:
                if event.type == 'note':
                    n = note.Note(
                        pitch=event.pitch,
                        quarterLength=event.duration
                    )
                    n.volume.velocity = event.velocity
                else:
                    n = chord.Chord(
                        event.pitch,
                        quarterLength=event.duration
                    )
                    n.volume.velocity = event.velocity
                
                n.offset = event.time
                part.append(n)
            elif event.type == 'tempo':
                t = tempo.MetronomeMark(number=event.tempo)
                t.offset = event.time
                part.append(t)
            elif event.type == 'time_sig':
                ts = meter.TimeSignature(f'{event.numerator}/{event.denominator}')
                ts.offset = event.time
                part.append(ts)
            elif event.type == 'key_sig':
                ks = key.KeySignature(key.Key(event.key).sharps)
                ks.offset = event.time
                part.append(ks)
        
        score.append(part)
        score.write('midi', str(output_path))