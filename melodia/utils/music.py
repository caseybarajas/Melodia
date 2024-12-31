# melodia/utils/music.py

from typing import List, Dict, Optional, Union, Tuple, Set
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Scale(Enum):
    """Common musical scales"""
    MAJOR = [0, 2, 4, 5, 7, 9, 11]
    NATURAL_MINOR = [0, 2, 3, 5, 7, 8, 10]
    HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]
    MELODIC_MINOR = [0, 2, 3, 5, 7, 9, 11]
    DORIAN = [0, 2, 3, 5, 7, 9, 10]
    PHRYGIAN = [0, 1, 3, 5, 7, 8, 10]
    LYDIAN = [0, 2, 4, 6, 7, 9, 11]
    MIXOLYDIAN = [0, 2, 4, 5, 7, 9, 10]
    PENTATONIC_MAJOR = [0, 2, 4, 7, 9]
    PENTATONIC_MINOR = [0, 3, 5, 7, 10]
    BLUES = [0, 3, 5, 6, 7, 10]

class ChordQuality(Enum):
    """Common chord qualities"""
    MAJOR = [0, 4, 7]
    MINOR = [0, 3, 7]
    DIMINISHED = [0, 3, 6]
    AUGMENTED = [0, 4, 8]
    MAJOR_7 = [0, 4, 7, 11]
    MINOR_7 = [0, 3, 7, 10]
    DOMINANT_7 = [0, 4, 7, 10]
    HALF_DIMINISHED_7 = [0, 3, 6, 10]
    DIMINISHED_7 = [0, 3, 6, 9]

class MusicTheory:
    """Music theory utilities"""
    
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    @classmethod
    def note_to_midi(cls, note: str) -> int:
        """Convert note name to MIDI note number"""
        note_name = note.strip()[:-1]
        octave = int(note.strip()[-1])
        note_idx = cls.NOTES.index(note_name)
        return (octave + 1) * 12 + note_idx
    
    @classmethod
    def midi_to_note(cls, midi_note: int) -> str:
        """Convert MIDI note number to note name"""
        octave = (midi_note // 12) - 1
        note_idx = midi_note % 12
        return f"{cls.NOTES[note_idx]}{octave}"
    
    @classmethod
    def get_scale(cls, root: Union[str, int], scale_type: Scale) -> List[int]:
        """Get MIDI note numbers for a scale"""
        if isinstance(root, str):
            root = cls.note_to_midi(root)
        root = root % 12
        return [(root + interval) % 12 for interval in scale_type.value]
    
    @classmethod
    def get_chord(
        cls,
        root: Union[str, int],
        quality: ChordQuality,
        inversion: int = 0
    ) -> List[int]:
        """Get MIDI note numbers for a chord"""
        if isinstance(root, str):
            root = cls.note_to_midi(root)
        root = root % 12
        notes = [(root + interval) % 12 for interval in quality.value]
        
        # Apply inversion
        if inversion > 0:
            notes = notes[inversion:] + [n + 12 for n in notes[:inversion]]
        
        return notes
    
    @staticmethod
    def is_consonant(interval: int) -> bool:
        """Check if an interval is consonant"""
        consonant_intervals = {0, 3, 4, 7, 8, 9}  # Unison, thirds, fourth, fifth, sixths
        return (interval % 12) in consonant_intervals
    
    @staticmethod
    def get_interval_quality(interval: int) -> str:
        """Get the quality of an interval"""
        qualities = {
            0: 'Perfect Unison',
            1: 'Minor Second',
            2: 'Major Second',
            3: 'Minor Third',
            4: 'Major Third',
            5: 'Perfect Fourth',
            6: 'Tritone',
            7: 'Perfect Fifth',
            8: 'Minor Sixth',
            9: 'Major Sixth',
            10: 'Minor Seventh',
            11: 'Major Seventh',
            12: 'Perfect Octave'
        }
        return qualities.get(interval % 12, 'Unknown')

class Progression:
    """Chord progression utilities"""
    
    # Common chord progressions
    COMMON_PROGRESSIONS = {
        'I-IV-V': [0, 5, 7],
        'I-V-vi-IV': [0, 7, 9, 5],
        'ii-V-I': [2, 7, 0],
        'I-vi-IV-V': [0, 9, 5, 7],
        'vi-IV-I-V': [9, 5, 0, 7],
        'I-IV-vi-V': [0, 5, 9, 7]
    }
    
    @classmethod
    def get_progression(
        cls,
        root: Union[str, int],
        progression_name: str,
        scale_type: Scale = Scale.MAJOR
    ) -> List[List[int]]:
        """Get a chord progression in a given key"""
        if isinstance(root, str):
            root = MusicTheory.note_to_midi(root)
        root = root % 12
        
        base_degrees = cls.COMMON_PROGRESSIONS[progression_name]
        scale_degrees = MusicTheory.get_scale(root, scale_type)
        
        progression = []
        for degree in base_degrees:
            chord_root = scale_degrees[degree % len(scale_degrees)]
            if scale_type == Scale.MAJOR:
                if degree in [0, 5, 7]:  # I, IV, V
                    chord = MusicTheory.get_chord(chord_root, ChordQuality.MAJOR)
                else:  # ii, iii, vi, vii
                    chord = MusicTheory.get_chord(chord_root, ChordQuality.MINOR)
            else:
                if degree in [2, 5]:  # ii, v
                    chord = MusicTheory.get_chord(chord_root, ChordQuality.DIMINISHED)
                else:
                    chord = MusicTheory.get_chord(chord_root, ChordQuality.MINOR)
            progression.append(chord)
        
        return progression
    
    @staticmethod
    def analyze_progression(chords: List[List[int]]) -> List[str]:
        """Analyze a chord progression and identify common patterns"""
        # Convert to root notes
        roots = [min(chord) % 12 for chord in chords]
        
        # Look for common patterns
        known_patterns = {
            (0, 5, 7): 'I-IV-V',
            (0, 7, 9, 5): 'I-V-vi-IV',
            (2, 7, 0): 'ii-V-I',
            (0, 9, 5, 7): 'I-vi-IV-V'
        }
        
        # Normalize to C (root = 0)
        offset = roots[0]
        normalized = [(r - offset) % 12 for r in roots]
        
        # Check for known patterns
        patterns = []
        for i in range(len(normalized)):
            for length in range(3, len(normalized) + 1):
                if i + length <= len(normalized):
                    segment = tuple(normalized[i:i+length])
                    if segment in known_patterns:
                        patterns.append((i, known_patterns[segment]))
        
        return patterns

class VoiceLeading:
    """Voice leading utilities"""
    
    @staticmethod
    def minimize_movement(
        chord1: List[int],
        chord2: List[int]
    ) -> List[int]:
        """Optimize voice leading between two chords"""
        if not chord1 or not chord2:
            return chord2
        
        # Extend chords to same length
        max_len = max(len(chord1), len(chord2))
        c1 = chord1 + [n + 12 for n in chord1[:max_len - len(chord1)]]
        c2 = chord2 + [n + 12 for n in chord2[:max_len - len(chord2)]]
        
        # Find closest notes
        result = []
        used = set()
        
        for note1 in c1:
            distances = [(abs(note1 - note2), note2) for note2 in c2 
                        if note2 not in used]
            if distances:
                _, closest = min(distances)
                result.append(closest)
                used.add(closest)
        
        return sorted(result)
    
    @staticmethod
    def resolve_leading_tone(
        chord: List[int],
        scale: List[int]
    ) -> List[int]:
        """Resolve leading tone to tonic"""
        tonic = scale[0]
        leading_tone = scale[-1]
        
        resolved = []
        for note in chord:
            if note % 12 == leading_tone:
                resolved.append(tonic + (note // 12) * 12)
            else:
                resolved.append(note)
        
        return resolved
    
    @staticmethod
    def avoid_parallel_fifths(
        chord1: List[int],
        chord2: List[int]
    ) -> List[int]:
        """Adjust voice leading to avoid parallel fifths"""
        if len(chord1) < 2 or len(chord2) < 2:
            return chord2
        
        # Find fifths in first chord
        fifths1 = []
        for i in range(len(chord1)):
            for j in range(i + 1, len(chord1)):
                if (chord1[j] - chord1[i]) % 12 == 7:
                    fifths1.append((i, j))
        
        # Check and adjust parallel fifths
        result = list(chord2)
        for i, j in fifths1:
            if i < len(chord2) and j < len(chord2):
                if (chord2[j] - chord2[i]) % 12 == 7:
                    # Adjust second chord to avoid parallel fifth
                    result[j] = (result[j] + 1) % 12
        
        return result