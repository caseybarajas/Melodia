# melodia/generation/controls.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import logging
from enum import Enum
from ..config import GenerationConfig

logger = logging.getLogger(__name__)

class MusicalStyle(Enum):
    """Predefined musical styles"""
    CLASSICAL = "classical"
    JAZZ = "jazz"
    FOLK = "folk"
    BLUES = "blues"
    CONTEMPORARY = "contemporary"

@dataclass
class HarmonicControls:
    """Controls for harmonic generation"""
    key: Optional[str] = None  # e.g., "C", "F#", "Bb"
    mode: str = "major"  # "major" or "minor"
    chord_progression: Optional[List[str]] = None  # e.g., ["I", "IV", "V", "I"]
    allowed_chords: Optional[List[str]] = None
    harmonic_rhythm: float = 2.0  # Chord changes per bar
    modulation_probability: float = 0.1
    chromaticism: float = 0.1  # 0-1, amount of chromatic notes allowed
    
    def __post_init__(self):
        if self.allowed_chords is None:
            self.allowed_chords = ["maj", "min", "7", "maj7", "min7", "dim"]
        if self.chord_progression is None:
            self.chord_progression = ["I", "IV", "V", "I"]

@dataclass
class RhythmicControls:
    """Controls for rhythmic generation"""
    time_signature: Tuple[int, int] = (4, 4)
    tempo: int = 120
    tempo_variation: float = 0.1  # Allowed tempo variation
    syncopation: float = 0.3  # 0-1, amount of syncopation
    swing: float = 0.0  # 0-1, amount of swing
    complexity: float = 0.5  # 0-1, rhythmic complexity
    allowed_note_values: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    
    def get_quantized_duration(self, duration: float) -> float:
        """Quantize a duration to nearest allowed value"""
        return min(self.allowed_note_values, 
                  key=lambda x: abs(x - duration))

@dataclass
class MelodicControls:
    """Controls for melodic generation"""
    range_min: int = 48  # MIDI note number
    range_max: int = 84
    preferred_intervals: List[int] = field(default_factory=lambda: [0, 2, 3, 4, 5, 7])
    step_probability: float = 0.7  # Probability of stepwise motion
    leap_probability: float = 0.3  # Probability of melodic leaps
    direction_change_probability: float = 0.3
    register_weights: Optional[List[float]] = None  # Weights for different registers
    
    def __post_init__(self):
        if self.register_weights is None:
            # Default to normal distribution across range
            range_size = self.range_max - self.range_min
            center = range_size / 2
            self.register_weights = [
                np.exp(-(i - center)**2 / (2 * (range_size/4)**2))
                for i in range(range_size)
            ]

@dataclass
class StructuralControls:
    """Controls for musical structure"""
    form: str = "AABA"  # Musical form
    section_length: int = 8  # Bars per section
    phrase_length: int = 4  # Bars per phrase
    repetition_probability: float = 0.3
    variation_probability: float = 0.4
    development_probability: float = 0.3
    cadence_points: List[int] = field(default_factory=lambda: [4, 8, 12, 16])

@dataclass
class ExpressionControls:
    """Controls for musical expression"""
    dynamics_range: Tuple[int, int] = (40, 100)  # MIDI velocity
    dynamics_variation: float = 0.2
    articulation_probabilities: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.7,
        "staccato": 0.15,
        "legato": 0.15
    })
    accent_probability: float = 0.1
    rubato_amount: float = 0.1  # Timing variation

class GenerationControls:
    """Main class for managing all generation controls"""
    
    def __init__(
        self,
        config: GenerationConfig,
        style: Optional[Union[str, MusicalStyle]] = None
    ):
        self.config = config
        self.style = MusicalStyle(style) if style else None
        
        # Initialize control components
        self.harmonic = HarmonicControls()
        self.rhythmic = RhythmicControls()
        self.melodic = MelodicControls()
        self.structural = StructuralControls()
        self.expression = ExpressionControls()
        
        if self.style:
            self._apply_style_presets()
    
    def _apply_style_presets(self):
        """Apply predefined settings based on musical style"""
        if self.style == MusicalStyle.JAZZ:
            self.harmonic.allowed_chords.extend(["9", "13", "dim7", "alt"])
            self.harmonic.chromaticism = 0.4
            self.rhythmic.swing = 0.6
            self.rhythmic.syncopation = 0.6
            self.melodic.step_probability = 0.5
            self.melodic.leap_probability = 0.5
            
        elif self.style == MusicalStyle.CLASSICAL:
            self.harmonic.modulation_probability = 0.2
            self.rhythmic.complexity = 0.4
            self.melodic.step_probability = 0.8
            self.structural.form = "ABACA"
            
        elif self.style == MusicalStyle.FOLK:
            self.harmonic.allowed_chords = ["maj", "min"]
            self.rhythmic.complexity = 0.3
            self.melodic.step_probability = 0.9
            self.structural.form = "AABB"
            
        elif self.style == MusicalStyle.BLUES:
            self.harmonic.chord_progression = ["I", "I", "I", "I",
                                             "IV", "IV", "I", "I",
                                             "V", "IV", "I", "V"]
            self.rhythmic.swing = 0.4
            self.melodic.preferred_intervals = [0, 3, 4, 7, 10]
    
    def validate(self) -> bool:
        """Validate all control settings"""
        try:
            self._validate_harmonic()
            self._validate_rhythmic()
            self._validate_melodic()
            self._validate_structural()
            self._validate_expression()
            return True
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return False
    
    def _validate_harmonic(self):
        """Validate harmonic controls"""
        if self.harmonic.key and not self._is_valid_key(self.harmonic.key):
            raise ValueError(f"Invalid key: {self.harmonic.key}")
        if not 0 <= self.harmonic.chromaticism <= 1:
            raise ValueError("Chromaticism must be between 0 and 1")
    
    def _validate_rhythmic(self):
        """Validate rhythmic controls"""
        if self.rhythmic.tempo <= 0:
            raise ValueError("Tempo must be positive")
        if not 0 <= self.rhythmic.swing <= 1:
            raise ValueError("Swing must be between 0 and 1")
    
    def _validate_melodic(self):
        """Validate melodic controls"""
        if self.melodic.range_min >= self.melodic.range_max:
            raise ValueError("Invalid melodic range")
        if not np.isclose(sum(self.melodic.register_weights), 1.0, atol=1e-5):
            logger.warning("Register weights don't sum to 1.0, normalizing...")
            total = sum(self.melodic.register_weights)
            self.melodic.register_weights = [w/total for w in self.melodic.register_weights]
    
    def _validate_structural(self):
        """Validate structural controls"""
        total_prob = (self.structural.repetition_probability +
                     self.structural.variation_probability +
                     self.structural.development_probability)
        if not np.isclose(total_prob, 1.0, atol=1e-5):
            raise ValueError("Structural probabilities must sum to 1.0")
    
    def _validate_expression(self):
        """Validate expression controls"""
        if not np.isclose(sum(self.expression.articulation_probabilities.values()), 1.0, atol=1e-5):
            raise ValueError("Articulation probabilities must sum to 1.0")
    
    @staticmethod
    def _is_valid_key(key: str) -> bool:
        """Check if key is valid"""
        valid_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                     'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
        return key in valid_keys
    
    def to_dict(self) -> Dict:
        """Convert controls to dictionary"""
        return {
            'style': self.style.value if self.style else None,
            'harmonic': self.harmonic.__dict__,
            'rhythmic': self.rhythmic.__dict__,
            'melodic': self.melodic.__dict__,
            'structural': self.structural.__dict__,
            'expression': self.expression.__dict__
        }
    
    @classmethod
    def from_dict(cls, data: Dict, config: GenerationConfig) -> 'GenerationControls':
        """Create controls from dictionary"""
        controls = cls(config, style=data.get('style'))
        
        for category in ['harmonic', 'rhythmic', 'melodic', 'structural', 'expression']:
            if category in data:
                setattr(controls, category, globals()[f"{category.capitalize()}Controls"](**data[category]))
        
        return controls
    
    def get_next_note_constraints(self, context: Dict) -> Dict:
        """Get constraints for next note based on current context"""
        constraints = {
            'pitch_probabilities': self._get_pitch_probabilities(context),
            'duration_probabilities': self._get_duration_probabilities(context),
            'velocity_probabilities': self._get_velocity_probabilities(context),
            'articulation_probabilities': self.expression.articulation_probabilities.copy()
        }
        return constraints
    
    def _get_pitch_probabilities(self, context: Dict) -> np.ndarray:
        """Calculate pitch probabilities based on context"""
        probs = np.zeros(128)
        
        # Set basic range
        probs[self.melodic.range_min:self.melodic.range_max+1] = self.melodic.register_weights
        
        # Adjust for current harmony
        if 'current_chord' in context:
            chord_pitches = context['current_chord']
            probs[chord_pitches] *= 2.0
        
        # Adjust for melodic tendencies
        if 'previous_pitch' in context:
            prev_pitch = context['previous_pitch']
            
            # Stepwise motion
            for step in [-2, -1, 1, 2]:
                if self.melodic.range_min <= prev_pitch + step <= self.melodic.range_max:
                    probs[prev_pitch + step] *= self.melodic.step_probability
            
            # Leaps
            for leap in [-7, -5, -4, 4, 5, 7]:
                if self.melodic.range_min <= prev_pitch + leap <= self.melodic.range_max:
                    probs[prev_pitch + leap] *= self.melodic.leap_probability
        
        # Normalize
        return probs / probs.sum()
    
    def _get_duration_probabilities(self, context: Dict) -> np.ndarray:
        """Calculate duration probabilities based on context"""
        probs = np.zeros(len(self.rhythmic.allowed_note_values))
        
        # Basic probabilities based on complexity
        if self.rhythmic.complexity < 0.3:
            # Prefer longer notes
            probs = np.array([0.1, 0.2, 0.4, 0.3])
        elif self.rhythmic.complexity < 0.7:
            # Balanced distribution
            probs = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            # Prefer shorter notes
            probs = np.array([0.4, 0.3, 0.2, 0.1])
        
        return probs
    
    def _get_velocity_probabilities(self, context: Dict) -> np.ndarray:
        """Calculate velocity probabilities based on context"""
        velocities = np.arange(self.expression.dynamics_range[0],
                             self.expression.dynamics_range[1] + 1)
        
        # Center around current dynamic level
        if 'current_dynamic' in context:
            center = context['current_dynamic']
            variance = (self.expression.dynamics_range[1] - 
                       self.expression.dynamics_range[0]) * self.expression.dynamics_variation
            
            probs = np.exp(-(velocities - center)**2 / (2 * variance**2))
            return probs / probs.sum()
        
        # Default to middle of range
        return np.ones_like(velocities) / len(velocities)