# tests/test_generation.py

import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import Mock, patch
from melodia.generation.generator import MusicGenerator
from melodia.generation.controls import (
    GenerationControls,
    HarmonicControls,
    RhythmicControls,
    MelodicControls,
    StructuralControls,
    ExpressionControls,
    MusicalStyle
)
from melodia.config import GenerationConfig, ModelConfig
from melodia.data.loader import MusicEvent

@pytest.fixture
def generation_config():
    return GenerationConfig(
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_length=512
    )

@pytest.fixture
def model_config():
    return ModelConfig(
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        vocab_size=1024
    )

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    # Mock predict method to return random logits
    model.predict.return_value = tf.random.normal((1, 1, 1024))
    return model

@pytest.fixture
def generator(generation_config, mock_model):
    return MusicGenerator(mock_model, generation_config)

@pytest.fixture
def controls(generation_config):
    return GenerationControls(generation_config)

def test_generator_initialization(generator, generation_config):
    """Test generator initialization"""
    assert isinstance(generator, MusicGenerator)
    assert generator.config.temperature == generation_config.temperature
    assert generator.config.top_k == generation_config.top_k
    assert generator.config.top_p == generation_config.top_p

def test_basic_generation(generator):
    """Test basic music generation without controls"""
    events = generator.generate(max_length=64)
    assert isinstance(events, list)
    assert len(events) > 0
    assert all(isinstance(e, MusicEvent) for e in events)

def test_controlled_generation(generator, controls):
    """Test generation with specific controls"""
    # Configure controls
    controls.harmonic.key = "C"
    controls.harmonic.mode = "major"
    controls.rhythmic.tempo = 120
    controls.rhythmic.time_signature = (4, 4)
    
    events = generator.generate(
        max_length=64,
        controls=controls
    )
    
    assert len(events) > 0
    # Verify key and tempo were applied
    assert any(e.type == 'key_signature' and 'C' in e.key 
              for e in events)
    assert any(e.type == 'tempo' and e.tempo == 120 
              for e in events)

def test_temperature_sampling(generator):
    """Test temperature effects on generation"""
    # Generate with different temperatures
    events_high_temp = generator.generate(
        max_length=32,
        temperature=1.5
    )
    
    events_low_temp = generator.generate(
        max_length=32,
        temperature=0.2
    )
    
    # While we can't deterministically test randomness,
    # we can verify basic properties
    assert len(events_high_temp) > 0
    assert len(events_low_temp) > 0
    assert events_high_temp != events_low_temp

def test_top_k_sampling(generator):
    """Test top-k sampling mechanism"""
    k_values = [1, 10, 50]
    generated_sequences = []
    
    for k in k_values:
        events = generator.generate(
            max_length=32,
            top_k=k
        )
        generated_sequences.append(events)
        assert len(events) > 0
    
    # Sequences should be different
    assert len(set(str(seq) for seq in generated_sequences)) > 1

def test_top_p_sampling(generator):
    """Test nucleus (top-p) sampling"""
    p_values = [0.5, 0.9]
    generated_sequences = []
    
    for p in p_values:
        events = generator.generate(
            max_length=32,
            top_p=p
        )
        generated_sequences.append(events)
        assert len(events) > 0
    
    # Sequences should be different
    assert len(set(str(seq) for seq in generated_sequences)) > 1

def test_repetition_penalty(generator):
    """Test repetition penalty mechanism"""
    # Generate with different penalties
    events_high_penalty = generator.generate(
        max_length=64,
        repetition_penalty=2.0
    )
    
    events_low_penalty = generator.generate(
        max_length=64,
        repetition_penalty=1.0
    )
    
    assert len(events_high_penalty) > 0
    assert len(events_low_penalty) > 0

def test_harmonic_controls(controls):
    """Test harmonic control settings"""
    controls.harmonic.key = "F#"
    controls.harmonic.mode = "minor"
    controls.harmonic.chord_progression = ["i", "iv", "v"]
    
    assert controls.harmonic.key == "F#"
    assert controls.harmonic.mode == "minor"
    assert len(controls.harmonic.chord_progression) == 3

def test_rhythmic_controls(controls):
    """Test rhythmic control settings"""
    controls.rhythmic.tempo = 140
    controls.rhythmic.time_signature = (6, 8)
    controls.rhythmic.swing = 0.5
    
    assert controls.rhythmic.tempo == 140
    assert controls.rhythmic.time_signature == (6, 8)
    assert controls.rhythmic.swing == 0.5

def test_melodic_controls(controls):
    """Test melodic control settings"""
    controls.melodic.range_min = 48  # C3
    controls.melodic.range_max = 72  # C5
    controls.melodic.step_probability = 0.8
    
    assert controls.melodic.range_min == 48
    assert controls.melodic.range_max == 72
    assert controls.melodic.step_probability == 0.8

def test_structural_controls(controls):
    """Test structural control settings"""
    controls.structural.form = "AABA"
    controls.structural.phrase_length = 4
    
    assert controls.structural.form == "AABA"
    assert controls.structural.phrase_length == 4

def test_style_based_generation(generator):
    """Test generation with different musical styles"""
    for style in MusicalStyle:
        events = generator.generate(
            max_length=32,
            style=style
        )
        assert len(events) > 0

def test_continuation_generation(generator):
    """Test generating continuation of existing music"""
    # Create seed sequence
    seed_events = [
        MusicEvent(type='note', time=0.0, pitch=60, velocity=64, duration=1.0),
        MusicEvent(type='note', time=1.0, pitch=62, velocity=64, duration=1.0)
    ]
    
    continued_events = generator.continue_sequence(
        seed_events,
        additional_length=32
    )
    
    assert len(continued_events) > len(seed_events)
    # Verify continuation starts where seed ends
    assert continued_events[:len(seed_events)] == seed_events

def test_generator_error_handling(generator):
    """Test error handling in generation"""
    # Test invalid length
    with pytest.raises(ValueError):
        generator.generate(max_length=-1)
    
    # Test invalid temperature
    with pytest.raises(ValueError):
        generator.generate(max_length=32, temperature=0)
    
    # Test invalid top_k
    with pytest.raises(ValueError):
        generator.generate(max_length=32, top_k=-1)
    
    # Test invalid top_p
    with pytest.raises(ValueError):
        generator.generate(max_length=32, top_p=1.5)

def test_generation_with_constraints(generator, controls):
    """Test generation with specific constraints"""
    # Set up constraints
    controls.harmonic.allowed_chords = ["maj", "min"]
    controls.melodic.preferred_intervals = [0, 2, 4, 5, 7]
    
    events = generator.generate(
        max_length=32,
        controls=controls
    )
    
    assert len(events) > 0
    # Verify note events follow constraints
    note_events = [e for e in events if e.type == 'note']
    assert len(note_events) > 0