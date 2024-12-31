# tests/test_evaluation.py

import pytest
import numpy as np
from melodia.evaluation.metrics import MusicEvaluator
from melodia.evaluation.analyzer import MusicTheoryAnalyzer, StructureAnalyzer
from melodia.data.loader import MusicEvent
import music21

@pytest.fixture
def sample_events():
    """Create sample music events for testing"""
    return [
        MusicEvent(type='note', time=0.0, pitch=60, velocity=64, duration=1.0),  # C4
        MusicEvent(type='note', time=1.0, pitch=64, velocity=64, duration=1.0),  # E4
        MusicEvent(type='note', time=2.0, pitch=67, velocity=64, duration=1.0),  # G4
        MusicEvent(type='note', time=3.0, pitch=72, velocity=64, duration=1.0),  # C5
        MusicEvent(type='tempo', time=0.0, tempo=120),
        MusicEvent(type='time_signature', time=0.0, numerator=4, denominator=4),
        MusicEvent(type='key_signature', time=0.0, key='C', mode='major')
    ]

@pytest.fixture
def evaluator():
    return MusicEvaluator()

@pytest.fixture
def theory_analyzer():
    return MusicTheoryAnalyzer()

@pytest.fixture
def structure_analyzer():
    return StructureAnalyzer()

def test_basic_evaluation(evaluator, sample_events):
    """Test basic evaluation metrics"""
    metrics = evaluator.evaluate(sample_events)
    
    # Check for required metric categories
    assert 'note_density' in metrics
    assert 'pitch_range' in metrics
    assert 'rhythmic_diversity' in metrics
    assert 'harmonic_consistency' in metrics
    
    # Check metric values are in valid ranges
    assert 0 <= metrics['note_density'] <= 100
    assert metrics['pitch_range'] == 12  # C4 to C5 = 12 semitones

def test_harmonic_analysis(theory_analyzer, sample_events):
    """Test harmonic analysis"""
    analysis = theory_analyzer.analyze_harmony(sample_events)
    
    # Check key detection
    assert analysis['key']['root'] == 'C'
    assert analysis['key']['mode'] == 'major'
    
    # Check chord progression
    assert len(analysis['chord_progression']) > 0
    
    # Check cadences
    assert 'cadences' in analysis
    
    # Check scale degrees
    scale_degrees = theory_analyzer.get_scale_degrees(sample_events)
    assert 1 in scale_degrees  # Tonic (C)
    assert 3 in scale_degrees  # Mediant (E)
    assert 5 in scale_degrees  # Dominant (G)

def test_structure_analysis(structure_analyzer, sample_events):
    """Test structural analysis"""
    analysis = structure_analyzer.analyze_structure(sample_events)
    
    # Check phrase detection
    assert 'phrases' in analysis
    
    # Check motif detection
    assert 'motifs' in analysis
    assert isinstance(analysis['motifs'], list)
    
    # Check repetition patterns
    assert 'repetition_patterns' in analysis
    
    # Check form analysis
    assert 'form' in analysis

def test_rhythm_metrics(evaluator, sample_events):
    """Test rhythm-specific metrics"""
    metrics = evaluator.compute_rhythmic_metrics(sample_events)
    
    assert 'rhythmic_complexity' in metrics
    assert 'syncopation' in metrics
    assert 'groove_strength' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['rhythmic_complexity'] <= 1
    assert 0 <= metrics['syncopation'] <= 1
    assert 0 <= metrics['groove_strength'] <= 1

def test_melodic_metrics(evaluator, sample_events):
    """Test melody-specific metrics"""
    metrics = evaluator.compute_melodic_metrics(sample_events)
    
    assert 'melodic_contour' in metrics
    assert 'pitch_variety' in metrics
    assert 'interval_distribution' in metrics
    
    # Check contour values
    assert isinstance(metrics['melodic_contour'], dict)
    assert all(0 <= v <= 1 for v in metrics['melodic_contour'].values())

def test_harmonic_metrics(evaluator, sample_events):
    """Test harmony-specific metrics"""
    metrics = evaluator.compute_harmonic_metrics(sample_events)
    
    assert 'key_strength' in metrics
    assert 'chord_progression_complexity' in metrics
    assert 'tonal_stability' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['key_strength'] <= 1
    assert 0 <= metrics['tonal_stability'] <= 1

def test_style_consistency(evaluator, sample_events):
    """Test style consistency metrics"""
    # Create reference sequence
    reference_events = sample_events.copy()
    
    metrics = evaluator.compute_style_metrics(sample_events, reference_events)
    
    assert 'style_similarity' in metrics
    assert 'rhythm_similarity' in metrics
    assert 'harmony_similarity' in metrics
    
    # Check similarity ranges
    assert all(0 <= v <= 1 for v in metrics.values())

def test_structural_coherence(evaluator, sample_events):
    """Test structural coherence metrics"""
    metrics = evaluator.compute_structure_metrics(sample_events)
    
    assert 'phrase_regularity' in metrics
    assert 'formal_clarity' in metrics
    assert 'sectional_balance' in metrics
    
    # Check metric ranges
    assert all(0 <= v <= 1 for v in metrics.values())

def test_tension_profile(evaluator, sample_events):
    """Test musical tension analysis"""
    tension = evaluator.compute_tension_profile(sample_events)
    
    assert 'harmonic_tension' in tension
    assert 'rhythmic_tension' in tension
    assert 'melodic_tension' in tension
    assert 'overall_tension' in tension
    
    # Check tension ranges
    assert all(0 <= v <= 1 for v in tension.values())

def test_error_handling(evaluator):
    """Test error handling in evaluation"""
    # Test empty sequence
    with pytest.raises(ValueError):
        evaluator.evaluate([])
    
    # Test invalid events
    invalid_events = [
        MusicEvent(type='invalid', time=0.0)
    ]
    with pytest.raises(ValueError):
        evaluator.evaluate(invalid_events)

def test_comparative_evaluation(evaluator, sample_events):
    """Test comparative evaluation between sequences"""
    sequence1 = sample_events
    sequence2 = sample_events.copy()
    sequence2[0].pitch += 2  # Slightly modify second sequence
    
    comparison = evaluator.compare_sequences(sequence1, sequence2)
    
    assert 'similarity_score' in comparison
    assert 'differences' in comparison
    assert 'shared_features' in comparison

def test_perceptual_metrics(evaluator, sample_events):
    """Test perceptual music metrics"""
    metrics = evaluator.compute_perceptual_metrics(sample_events)
    
    assert 'complexity' in metrics
    assert 'consonance' in metrics
    assert 'energy' in metrics
    assert 'brightness' in metrics
    
    # Check metric ranges
    assert all(0 <= v <= 1 for v in metrics.values())

def test_batch_evaluation(evaluator, sample_events):
    """Test batch evaluation capabilities"""
    sequences = [sample_events, sample_events.copy()]
    batch_metrics = evaluator.evaluate_batch(sequences)
    
    assert len(batch_metrics) == len(sequences)
    assert all(isinstance(m, dict) for m in batch_metrics)

def test_metric_aggregation(evaluator, sample_events):
    """Test metric aggregation functionality"""
    # Evaluate multiple sections
    section_metrics = []
    for i in range(3):
        section = sample_events.copy()
        metrics = evaluator.evaluate(section)
        section_metrics.append(metrics)
    
    # Aggregate metrics
    aggregated = evaluator.aggregate_metrics(section_metrics)
    
    assert 'mean' in aggregated
    assert 'std' in aggregated
    assert 'min' in aggregated
    assert 'max' in aggregated