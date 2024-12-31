# tests/test_data.py

import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
from melodia.data.loader import MIDILoader, MusicEvent
from melodia.data.processor import DataProcessor
from melodia.data.tokenizer import MusicTokenizer
from melodia.config import DataConfig

@pytest.fixture
def data_config():
    return DataConfig(
        max_sequence_length=512,
        min_sequence_length=64,
        valid_time_signatures=[(4, 4), (3, 4), (6, 8)]
    )

@pytest.fixture
def midi_loader():
    return MIDILoader()

@pytest.fixture
def data_processor(data_config):
    return DataProcessor(data_config)

@pytest.fixture
def tokenizer(data_config):
    return MusicTokenizer(data_config)

@pytest.fixture
def sample_events():
    """Create sample music events for testing"""
    return [
        MusicEvent(type='note', time=0.0, pitch=60, velocity=64, duration=1.0),
        MusicEvent(type='note', time=1.0, pitch=62, velocity=64, duration=1.0),
        MusicEvent(type='note', time=2.0, pitch=64, velocity=64, duration=1.0),
        MusicEvent(type='tempo', time=0.0, tempo=120),
        MusicEvent(type='time_signature', time=0.0, numerator=4, denominator=4)
    ]

def test_music_event_creation():
    """Test MusicEvent creation and properties"""
    event = MusicEvent(
        type='note',
        time=1.0,
        pitch=60,
        velocity=64,
        duration=1.0
    )
    assert event.type == 'note'
    assert event.time == 1.0
    assert event.pitch == 60
    assert event.velocity == 64
    assert event.duration == 1.0

def test_midi_loader_read_write(midi_loader, sample_events, tmp_path):
    """Test MIDI file reading and writing"""
    # Write MIDI file
    test_file = tmp_path / "test.mid"
    midi_loader.write_midi(sample_events, test_file)
    assert test_file.exists()
    
    # Read MIDI file
    loaded_events = midi_loader.read_midi(test_file)
    assert len(loaded_events) > 0
    
    # Check event properties are preserved
    assert loaded_events[0].type == sample_events[0].type
    assert loaded_events[0].pitch == sample_events[0].pitch
    assert np.isclose(loaded_events[0].time, sample_events[0].time)

def test_tokenizer_encode_decode(tokenizer, sample_events):
    """Test tokenization and detokenization"""
    # Encode events
    tokens = tokenizer.encode_events(sample_events)
    assert len(tokens) > 0
    assert tokens[0] == tokenizer.BOS_TOKEN
    assert tokens[-1] == tokenizer.EOS_TOKEN
    
    # Decode tokens
    decoded_events = tokenizer.decode_tokens(tokens)
    assert len(decoded_events) > 0
    
    # Check event properties are preserved
    assert decoded_events[0].type == sample_events[0].type
    assert decoded_events[0].pitch == sample_events[0].pitch

def test_data_processor_sequence_preparation(data_processor, sample_events):
    """Test sequence preparation"""
    # Create sequences
    sequences = data_processor.prepare_sequences([sample_events])
    assert len(sequences) > 0
    
    # Check sequence properties
    assert all(len(seq) >= data_processor.config.min_sequence_length
              for seq in sequences)
    assert all(len(seq) <= data_processor.config.max_sequence_length
              for seq in sequences)

def test_data_processor_augmentation(data_processor, sample_events):
    """Test data augmentation"""
    # Test pitch transposition
    transposed = data_processor.augment_pitch(sample_events, semitones=2)
    assert len(transposed) == len(sample_events)
    assert transposed[0].pitch == sample_events[0].pitch + 2
    
    # Test time stretching
    stretched = data_processor.augment_time(sample_events, factor=2.0)
    assert len(stretched) == len(sample_events)
    assert stretched[0].duration == sample_events[0].duration * 2.0
    
    # Test velocity scaling
    scaled = data_processor.augment_velocity(sample_events, factor=0.8)
    assert len(scaled) == len(sample_events)
    assert scaled[0].velocity == int(sample_events[0].velocity * 0.8)

def test_dataset_creation(data_processor, sample_events):
    """Test dataset creation and properties"""
    # Create dataset
    dataset = data_processor.create_dataset(
        [sample_events],
        batch_size=2,
        shuffle=True
    )
    
    # Check dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    
    # Check batch structure
    for batch in dataset.take(1):
        assert len(batch) == 2  # (input_ids, target_ids)
        assert isinstance(batch[0], tf.Tensor)
        assert isinstance(batch[1], tf.Tensor)
        assert batch[0].shape[0] == 2  # batch size
        
def test_sequence_padding(data_processor, sample_events):
    """Test sequence padding"""
    # Create short sequence
    short_sequence = sample_events[:2]
    
    # Pad sequence
    padded = data_processor.pad_sequence(
        short_sequence,
        max_length=data_processor.config.max_sequence_length
    )
    
    # Check padding
    assert len(padded) == data_processor.config.max_sequence_length
    assert padded[-1] == data_processor.tokenizer.PAD_TOKEN
    
    # Check original content is preserved
    tokens = data_processor.tokenizer.encode_events(short_sequence)
    assert all(a == b for a, b in zip(tokens, padded[:len(tokens)]))

def test_data_processor_validation(data_processor):
    """Test data validation"""
    # Test valid events
    valid_events = [
        MusicEvent(type='note', time=0.0, pitch=60, velocity=64, duration=1.0),
        MusicEvent(type='note', time=1.0, pitch=62, velocity=64, duration=1.0)
    ]
    assert data_processor.is_valid_sequence(valid_events)
    
    # Test invalid events (empty)
    assert not data_processor.is_valid_sequence([])
    
    # Test invalid events (wrong type)
    invalid_events = [
        MusicEvent(type='invalid', time=0.0, pitch=60, velocity=64, duration=1.0)
    ]
    assert not data_processor.is_valid_sequence(invalid_events)

def test_data_processor_batch_processing(data_processor, sample_events):
    """Test batch processing"""
    # Create multiple sequences
    sequences = [sample_events, sample_events[:2]]
    
    # Process batch
    batch = data_processor.process_batch(sequences)
    assert isinstance(batch, dict)
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    
    # Check batch dimensions
    assert batch['input_ids'].shape[0] == len(sequences)
    assert batch['attention_mask'].shape[0] == len(sequences)

def test_data_processor_save_load(data_processor, tmp_path):
    """Test saving and loading processor state"""
    # Save processor
    save_path = tmp_path / "processor_state"
    data_processor.save(save_path)
    
    # Load processor
    loaded_processor = DataProcessor.load(save_path)
    
    # Check configurations match
    assert loaded_processor.config.max_sequence_length == data_processor.config.max_sequence_length
    assert loaded_processor.tokenizer.vocab_size == data_processor.tokenizer.vocab_size
    
    # Check tokenization produces same results
    test_events = [
        MusicEvent(type='note', time=0.0, pitch=60, velocity=64, duration=1.0)
    ]
    original_tokens = data_processor.tokenizer.encode_events(test_events)
    loaded_tokens = loaded_processor.tokenizer.encode_events(test_events)
    assert original_tokens == loaded_tokens

def test_data_processor_streaming(data_processor, tmp_path):
    """Test streaming dataset creation"""
    # Create test files
    test_files = []
    for i in range(3):
        file_path = tmp_path / f"test_{i}.mid"
        midi_loader = MIDILoader()
        midi_loader.write_midi([
            MusicEvent(type='note', time=float(i), pitch=60+i, 
                      velocity=64, duration=1.0)
        ], file_path)
        test_files.append(file_path)
    
    # Create streaming dataset
    dataset = data_processor.create_streaming_dataset(
        test_files,
        batch_size=2,
        shuffle=True
    )
    
    # Check dataset properties
    assert isinstance(dataset, tf.data.Dataset)
    
    # Verify we can iterate through it
    batches = list(dataset.take(2))
    assert len(batches) > 0

def test_sequence_filtering(data_processor, sample_events):
    """Test sequence filtering"""
    # Create sequences of different lengths
    sequences = [
        sample_events,
        sample_events[:1],  # Too short
        sample_events * 100  # Too long
    ]
    
    # Filter sequences
    filtered = data_processor.filter_sequences(sequences)
    
    # Check filtering
    assert len(filtered) == 1
    assert len(filtered[0]) == len(sample_events)

def test_data_processor_error_handling(data_processor):
    """Test error handling"""
    # Test invalid event type
    with pytest.raises(ValueError):
        data_processor.process_event(
            MusicEvent(type='invalid', time=0.0)
        )
    
    # Test invalid sequence length
    with pytest.raises(ValueError):
        data_processor.prepare_sequences(
            [],
            max_length=-1
        )
    
    # Test invalid batch size
    with pytest.raises(ValueError):
        data_processor.create_dataset(
            [sample_events],
            batch_size=0
        )