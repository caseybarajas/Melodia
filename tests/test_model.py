# tests/test_model.py

import pytest
import tensorflow as tf
import numpy as np
from melodia.model.architecture import MelodiaModel
from melodia.config import ModelConfig

@pytest.fixture
def model_config():
    return ModelConfig(
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        dropout_rate=0.1,
        max_sequence_length=512,
        vocab_size=1024
    )

@pytest.fixture
def model(model_config):
    return MelodiaModel(model_config)

def test_model_creation(model):
    """Test model initialization"""
    assert isinstance(model, MelodiaModel)
    assert model.config.embedding_dim == 256
    assert model.config.num_layers == 4

def test_model_forward_pass(model):
    """Test model forward pass"""
    batch_size = 4
    seq_length = 32
    
    # Create dummy input
    inputs = tf.random.uniform(
        (batch_size, seq_length),
        maxval=model.config.vocab_size,
        dtype=tf.int32
    )
    
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Check output shape
    assert outputs.shape == (batch_size, seq_length, model.config.vocab_size)

def test_model_training_step(model):
    """Test model training step"""
    batch_size = 4
    seq_length = 32
    
    # Create dummy batch
    inputs = tf.random.uniform(
        (batch_size, seq_length),
        maxval=model.config.vocab_size,
        dtype=tf.int32
    )
    targets = tf.random.uniform(
        (batch_size, seq_length),
        maxval=model.config.vocab_size,
        dtype=tf.int32
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    
    # Training step
    loss = model.train_on_batch(inputs, targets)
    assert isinstance(loss, float)

def test_model_save_load(model, tmp_path):
    """Test model saving and loading"""
    save_path = tmp_path / "test_model"
    
    # Save model
    model.save(save_path)
    
    # Load model
    loaded_model = tf.keras.models.load_model(save_path)
    
    # Check configurations match
    assert loaded_model.config.embedding_dim == model.config.embedding_dim
    assert loaded_model.config.num_layers == model.config.num_layers

def test_model_attention_mechanism(model):
    """Test attention mechanism"""
    batch_size = 4
    seq_length = 32
    
    # Create input with attention mask
    inputs = tf.random.uniform(
        (batch_size, seq_length),
        maxval=model.config.vocab_size,
        dtype=tf.int32
    )
    mask = tf.ones((batch_size, seq_length))
    
    # Forward pass with attention mask
    outputs = model(inputs, training=False, mask=mask)
    assert outputs.shape == (batch_size, seq_length, model.config.vocab_size)

def test_model_gradient_flow(model):
    """Test gradient flow through model"""
    batch_size = 4
    seq_length = 32
    
    inputs = tf.random.uniform(
        (batch_size, seq_length),
        maxval=model.config.vocab_size,
        dtype=tf.int32
    )
    targets = tf.random.uniform(
        (batch_size, seq_length),
        maxval=model.config.vocab_size,
        dtype=tf.int32
    )
    
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            targets, predictions, from_logits=True
        )
    
    # Check gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in gradients)

def test_model_inference(model):
    """Test model inference"""
    # Test single sequence generation
    seed = tf.random.uniform((1, 16), maxval=model.config.vocab_size, dtype=tf.int32)
    max_length = 32
    
    generated = []
    current_input = seed
    
    for _ in range(max_length - seed.shape[1]):
        predictions = model(current_input, training=False)
        next_token = tf.argmax(predictions[:, -1:], axis=-1)
        current_input = tf.concat([current_input, next_token], axis=1)
        generated.append(next_token[0, 0].numpy())
    
    assert len(generated) == max_length - seed.shape[1]