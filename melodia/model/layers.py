# melodia/model/layers.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple, List, Union
from ..config import ModelConfig
from .attention import MusicAttention, HierarchicalAttention

class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding layer"""
    
    def __init__(self, max_length: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.pos_encoding = self.create_positional_encoding()
    
    def create_positional_encoding(self) -> tf.Tensor:
        """Create sinusoidal position encoding table"""
        positions = np.arange(self.max_length)[:, np.newaxis]
        dims = np.arange(self.embedding_dim)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / self.embedding_dim)
        
        # Apply sin to even indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        # Apply cos to odd indices
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Add positional encoding to input"""
        seq_length = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_length, :]

class RhythmAwareEmbedding(layers.Layer):
    """Embedding layer that incorporates rhythmic structure"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            mask_zero=True
        )
        self.positional_encoding = PositionalEncoding(
            max_length,
            embedding_dim
        )
        
        # Rhythm-specific embeddings
        self.beat_embedding = layers.Embedding(
            4,  # Four possible positions in a beat
            embedding_dim
        )
        self.bar_embedding = layers.Embedding(
            16,  # Sixteen possible positions in a bar
            embedding_dim
        )
    
    def compute_rhythm_position(self, seq_length: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute position within beat and bar"""
        positions = tf.range(seq_length)
        beat_positions = tf.math.mod(positions, 4)  # Position within beat
        bar_positions = tf.math.mod(positions, 16)  # Position within bar
        return beat_positions, bar_positions
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Combine token, position, and rhythm embeddings"""
        seq_length = tf.shape(x)[1]
        
        # Get basic embeddings
        token_emb = self.token_embedding(x)
        
        # Add positional encoding
        pos_emb = self.positional_encoding(token_emb)
        
        # Add rhythm embeddings
        beat_pos, bar_pos = self.compute_rhythm_position(seq_length)
        rhythm_emb = (
            self.beat_embedding(beat_pos) +
            self.bar_embedding(bar_pos)
        )
        
        return pos_emb + rhythm_emb

class StructureAwareTransformer(layers.Layer):
    """Transformer layer with awareness of musical structure"""
    
    def __init__(
        self,
        config: ModelConfig,
        use_hierarchical: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Attention mechanism
        self.attention = (
            HierarchicalAttention(config)
            if use_hierarchical
            else MusicAttention(config)
        )
        
        # Processing layers
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(config.ff_dim, activation="gelu"),
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.embedding_dim)
        ])
        
        # Dropouts
        self.dropout1 = layers.Dropout(config.dropout_rate)
        self.dropout2 = layers.Dropout(config.dropout_rate)
    
    def call(
        self,
        x: tf.Tensor,
        training: bool,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Apply transformer layer with structure awareness"""
        # Multi-head attention
        attn_output = self.attention(x, x, x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class StyleConditioningLayer(layers.Layer):
    """Layer for conditioning generation on musical style"""
    
    def __init__(
        self,
        config: ModelConfig,
        num_styles: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.style_embedding = layers.Embedding(
            num_styles,
            config.embedding_dim
        )
        self.style_projection = layers.Dense(config.embedding_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(
        self,
        x: tf.Tensor,
        style_ids: tf.Tensor
    ) -> tf.Tensor:
        """Apply style conditioning"""
        # Get style embeddings
        style_emb = self.style_embedding(style_ids)
        
        # Project to the same dimension as input
        style_features = self.style_projection(style_emb)
        
        # Add style information
        # Broadcast style features to match sequence length
        style_features = style_features[:, None, :]
        style_features = tf.tile(
            style_features,
            [1, tf.shape(x)[1], 1]
        )
        
        # Combine with input
        return self.layer_norm(x + style_features)

class MusicalMemory(layers.Layer):
    """Long-term memory layer for capturing musical patterns"""
    
    def __init__(
        self,
        config: ModelConfig,
        memory_size: int = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = config.embedding_dim
        
        # Memory components
        self.memory = self.add_weight(
            name="memory_bank",
            shape=(memory_size, self.memory_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        # Memory attention
        self.memory_attention = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.embedding_dim // config.num_heads
        )
        
        self.output_layer = layers.Dense(config.embedding_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(
        self,
        x: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Apply memory attention mechanism"""
        # Query the memory using the input
        memory_output = self.memory_attention(
            query=x,
            key=self.memory,
            value=self.memory,
            training=training
        )
        
        # Process memory output
        memory_output = self.output_layer(memory_output)
        
        # Combine with input
        return self.layer_norm(x + memory_output)

class ChordConditioningLayer(layers.Layer):
    """Layer for conditioning on chord progressions"""
    
    def __init__(
        self,
        config: ModelConfig,
        num_chords: int = 24,  # Major and minor chords
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chord_embedding = layers.Embedding(
            num_chords,
            config.embedding_dim
        )
        self.chord_attention = MusicAttention(config)
        self.output_projection = layers.Dense(config.embedding_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(
        self,
        x: tf.Tensor,
        chord_ids: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """Apply chord conditioning"""
        # Embed chords
        chord_emb = self.chord_embedding(chord_ids)
        
        # Apply attention between input and chord embeddings
        chord_context = self.chord_attention(
            x, chord_emb, chord_emb,
            training=training
        )
        
        # Project and combine
        chord_features = self.output_projection(chord_context)
        return self.layer_norm(x + chord_features)