# melodia/model/architecture.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Optional, Tuple
import numpy as np
from ..config import ModelConfig

class MultiHeadAttention(layers.Layer):
    """Custom multi-head attention with relative position encoding"""
    
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = config.num_heads
        self.key_dim = config.embedding_dim // config.num_heads
        
        self.query_dense = layers.Dense(config.embedding_dim)
        self.key_dense = layers.Dense(config.embedding_dim)
        self.value_dense = layers.Dense(config.embedding_dim)
        self.combine_heads = layers.Dense(config.embedding_dim)
        
        # Relative position embeddings
        self.rel_pos_embedding = self.add_weight(
            name="rel_pos_embedding",
            shape=(2 * config.max_sequence_length - 1, self.key_dim),
            initializer="random_normal",
            trainable=True
        )
        
    def _rel_shift(self, x: tf.Tensor) -> tf.Tensor:
        """Compute relative positional attention"""
        batch_size, num_heads, seq_len, _ = tf.shape(x)
        x = tf.reshape(x, [batch_size, num_heads, seq_len, seq_len])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [batch_size, num_heads, seq_len + 1, seq_len])
        x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, [batch_size, num_heads, seq_len, seq_len])
        return x
    
    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        
        # Linear transformations
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # Split heads
        query = tf.reshape(
            query,
            (batch_size, -1, self.num_heads, self.key_dim)
        )
        key = tf.reshape(
            key,
            (batch_size, -1, self.num_heads, self.key_dim)
        )
        value = tf.reshape(
            value,
            (batch_size, -1, self.num_heads, self.key_dim)
        )
        
        # Transpose for attention
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attention = tf.matmul(query, key, transpose_b=True) / scale
        
        # Add relative position bias
        position_ids = tf.range(seq_len)[:, None] - tf.range(seq_len)[None, :]
        position_ids = position_ids + seq_len - 1
        rel_pos_bias = tf.gather(self.rel_pos_embedding, position_ids)
        rel_pos_bias = self._rel_shift(rel_pos_bias)
        attention = attention + rel_pos_bias
        
        if mask is not None:
            attention += (mask * -1e9)
        
        attention = tf.nn.softmax(attention, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention, value)
        
        # Combine heads
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.num_heads * self.key_dim))
        output = self.combine_heads(output)
        
        return output

class TransformerBlock(layers.Layer):
    """Transformer block with relative position encoding"""
    
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(config)
        self.ffn = tf.keras.Sequential([
            layers.Dense(config.ff_dim, activation="gelu"),
            layers.Dense(config.embedding_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(config.dropout_rate)
        self.dropout2 = layers.Dropout(config.dropout_rate)
    
    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        # Self-attention
        attention_output = self.attention(inputs, inputs, inputs, mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class MelodiaModel(Model):
    """Main Melodia model architecture"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = layers.Embedding(
            config.vocab_size,
            config.embedding_dim
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ]
        
        # Output layer
        self.output_layer = layers.Dense(config.vocab_size)
        
        # Loss tracker
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        x = self.token_embedding(inputs)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        return self.output_layer(x)
    
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> dict:
        """Custom training step"""
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Apply gradient clipping
        gradients, _ = tf.clip_by_global_norm(
            gradients,
            self.config.gradient_clip_val
        )
        
        # Update weights
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}