# melodia/model/attention.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple, Union
from ..config import ModelConfig

class RelativePositionEmbedding(layers.Layer):
    """Relative positional embeddings for attention"""
    
    def __init__(self, max_length: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Create relative position embeddings matrix
        self.rel_pos_emb = self.add_weight(
            name="rel_pos_emb",
            shape=(2 * max_length - 1, embedding_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        # Create relative position bias table
        self.rel_pos_bias = self.add_weight(
            name="rel_pos_bias",
            shape=(2 * max_length - 1,),
            initializer="zeros",
            trainable=True
        )
    
    def call(self, seq_length: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute relative position embeddings and bias for given sequence length"""
        # Create position ids matrix
        pos_ids = tf.range(seq_length)[:, None] - tf.range(seq_length)[None, :]
        pos_ids = pos_ids + self.max_length - 1
        
        # Get relative position embeddings
        rel_pos_emb = tf.gather(self.rel_pos_emb, pos_ids)
        rel_pos_bias = tf.gather(self.rel_pos_bias, pos_ids)
        
        return rel_pos_emb, rel_pos_bias

class MusicAttention(layers.Layer):
    """Music-aware multi-head attention with relative position encoding"""
    
    def __init__(
        self,
        config: ModelConfig,
        use_relative_attention: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = config.num_heads
        self.key_dim = config.embedding_dim // config.num_heads
        self.dropout_rate = config.dropout_rate
        self.use_relative_attention = use_relative_attention
        
        # Linear projections
        self.query_dense = layers.Dense(config.embedding_dim)
        self.key_dense = layers.Dense(config.embedding_dim)
        self.value_dense = layers.Dense(config.embedding_dim)
        self.combine_heads = layers.Dense(config.embedding_dim)
        
        # Relative position encoding
        if use_relative_attention:
            self.relative_position = RelativePositionEmbedding(
                config.max_sequence_length,
                self.key_dim
            )
        
        # Dropout
        self.dropout = layers.Dropout(config.dropout_rate)
    
    def split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth)"""
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        
        # Reshape to (batch_size, seq_length, num_heads, depth)
        x = tf.reshape(x, (batch_size, length, self.num_heads, self.key_dim))
        
        # Transpose to (batch_size, num_heads, seq_length, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def _relative_shift(self, x: tf.Tensor) -> tf.Tensor:
        """Compute relative positional attention"""
        batch_size, num_heads, seq_length = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        # Pad the input
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [batch_size, num_heads, seq_length + 1, seq_length])
        x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
        
        return x
    
    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        return_attention: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Apply music-aware attention with relative position encoding"""
        batch_size = tf.shape(query)[0]
        seq_length = tf.shape(query)[1]
        
        # Linear transformations and split heads
        query = self.split_heads(self.query_dense(query))
        key = self.split_heads(self.key_dense(key))
        value = self.split_heads(self.value_dense(value))
        
        # Scale query
        depth = tf.cast(self.key_dim, tf.float32)
        query = query / tf.math.sqrt(depth)
        
        # Compute raw attention scores
        attention_scores = tf.matmul(query, key, transpose_b=True)
        
        # Add relative position embeddings if enabled
        if self.use_relative_attention:
            rel_pos_emb, rel_pos_bias = self.relative_position(seq_length)
            rel_pos_att = tf.matmul(query, tf.transpose(rel_pos_emb, [0, 2, 1]))
            rel_pos_att = self._relative_shift(rel_pos_att)
            attention_scores = attention_scores + rel_pos_att + rel_pos_bias
        
        # Apply mask if provided
        if mask is not None:
            attention_scores += (mask * -1e9)
        
        # Apply softmax and dropout
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, value)
        
        # Restore shape and combine heads
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, seq_length, -1))
        output = self.combine_heads(output)
        
        if return_attention:
            return output, attention_weights
        return output

class LocalAttention(MusicAttention):
    """Attention layer with local windowing for long sequences"""
    
    def __init__(
        self,
        config: ModelConfig,
        window_size: int = 256,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.window_size = window_size
    
    def create_local_mask(self, seq_length: int) -> tf.Tensor:
        """Create a mask for local attention windowing"""
        # Create position matrix
        pos = tf.range(seq_length)
        pos_diff = pos[:, None] - pos[None, :]
        
        # Create window mask
        window_mask = tf.abs(pos_diff) <= self.window_size
        return tf.cast(window_mask, tf.float32)
    
    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        return_attention: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Apply local attention with windowing"""
        # Create local attention mask
        local_mask = self.create_local_mask(tf.shape(query)[1])
        
        # Combine with provided mask if any
        if mask is not None:
            local_mask = local_mask * mask
        
        return super().call(
            query=query,
            key=key,
            value=value,
            mask=local_mask,
            return_attention=return_attention
        )

class HierarchicalAttention(layers.Layer):
    """Hierarchical attention for capturing both local and global patterns"""
    
    def __init__(
        self,
        config: ModelConfig,
        num_levels: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_levels = num_levels
        
        # Create attention layers for each level
        self.attention_levels = [
            MusicAttention(config)
            for _ in range(num_levels)
        ]
        
        # Projections for hierarchical processing
        self.level_projections = [
            layers.Dense(config.embedding_dim)
            for _ in range(num_levels - 1)
        ]
    
    def downsample(self, x: tf.Tensor, level: int) -> tf.Tensor:
        """Downsample sequence for hierarchical processing"""
        # Average pooling with stride 2^level
        pool_size = 2 ** level
        return tf.nn.avg_pool1d(
            x,
            ksize=pool_size,
            strides=pool_size,
            padding='VALID'
        )
    
    def upsample(self, x: tf.Tensor, target_length: int) -> tf.Tensor:
        """Upsample sequence back to original length"""
        return tf.image.resize(
            x[:, :, None],
            (target_length, 1),
            method='nearest'
        )[:, :, 0]
    
    def call(
        self,
        x: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Apply hierarchical attention"""
        original_length = tf.shape(x)[1]
        outputs = []
        
        # Process each level
        current_input = x
        for level in range(self.num_levels):
            # Apply attention at current level
            level_output = self.attention_levels[level](
                current_input,
                current_input,
                current_input,
                mask=mask
            )
            outputs.append(level_output)
            
            # Prepare for next level if not last
            if level < self.num_levels - 1:
                # Downsample
                current_input = self.downsample(current_input, level + 1)
                # Project
                current_input = self.level_projections[level](current_input)
        
        # Combine outputs from all levels
        final_output = outputs[0]
        for level in range(1, self.num_levels):
            level_output = outputs[level]
            # Upsample to original length
            level_output = self.upsample(level_output, original_length)
            # Add to final output
            final_output = final_output + level_output
        
        return final_output