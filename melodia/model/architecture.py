# melodia/model/architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from ..config import ModelConfig

class MultiHeadAttention(nn.Module):
    """PyTorch implementation of multi-head attention with relative position encoding"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.key_dim = config.embedding_dim // config.num_heads
        self.embedding_dim = config.embedding_dim
        
        self.query_dense = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key_dense = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value_dense = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.combine_heads = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Relative position embeddings - disabled for now for stability
        # self.rel_pos_embedding = nn.Parameter(
        #     torch.randn(2 * config.max_sequence_length - 1, self.key_dim)
        # )
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Linear transformations
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # Split heads
        query = query.view(batch_size, seq_len, self.num_heads, self.key_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.key_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.key_dim)
        
        # Transpose for attention (batch_size, num_heads, seq_len, key_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32, device=query.device))
        attention = torch.matmul(query, key.transpose(-2, -1)) / scale
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, value)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embedding_dim)
        output = self.combine_heads(output)
        
        return output

class TransformerBlock(nn.Module):
    """PyTorch transformer block with layer normalization and dropout"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embedding_dim)
        )
        
        self.layernorm1 = nn.LayerNorm(config.embedding_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(config.embedding_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attention_output = self.attention(inputs, inputs, inputs, mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class MelodiaModel(nn.Module):
    """PyTorch implementation of the Melodia model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=0  # Assuming 0 is the PAD token
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Token embeddings
        x = self.token_embedding(inputs)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=mask)
        
        # Output projection
        logits = self.output_layer(x)
        
        return logits
    
    def get_device(self):
        """Get the device the model is on"""
        return next(self.parameters()).device

class MelodiaTrainer:
    """PyTorch trainer for the Melodia model"""
    
    def __init__(self, model: MelodiaModel, model_config: ModelConfig, training_config=None):
        self.model = model
        self.config = model_config
        self.training_config = training_config
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer with gradient clipping
        learning_rate = training_config.learning_rate if training_config else 0.001
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Learning rate scheduler
        max_epochs = training_config.max_epochs if training_config else 100
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=learning_rate * 0.1
        )
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        
    def train_step(self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor) -> Tuple[float, float]:
        """Perform a single training step"""
        self.model.train()
        
        # Move data to device
        batch_inputs = batch_inputs.to(self.device)
        batch_targets = batch_targets.to(self.device)
        
        # Forward pass
        logits = self.model(batch_inputs)
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = batch_targets.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clip_val = self.training_config.gradient_clip_val if self.training_config else self.config.gradient_clip_val
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            gradient_clip_val
        )
        
        # Update weights
        self.optimizer.step()
        
        # Calculate accuracy (excluding padding tokens)
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            mask = targets != 0  # Exclude padding tokens
            if mask.sum() > 0:
                accuracy = (predictions[mask] == targets[mask]).float().mean().item()
            else:
                accuracy = 0.0
        
        return loss.item(), accuracy
    
    def validate_step(self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor) -> Tuple[float, float]:
        """Perform a single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            logits = self.model(batch_inputs)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = batch_targets.view(-1)
            
            # Calculate loss
            loss = self.criterion(logits, targets)
            
            # Calculate accuracy (excluding padding tokens)
            predictions = torch.argmax(logits, dim=-1)
            mask = targets != 0  # Exclude padding tokens
            if mask.sum() > 0:
                accuracy = (predictions[mask] == targets[mask]).float().mean().item()
            else:
                accuracy = 0.0
        
        return loss.item(), accuracy
    
    def save_model(self, path: str):
        """Save the model and optimizer state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
        }, path)
    
    def load_model(self, path: str):
        """Load the model and optimizer state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        
    def get_device_info(self) -> str:
        """Get information about the device being used"""
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            return "CPU"