# melodia/generation/generator.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from ..config import GenerationConfig, ModelConfig
from ..data.loader import MusicEvent
from ..data.processor import EventTokenizer
import logging

logger = logging.getLogger(__name__)

class MusicGenerator:
    """Handles music generation using the trained PyTorch model"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: EventTokenizer,
        config: GenerationConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def generate(
        self,
        seed_events: Optional[List[MusicEvent]] = None,
        conditions: Optional[Dict] = None,
        max_length: Optional[int] = None,
        **kwargs
    ) -> List[MusicEvent]:
        """Generate a new musical sequence"""
        # Process generation parameters
        max_length = max_length or self.config.max_length
        temperature = kwargs.get('temperature', self.config.temperature)
        top_k = kwargs.get('top_k', self.config.top_k)
        top_p = kwargs.get('top_p', self.config.top_p)
        repetition_penalty = kwargs.get('repetition_penalty', 
                                      self.config.repetition_penalty)
        
        # Initialize sequence
        if seed_events:
            initial_tokens = self.tokenizer.encode_events(seed_events)
        else:
            initial_tokens = [self.tokenizer.BOS_TOKEN]
        
        current_tokens = initial_tokens
        
        # Generate tokens
        with torch.no_grad():
            while len(current_tokens) < max_length:
                next_token = self._generate_next_token(
                    current_tokens,
                    conditions,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty
                )
                
                if next_token == self.tokenizer.EOS_TOKEN:
                    break
                    
                current_tokens.append(next_token)
        
        # Convert back to events
        generated_events = self.tokenizer.decode_tokens(current_tokens)
        return generated_events
    
    def _generate_next_token(
        self,
        current_tokens: List[int],
        conditions: Optional[Dict],
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float
    ) -> int:
        """Generate the next token based on current sequence"""
        # Prepare input tensor
        input_tensor = torch.tensor([current_tokens], dtype=torch.long, device=self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Get logits for next token
            next_token_logits = outputs[0, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    current_tokens,
                    repetition_penalty
                )
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                next_token_logits = self._top_k_filtering(
                    next_token_logits,
                    top_k
                )
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                next_token_logits = self._top_p_filtering(
                    next_token_logits,
                    top_p
                )
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            return next_token
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        tokens: List[int],
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        logits = logits.clone()
        
        # Count token frequencies
        token_counts = {}
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1
        
        # Apply penalty
        for token, count in token_counts.items():
            if count > 0 and 0 <= token < len(logits):
                logits[token] = logits[token] / (penalty * count)
        
        return logits
    
    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        if k <= 0:
            return logits
            
        top_k_logits, top_k_indices = torch.topk(logits, k=min(k, logits.size(-1)))
        indices_to_remove = logits < top_k_logits[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits
    
    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        p: float
    ) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

class StructuredMusicGenerator(MusicGenerator):
    """Extended generator with structured composition capabilities"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: EventTokenizer,
        config: GenerationConfig,
        structure_config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, tokenizer, config, device)
        self.structure_config = structure_config or {}
    
    def generate_structured(
        self,
        style_id: Optional[int] = None,
        chord_progression: Optional[List[str]] = None,
        **kwargs
    ) -> List[MusicEvent]:
        """Generate music with structural constraints"""
        # Define musical form
        form = self.structure_config.get('form', 'AABA')
        section_length = self.structure_config.get('section_length', 8)
        
        all_events = []
        
        # Generate each section
        for i, section in enumerate(form):
            section_conditions = self._create_section_conditions(
                section, style_id, chord_progression
            )
            
            # Generate section
            if i == 0:
                # First section - start from beginning
                section_events = self.generate(
                    conditions=section_conditions,
                    max_length=self._calculate_section_length(),
                    **kwargs
                )
            else:
                # Subsequent sections - use some previous material as seed
                if section in [s for s in form[:i]]:
                    # Repeat/vary previous section
                    previous_idx = form[:i].index(section)
                    seed_length = min(16, len(all_events) // len(form))
                    start_idx = previous_idx * (len(all_events) // len(form))
                    seed_events = all_events[start_idx:start_idx + seed_length]
                else:
                    # New section - use transition from previous
                    seed_events = all_events[-8:] if all_events else None
                
                section_events = self.generate(
                    seed_events=seed_events,
                    conditions=section_conditions,
                    max_length=self._calculate_section_length(),
                    **kwargs
                )
            
            all_events.extend(section_events)
        
        # Post-process for musical coherence
        final_events = self._post_process_structure(all_events)
        return final_events
    
    def _create_section_conditions(
        self,
        section: str,
        style_id: Optional[int],
        chord_progression: Optional[List[str]]
    ) -> Dict:
        """Create conditioning information for a musical section"""
        conditions = {}
        
        if style_id is not None:
            conditions['style'] = style_id
        
        if chord_progression:
            # Map chord progression to this section
            section_chords = self._map_chords_to_section(chord_progression, section)
            conditions['chords'] = section_chords
        
        # Section-specific characteristics
        if section == 'A':
            conditions['energy'] = 'medium'
        elif section == 'B':
            conditions['energy'] = 'high'
        
        return conditions
    
    def _calculate_section_length(self) -> int:
        """Calculate appropriate length for a section"""
        base_length = self.structure_config.get('section_length', 64)
        return base_length
    
    def _map_chords_to_section(
        self,
        chord_progression: List[str],
        section: str
    ) -> List[str]:
        """Map chord progression to a specific section"""
        # Simple mapping - could be more sophisticated
        return chord_progression[:4]  # Use first 4 chords
    
    def _post_process_structure(
        self,
        events: List[MusicEvent]
    ) -> List[MusicEvent]:
        """Apply post-processing for better musical structure"""
        # Simple post-processing - could add more sophisticated rules
        processed_events = []
        
        for event in events:
            # Basic filtering and cleanup
            if event.type == 'note' and event.pitch is not None:
                # Ensure notes are in reasonable range
                if 21 <= event.pitch <= 108:  # Piano range
                    processed_events.append(event)
            else:
                processed_events.append(event)
        
        return processed_events