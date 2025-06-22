# melodia/generation/generator.py

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from ..config import GenerationConfig, ModelConfig
from ..data.loader import MusicEvent
from ..data.processor import EventTokenizer
import logging

logger = logging.getLogger(__name__)

class MusicGenerator:
    """Handles music generation using the trained model"""
    
    def __init__(
        self,
        model: tf.keras.Model,
        tokenizer: EventTokenizer,
        config: GenerationConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
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
        generated_events = self.tokenizer.tokens_to_events(current_tokens)
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
        # Prepare input
        model_input = tf.constant([current_tokens], dtype=tf.int32)
        
        # Get model predictions
        predictions = self.model(
            model_input,
            training=False
        )
        
        # Get logits for next token
        next_token_logits = predictions[0, -1, :]
        
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
        probs = tf.nn.softmax(next_token_logits)
        
        # Sample next token
        next_token = tf.random.categorical(
            tf.math.log(probs)[None, :],
            num_samples=1
        )[0, 0].numpy()
        
        return int(next_token)
    
    def _apply_repetition_penalty(
        self,
        logits: tf.Tensor,
        tokens: List[int],
        penalty: float
    ) -> tf.Tensor:
        """Apply repetition penalty to logits"""
        # Count token frequencies
        token_counts = {}
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1
        
        # Apply penalty
        for token, count in token_counts.items():
            if count > 0:
                logits = tf.tensor_scatter_nd_update(
                    logits,
                    [[token]],
                    [logits[token] / (penalty * count)]
                )
        
        return logits
    
    def _top_k_filtering(
        self,
        logits: tf.Tensor,
        k: int
    ) -> tf.Tensor:
        """Apply top-k filtering to logits"""
        top_k_logits, top_k_indices = tf.math.top_k(logits, k=k)
        indices_to_remove = logits < tf.math.reduce_min(top_k_logits)
        logits = tf.where(indices_to_remove, tf.float32.min, logits)
        return logits
    
    def _top_p_filtering(
        self,
        logits: tf.Tensor,
        p: float
    ) -> tf.Tensor:
        """Apply nucleus (top-p) filtering to logits"""
        sorted_logits, sorted_indices = tf.math.top_k(
            logits,
            k=tf.shape(logits)[-1]
        )
        cumulative_probs = tf.math.cumsum(
            tf.nn.softmax(sorted_logits),
            axis=-1
        )
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        
        # Keep tokens with cumulative probability less than p
        sorted_indices_to_remove = tf.concat(
            [
                tf.zeros_like(sorted_indices_to_remove[:1], dtype=bool),
                sorted_indices_to_remove[1:]
            ],
            axis=0
        )
        
        # Scatter back to original indices
        indices_to_remove = tf.scatter_nd(
            tf.expand_dims(sorted_indices, 1),
            sorted_indices_to_remove,
            tf.shape(sorted_indices_to_remove)
        )
        
        logits = tf.where(indices_to_remove, tf.float32.min, logits)
        return logits

class StructuredMusicGenerator(MusicGenerator):
    """Extended generator with structure-aware generation"""
    
    def __init__(
        self,
        model: tf.keras.Model,
        tokenizer: EventTokenizer,
        config: GenerationConfig,
        structure_config: Optional[Dict] = None
    ):
        super().__init__(model, tokenizer, config)
        self.structure_config = structure_config or {
            'sections': ['A', 'B', 'A'],  # Default song structure
            'section_length': 16,  # Bars per section
            'time_signature': (4, 4)
        }
    
    def generate_structured(
        self,
        style_id: Optional[int] = None,
        chord_progression: Optional[List[str]] = None,
        **kwargs
    ) -> List[MusicEvent]:
        """Generate music following a specific structure"""
        generated_sections = []
        
        for section in self.structure_config['sections']:
            # Generate section-specific conditions
            conditions = self._create_section_conditions(
                section,
                style_id,
                chord_progression
            )
            
            # Generate section
            section_events = self.generate(
                conditions=conditions,
                max_length=self._calculate_section_length(),
                **kwargs
            )
            
            generated_sections.extend(section_events)
        
        return self._post_process_structure(generated_sections)
    
    def _create_section_conditions(
        self,
        section: str,
        style_id: Optional[int],
        chord_progression: Optional[List[str]]
    ) -> Dict:
        """Create conditions for a specific section"""
        conditions = {
            'section': section,
            'style_id': style_id
        }
        
        if chord_progression:
            # Map chord progression to section
            section_chords = self._map_chords_to_section(
                chord_progression,
                section
            )
            conditions['chord_progression'] = section_chords
        
        return conditions
    
    def _calculate_section_length(self) -> int:
        """Calculate target length for a section"""
        bars = self.structure_config['section_length']
        beats_per_bar = self.structure_config['time_signature'][0]
        return bars * beats_per_bar * 4  # Assuming 16th note resolution
    
    def _map_chords_to_section(
        self,
        chord_progression: List[str],
        section: str
    ) -> List[str]:
        """Map chord progression to specific section"""
        # Implement chord progression mapping logic
        # This could involve repeating, transposing, or varying the progression
        return chord_progression
    
    def _post_process_structure(
        self,
        events: List[MusicEvent]
    ) -> List[MusicEvent]:
        """Post-process generated events to ensure structural coherence"""
        # Add section markers
        processed_events = []
        current_time = 0.0
        
        for section_idx, section in enumerate(self.structure_config['sections']):
            section_start = current_time
            section_end = section_start + (
                self.structure_config['section_length'] * 
                self.structure_config['time_signature'][0]
            )
            
            # Filter events for this section
            section_events = [
                event for event in events
                if section_start <= event.time < section_end
            ]
            
            # Add section metadata
            section_marker = MusicEvent(
                type='marker',
                time=section_start,
                key=f'Section_{section}'
            )
            processed_events.append(section_marker)
            
            # Add section events
            processed_events.extend(section_events)
            current_time = section_end
        
        return processed_events