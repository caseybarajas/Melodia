# melodia/evaluation/metrics.py

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from collections import defaultdict
import music21
from music21 import converter, analysis, key, meter, harmony, pitch
from ..data.loader import MusicEvent
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MusicEvaluator:
    """Evaluates generated music using multiple metrics"""
    
    def __init__(self):
        # Initialize analyzers without streams - will be created per analysis
        self.key_analyzer = None
        self.key_correlation_analyzer = None
    
    def evaluate(
        self,
        events: List[MusicEvent],
        reference: Optional[List[MusicEvent]] = None
    ) -> Dict[str, float]:
        """Evaluate a generated piece using multiple metrics"""
        # Convert events to music21 score
        score = self._events_to_score(events)
        ref_score = self._events_to_score(reference) if reference else None
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self.compute_basic_metrics(score))
        
        # Harmonic metrics
        metrics.update(self.compute_harmonic_metrics(score))
        
        # Rhythmic metrics
        metrics.update(self.compute_rhythmic_metrics(score))
        
        # Melodic metrics
        metrics.update(self.compute_melodic_metrics(score))
        
        # Structure metrics
        metrics.update(self.compute_structure_metrics(score))
        
        # Style consistency
        if ref_score:
            metrics.update(self.compute_style_metrics(score, ref_score))
        
        return metrics
    
    def _events_to_score(self, events: List[MusicEvent]) -> music21.stream.Score:
        """Convert MusicEvent list to music21 Score"""
        score = music21.stream.Score()
        part = music21.stream.Part()
        
        current_time = 0.0
        for event in events:
            if event.type == 'note':
                n = music21.note.Note(
                    pitch=event.pitch,
                    quarterLength=event.duration
                )
                n.offset = event.time
                part.append(n)
            elif event.type == 'chord':
                c = music21.chord.Chord(
                    event.pitch,
                    quarterLength=event.duration
                )
                c.offset = event.time
                part.append(c)
            
        score.append(part)
        return score
    
    def compute_basic_metrics(self, score: music21.stream.Score) -> Dict[str, float]:
        """Compute basic statistical metrics"""
        metrics = {}
        
        # Note density
        total_duration = score.duration.quarterLength
        if total_duration == 0:
            total_duration = 1.0  # Avoid division by zero
        
        note_count = len(score.flat.notes)
        metrics['note_density'] = note_count / total_duration
        
        # Collect all pitches from notes and chords
        pitches = []
        for n in score.flat.notes:
            if isinstance(n, music21.note.Note):
                pitches.append(n.pitch.midi)
            elif isinstance(n, music21.chord.Chord):
                # For chords, use all pitches or just the highest one
                pitches.extend([p.midi for p in n.pitches])
        
        if pitches:
            metrics['pitch_range'] = max(pitches) - min(pitches)
            metrics['avg_pitch'] = np.mean(pitches)
        else:
            metrics['pitch_range'] = 0
            metrics['avg_pitch'] = 60  # Middle C as default
        
        # Duration statistics
        durations = [n.duration.quarterLength for n in score.flat.notes]
        if durations:
            metrics['avg_duration'] = np.mean(durations)
            metrics['duration_variance'] = np.var(durations)
        else:
            metrics['avg_duration'] = 1.0
            metrics['duration_variance'] = 0.0
        
        return metrics
    
    def compute_harmonic_metrics(self, score: music21.stream.Score) -> Dict[str, float]:
        """Compute harmony-related metrics"""
        metrics = {}
        
        # Key strength
        try:
            key_analyzer = analysis.floatingKey.KeyAnalyzer(score)
            key_analysis = key_analyzer.process()
            if key_analysis:
                metrics['key_strength'] = key_analysis.correlation
        except Exception:
            metrics['key_strength'] = 0.0
        
        # Chord progression analysis
        chords = score.chordify()
        progression = []
        for c in chords.recurse().getElementsByClass('Chord'):
            progression.append(c)
        
        if progression:
            # Harmonic rhythm
            metrics['harmonic_rhythm'] = len(progression) / score.duration.quarterLength
            
            # Chord complexity
            chord_sizes = [len(c.pitches) for c in progression]
            metrics['avg_chord_complexity'] = np.mean(chord_sizes)
            
            # Harmonic tension
            metrics['harmonic_tension'] = self._compute_harmonic_tension(progression)
        
        return metrics
    
    def compute_rhythmic_metrics(self, score: music21.stream.Score) -> Dict[str, float]:
        """Compute rhythm-related metrics"""
        metrics = {}
        
        # Get all notes and rests
        elements = list(score.flat.notesAndRests)
        
        if elements:
            # Rhythmic complexity
            durations = [e.duration.quarterLength for e in elements]
            unique_durations = len(set(durations))
            metrics['rhythmic_complexity'] = unique_durations / len(durations)
            
            # Syncopation
            metrics['syncopation'] = self._compute_syncopation(elements)
            
            # Rhythmic consistency
            metrics['rhythmic_consistency'] = self._compute_rhythmic_consistency(elements)
            
            # Groove strength
            metrics['groove_strength'] = self._compute_groove_strength(elements)
        
        return metrics
    
    def compute_melodic_metrics(self, score: music21.stream.Score) -> Dict[str, float]:
        """Compute melody-related metrics"""
        metrics = {}
        
        # Get melody line (highest notes)
        melody = []
        for n in score.flat.notes:
            if isinstance(n, music21.note.Note):
                melody.append(n)
            elif isinstance(n, music21.chord.Chord):
                # Extract the highest note from the chord using pitches
                if n.pitches:
                    highest_pitch = max(n.pitches, key=lambda p: p.midi)
                    highest_note = music21.note.Note(
                        pitch=highest_pitch,
                        quarterLength=n.duration.quarterLength
                    )
                    highest_note.offset = n.offset
                    melody.append(highest_note)
        
        if melody and len(melody) > 1:
            # Melodic intervals
            intervals = []
            for i in range(len(melody)-1):
                try:
                    interval = abs(melody[i+1].pitch.midi - melody[i].pitch.midi)
                    intervals.append(interval)
                except AttributeError:
                    # Skip if pitch access fails
                    continue
            
            if intervals:
                metrics['avg_melodic_interval'] = np.mean(intervals)
                metrics['melodic_range'] = max(intervals)
                
                # Stepwise motion ratio
                stepwise = sum(1 for i in intervals if i <= 2)
                metrics['stepwise_motion_ratio'] = stepwise / len(intervals)
                
                # Melodic contour
                metrics['contour_variety'] = self._compute_contour_variety(melody)
            else:
                metrics['avg_melodic_interval'] = 0.0
                metrics['melodic_range'] = 0
                metrics['stepwise_motion_ratio'] = 0.0
                metrics['contour_variety'] = 0.0
        else:
            metrics['avg_melodic_interval'] = 0.0
            metrics['melodic_range'] = 0
            metrics['stepwise_motion_ratio'] = 0.0
            metrics['contour_variety'] = 0.0
        
        return metrics
    
    def compute_structure_metrics(self, score: music21.stream.Score) -> Dict[str, float]:
        """Compute structure-related metrics"""
        metrics = {}
        
        # Phrase detection
        phrases = self._detect_phrases(score)
        if phrases:
            # Phrase length consistency
            phrase_lengths = [len(p) for p in phrases]
            metrics['phrase_length_consistency'] = 1.0 - np.std(phrase_lengths) / np.mean(phrase_lengths)
            
            # Phrase similarity
            metrics['phrase_similarity'] = self._compute_phrase_similarity(phrases)
            
            # Form strength
            metrics['form_strength'] = self._compute_form_strength(phrases)
        
        return metrics
    
    def compute_style_metrics(
        self,
        score: music21.stream.Score,
        reference: music21.stream.Score
    ) -> Dict[str, float]:
        """Compute style consistency metrics"""
        metrics = {}
        
        # Note distribution similarity
        score_notes = [n.pitch.midi for n in score.flat.notes]
        ref_notes = [n.pitch.midi for n in reference.flat.notes]
        
        metrics['pitch_distribution_similarity'] = self._compute_distribution_similarity(
            score_notes, ref_notes
        )
        
        # Rhythm distribution similarity
        score_durations = [n.duration.quarterLength for n in score.flat.notes]
        ref_durations = [n.duration.quarterLength for n in reference.flat.notes]
        
        metrics['rhythm_distribution_similarity'] = self._compute_distribution_similarity(
            score_durations, ref_durations
        )
        
        # Interval pattern similarity
        metrics['interval_pattern_similarity'] = self._compute_interval_similarity(
            score, reference
        )
        
        return metrics
    
    def _compute_harmonic_tension(self, progression: List[music21.chord.Chord]) -> float:
        """Compute harmonic tension based on chord complexity and dissonance"""
        tensions = []
        for chord in progression:
            # Count dissonant intervals
            intervals = chord.intervalVector
            dissonance = sum(intervals[2:])  # Count intervals larger than major second
            
            # Chord complexity based on number of unique pitch classes
            complexity = len(set(p.pitchClass for p in chord.pitches))
            
            tension = (dissonance + complexity) / (6 + 12)  # Normalize
            tensions.append(tension)
        
        return np.mean(tensions)
    
    def _compute_syncopation(self, elements: List[music21.note.GeneralNote]) -> float:
        """Compute degree of syncopation"""
        syncopation = 0
        total_notes = 0
        
        for i in range(1, len(elements)):
            if isinstance(elements[i], music21.note.Note):
                total_notes += 1
                # Check if note starts on weak beat
                if elements[i].offset % 1.0 != 0:
                    syncopation += 1
        
        return syncopation / total_notes if total_notes > 0 else 0
    
    def _compute_rhythmic_consistency(
        self,
        elements: List[music21.note.GeneralNote]
    ) -> float:
        """Compute rhythmic pattern consistency"""
        patterns = defaultdict(int)
        pattern_length = 4  # Look for 4-beat patterns
        
        for i in range(len(elements) - pattern_length):
            pattern = tuple(e.duration.quarterLength for e in elements[i:i+pattern_length])
            patterns[pattern] += 1
        
        if patterns:
            most_common = max(patterns.values())
            return most_common / len(elements)
        return 0.0
    
    def _compute_groove_strength(
        self,
        elements: List[music21.note.GeneralNote]
    ) -> float:
        """Compute strength of rhythmic groove"""
        beats = defaultdict(int)
        total_notes = 0
        
        for e in elements:
            if isinstance(e, music21.note.Note):
                beat_position = e.offset % 4.0  # Assuming 4/4 time
                beats[beat_position] += 1
                total_notes += 1
        
        if total_notes == 0:
            return 0.0
        
        # Calculate groove strength based on beat distribution
        groove_strength = sum(count * (1.0 if pos % 1.0 == 0 else 0.5)
                            for pos, count in beats.items())
        return groove_strength / total_notes
    
    def _compute_contour_variety(self, melody: List[music21.note.Note]) -> float:
        """Compute melodic contour variety"""
        if len(melody) < 3:
            return 0.0
        
        contours = []
        for i in range(len(melody)-2):
            try:
                # Determine contour direction
                pitch1 = melody[i].pitch.midi
                pitch2 = melody[i+1].pitch.midi
                pitch3 = melody[i+2].pitch.midi
                
                if pitch2 > pitch1:
                    if pitch3 > pitch2:
                        contours.append('up')
                    else:
                        contours.append('peak')
                else:
                    if pitch3 < pitch2:
                        contours.append('down')
                    else:
                        contours.append('valley')
            except AttributeError:
                # Skip if pitch access fails
                continue
        
        # Count unique contour patterns
        unique_patterns = len(set(contours))
        return unique_patterns / len(contours) if contours else 0.0
    
    def _detect_phrases(self, score: music21.stream.Score) -> List[List[music21.note.Note]]:
        """Detect musical phrases based on rests and melodic contour"""
        phrases = []
        current_phrase = []
        
        for element in score.flat.notesAndRests:
            if isinstance(element, music21.note.Rest) and element.duration.quarterLength >= 1.0:
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []
            elif isinstance(element, music21.note.Note):
                current_phrase.append(element)
        
        if current_phrase:
            phrases.append(current_phrase)
        
        return phrases
    
    def _compute_phrase_similarity(
        self,
        phrases: List[List[music21.note.Note]]
    ) -> float:
        """Compute similarity between phrases"""
        if len(phrases) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(phrases)):
            for j in range(i+1, len(phrases)):
                sim = self._phrase_similarity(phrases[i], phrases[j])
                similarities.append(sim)
        
        return np.mean(similarities)
    
    def _phrase_similarity(
        self,
        phrase1: List[music21.note.Note],
        phrase2: List[music21.note.Note]
    ) -> float:
        """Compute similarity between two phrases"""
        # Convert to intervals for transposition-invariant comparison
        intervals1 = [
            phrase1[i+1].pitch.midi - phrase1[i].pitch.midi
            for i in range(len(phrase1)-1)
        ]
        intervals2 = [
            phrase2[i+1].pitch.midi - phrase2[i].pitch.midi
            for i in range(len(phrase2)-1)
        ]
        
        # Dynamic Time Warping could be used here for more sophisticated comparison
        min_len = min(len(intervals1), len(intervals2))
        if min_len == 0:
            return 0.0
        
        matches = sum(abs(i1 - i2) <= 2 for i1, i2 in
                     zip(intervals1[:min_len], intervals2[:min_len]))
        return matches / min_len
    
    def _compute_form_strength(self, phrases: List[List[music21.note.Note]]) -> float:
        """Compute strength of musical form based on phrase relationships"""
        if len(phrases) < 2:
            return 0.0
        
        # Create similarity matrix
        n_phrases = len(phrases)
        similarity_matrix = np.zeros((n_phrases, n_phrases))
        
        for i in range(n_phrases):
            for j in range(i+1, n_phrases):
                sim = self._phrase_similarity(phrases[i], phrases[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Look for common forms (e.g., AABA, ABAC)
        common_forms = {
            'AABA': [[1, 1, 0, 1]],
            'ABAC': [[1, 0, 1, 0]],
            'ABAB': [[1, 0, 1, 0]]
        }
        
        form_strengths = []
        for form_name, patterns in common_forms.items():
            if n_phrases >= len(patterns[0]):
                strengths = []
                for pattern in patterns:
                    strength = 0
                    for i in range(len(pattern)):
                        for j in range(i + 1, len(pattern)):
                            if pattern[i] == pattern[j]:
                                strength += similarity_matrix[i, j]
                            else:
                                strength += (1 - similarity_matrix[i, j])
                    strengths.append(strength / (len(pattern) * (len(pattern) - 1) / 2))
                form_strengths.append(max(strengths))
        
        return max(form_strengths) if form_strengths else 0.0
    
    def _compute_distribution_similarity(
        self,
        dist1: List[float],
        dist2: List[float]
    ) -> float:
        """Compute similarity between two distributions using KL divergence"""
        # Create histograms
        hist1, bins = np.histogram(dist1, bins=50, density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)
        
        # Add small constant to avoid division by zero
        eps = 1e-10
        hist1 = hist1 + eps
        hist2 = hist2 + eps
        
        # Normalize
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Compute KL divergence
        kl_div = np.sum(hist1 * np.log(hist1 / hist2))
        
        # Convert to similarity score
        similarity = 1 / (1 + kl_div)
        return similarity
    
    def _compute_interval_similarity(
        self,
        score: music21.stream.Score,
        reference: music21.stream.Score
    ) -> float:
        """Compute similarity of interval patterns"""
        def get_interval_patterns(s: music21.stream.Score) -> List[Tuple[int, ...]]:
            patterns = []
            notes = list(s.flat.notes)
            for i in range(len(notes) - 2):
                if isinstance(notes[i], music21.note.Note):
                    pattern = []
                    for j in range(3):  # Use 3-note patterns
                        if isinstance(notes[i+j], music21.note.Note):
                            if j > 0:
                                interval = notes[i+j].pitch.midi - notes[i+j-1].pitch.midi
                                pattern.append(interval)
                    if len(pattern) == 2:
                        patterns.append(tuple(pattern))
            return patterns
        
        score_patterns = get_interval_patterns(score)
        ref_patterns = get_interval_patterns(reference)
        
        if not score_patterns or not ref_patterns:
            return 0.0
        
        # Count pattern occurrences
        score_counts = defaultdict(int)
        ref_counts = defaultdict(int)
        
        for pattern in score_patterns:
            score_counts[pattern] += 1
        for pattern in ref_patterns:
            ref_counts[pattern] += 1
        
        # Convert to distributions
        all_patterns = set(score_counts.keys()) | set(ref_counts.keys())
        score_dist = [score_counts[p] / len(score_patterns) for p in all_patterns]
        ref_dist = [ref_counts[p] / len(ref_patterns) for p in all_patterns]
        
        return self._compute_distribution_similarity(score_dist, ref_dist)
    
    def get_detailed_report(
        self,
        metrics: Dict[str, float],
        include_descriptive: bool = True
    ) -> str:
        """Generate a detailed report of the evaluation metrics"""
        report = []
        report.append("Music Generation Evaluation Report")
        report.append("=" * 40)
        
        # Group metrics by category
        categories = {
            'Basic Statistics': ['note_density', 'pitch_range', 'avg_pitch', 
                               'avg_duration', 'duration_variance'],
            'Harmonic Quality': ['key_strength', 'harmonic_rhythm', 
                               'avg_chord_complexity', 'harmonic_tension'],
            'Rhythmic Quality': ['rhythmic_complexity', 'syncopation', 
                               'rhythmic_consistency', 'groove_strength'],
            'Melodic Quality': ['avg_melodic_interval', 'melodic_range', 
                              'stepwise_motion_ratio', 'contour_variety'],
            'Structure Quality': ['phrase_length_consistency', 'phrase_similarity',
                                'form_strength'],
            'Style Consistency': ['pitch_distribution_similarity',
                                'rhythm_distribution_similarity',
                                'interval_pattern_similarity']
        }
        
        # Metric descriptions
        descriptions = {
            'note_density': 'Average number of notes per quarter note',
            'key_strength': 'Confidence in detected musical key',
            'rhythmic_complexity': 'Variety of rhythm patterns used',
            'contour_variety': 'Diversity of melodic shapes',
            'form_strength': 'Clarity of musical form and structure'
            # Add more descriptions as needed
        }
        
        for category, metric_names in categories.items():
            report.append(f"\n{category}:")
            report.append("-" * len(category))
            
            for name in metric_names:
                if name in metrics:
                    value = metrics[name]
                    report.append(f"{name}: {value:.3f}")
                    if include_descriptive and name in descriptions:
                        report.append(f"  - {descriptions[name]}")
        
        return "\n".join(report)
    
    def get_summary_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute summary scores for each major category"""
        summaries = {
            'harmonic_quality': np.mean([
                metrics.get('key_strength', 0),
                metrics.get('harmonic_tension', 0),
                metrics.get('avg_chord_complexity', 0)
            ]),
            'rhythmic_quality': np.mean([
                metrics.get('rhythmic_complexity', 0),
                metrics.get('rhythmic_consistency', 0),
                metrics.get('groove_strength', 0)
            ]),
            'melodic_quality': np.mean([
                metrics.get('stepwise_motion_ratio', 0),
                metrics.get('contour_variety', 0)
            ]),
            'structural_quality': np.mean([
                metrics.get('phrase_length_consistency', 0),
                metrics.get('form_strength', 0)
            ]),
            'overall_quality': None  # Computed below
        }
        
        # Compute overall quality as weighted average
        weights = {
            'harmonic_quality': 0.3,
            'rhythmic_quality': 0.25,
            'melodic_quality': 0.25,
            'structural_quality': 0.2
        }
        
        summaries['overall_quality'] = sum(
            summaries[key] * weight
            for key, weight in weights.items()
        )
        
        return summaries