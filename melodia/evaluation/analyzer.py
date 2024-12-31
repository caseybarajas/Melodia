# melodia/evaluation/analyzer.py

import music21
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import logging
from ..data.loader import MusicEvent

logger = logging.getLogger(__name__)

class MusicTheoryAnalyzer:
    """Analyzes musical content using music theory principles"""
    
    def __init__(self):
        self.key_analyzer = music21.analysis.floatingKey.KeyAnalyzer()
        self.scale_degrees = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10]
        }
        self.chord_qualities = {
            'major': [(0, 4, 7)],       # Major triad
            'minor': [(0, 3, 7)],       # Minor triad
            'diminished': [(0, 3, 6)],  # Diminished triad
            'augmented': [(0, 4, 8)],   # Augmented triad
            'dominant7': [(0, 4, 7, 10)], # Dominant seventh
            'major7': [(0, 4, 7, 11)],  # Major seventh
            'minor7': [(0, 3, 7, 10)]   # Minor seventh
        }
    
    def analyze_harmony(self, score: music21.stream.Score) -> Dict:
        """Perform harmonic analysis of the score"""
        analysis = {
            'key': self._analyze_key(score),
            'chord_progression': self._analyze_chord_progression(score),
            'harmonic_rhythm': self._analyze_harmonic_rhythm(score),
            'modulations': self._detect_modulations(score),
            'cadences': self._detect_cadences(score)
        }
        return analysis
    
    def _analyze_key(self, score: music21.stream.Score) -> Dict:
        """Analyze the key of the piece"""
        key_analysis = self.key_analyzer.process(score)
        if not key_analysis:
            return {'root': None, 'mode': None, 'certainty': 0.0}
        
        return {
            'root': key_analysis.tonic.name,
            'mode': key_analysis.mode,
            'certainty': key_analysis.correlation
        }
    
    def _analyze_chord_progression(
        self,
        score: music21.stream.Score
    ) -> List[Dict]:
        """Analyze the chord progression"""
        chords = []
        for v_chord in score.chordify().recurse().getElementsByClass('Chord'):
            # Get basic chord information
            root = v_chord.root().name
            quality = self._determine_chord_quality(v_chord)
            
            # Add to progression
            chords.append({
                'root': root,
                'quality': quality,
                'time': float(v_chord.offset),
                'duration': float(v_chord.duration.quarterLength)
            })
        
        return chords
    
    def _determine_chord_quality(self, chord: music21.chord.Chord) -> str:
        """Determine the quality of a chord"""
        # Get intervals from root
        intervals = []
        root_pitch = chord.root().midi
        for pitch in chord.pitches:
            interval = (pitch.midi - root_pitch) % 12
            intervals.append(interval)
        intervals = tuple(sorted(intervals))
        
        # Match against known chord qualities
        for quality, patterns in self.chord_qualities.items():
            if intervals in patterns:
                return quality
        
        return 'unknown'
    
    def _analyze_harmonic_rhythm(
        self,
        score: music21.stream.Score
    ) -> Dict:
        """Analyze the harmonic rhythm"""
        chord_stream = score.chordify()
        changes = []
        prev_chord = None
        
        for chord in chord_stream.recurse().getElementsByClass('Chord'):
            if prev_chord:
                if not chord.pitches == prev_chord.pitches:
                    changes.append(float(chord.offset))
            prev_chord = chord
        
        if not changes:
            return {'rate': 0.0, 'regularity': 0.0}
        
        # Calculate average rate of change
        rate = len(changes) / float(score.duration.quarterLength)
        
        # Calculate regularity (std dev of intervals between changes)
        intervals = np.diff(changes)
        regularity = 1.0 - np.std(intervals) / np.mean(intervals) if len(intervals) > 0 else 0.0
        
        return {
            'rate': rate,
            'regularity': regularity
        }
    
    def _detect_modulations(
        self,
        score: music21.stream.Score
    ) -> List[Dict]:
        """Detect key changes and modulations"""
        modulations = []
        window_size = 8.0  # Analysis window in quarter notes
        
        for window_start in np.arange(0, float(score.duration.quarterLength), window_size/2):
            window = score.getElementsByOffset(
                window_start,
                window_start + window_size,
                includeEndBoundary=False
            )
            
            if window:
                key_analysis = self._analyze_key(window)
                if key_analysis['root']:
                    modulations.append({
                        'time': window_start,
                        'key': key_analysis
                    })
        
        # Filter out redundant modulations
        filtered = []
        prev_key = None
        for mod in modulations:
            if not prev_key or mod['key']['root'] != prev_key['root']:
                filtered.append(mod)
                prev_key = mod['key']
        
        return filtered
    
    def _detect_cadences(
        self,
        score: music21.stream.Score
    ) -> List[Dict]:
        """Detect cadential patterns"""
        cadences = []
        chord_stream = score.chordify()
        
        # Common cadential patterns (root movements)
        cadence_patterns = {
            'perfect': [(5, 1)],    # V-I
            'plagal': [(4, 1)],     # IV-I
            'deceptive': [(5, 6)],  # V-vi
            'half': [(1, 5), (4, 5)]  # I-V or IV-V
        }
        
        # Analyze chord pairs
        prev_chord = None
        for chord in chord_stream.recurse().getElementsByClass('Chord'):
            if prev_chord:
                # Get scale degrees of both chords
                prev_degree = prev_chord.root().scaleDegree
                curr_degree = chord.root().scaleDegree
                movement = (prev_degree, curr_degree)
                
                # Check for cadential patterns
                for cadence_type, patterns in cadence_patterns.items():
                    if movement in patterns:
                        cadences.append({
                            'type': cadence_type,
                            'time': float(chord.offset),
                            'chords': [
                                self._determine_chord_quality(prev_chord),
                                self._determine_chord_quality(chord)
                            ]
                        })
            
            prev_chord = chord
        
        return cadences

class StructureAnalyzer:
    """Analyzes musical structure and patterns"""
    
    def __init__(self):
        self.min_phrase_length = 4  # Minimum phrase length in beats
        self.max_phrase_length = 16  # Maximum phrase length in beats
    
    def analyze_structure(self, score: music21.stream.Score) -> Dict:
        """Analyze the structural organization of the piece"""
        analysis = {
            'phrases': self._detect_phrases(score),
            'motifs': self._detect_motifs(score),
            'sections': self._detect_sections(score),
            'repetition_patterns': self._analyze_repetition(score)
        }
        return analysis
    
    def _detect_phrases(self, score: music21.stream.Score) -> List[Dict]:
        """Detect musical phrases"""
        phrases = []
        current_phrase = []
        current_start = 0.0
        
        for note in score.flat.notesAndRests:
            # Check for phrase boundaries
            if (isinstance(note, music21.note.Rest) and 
                note.duration.quarterLength >= 1.0):
                
                if current_phrase:
                    phrase_length = note.offset - current_start
                    if (self.min_phrase_length <= phrase_length <= 
                        self.max_phrase_length):
                        phrases.append({
                            'start': current_start,
                            'end': note.offset,
                            'notes': current_phrase
                        })
                
                current_phrase = []
                current_start = note.offset + note.duration.quarterLength
            
            elif isinstance(note, music21.note.Note):
                current_phrase.append(note)
        
        return phrases
    
    def _detect_motifs(self, score: music21.stream.Score) -> List[Dict]:
        """Detect recurring melodic/rhythmic motifs"""
        motifs = []
        
        # Extract melodic patterns (3-8 notes)
        for length in range(3, 9):
            patterns = defaultdict(list)
            notes = list(score.flat.notes)
            
            for i in range(len(notes) - length):
                pattern = []
                for j in range(length):
                    if isinstance(notes[i+j], music21.note.Note):
                        # Store both pitch and rhythm
                        pattern.append((
                            notes[i+j].pitch.midi,
                            notes[i+j].duration.quarterLength
                        ))
                
                if pattern:
                    pattern_key = tuple(pattern)
                    patterns[pattern_key].append(float(notes[i].offset))
            
            # Keep patterns that occur multiple times
            for pattern, occurrences in patterns.items():
                if len(occurrences) > 1:
                    motifs.append({
                        'pattern': pattern,
                        'length': length,
                        'occurrences': occurrences
                    })
        
        return motifs
    
    def _detect_sections(self, score: music21.stream.Score) -> List[Dict]:
        """Detect major sections based on similarity"""
        sections = []
        measures = list(score.measures(0, None))
        
        if not measures:
            return sections
        
        # Group measures into potential sections
        section_length = 8  # Typical section length in measures
        current_section = []
        current_start = 0
        
        for i in range(0, len(measures), section_length):
            section_measures = measures[i:i+section_length]
            if section_measures:
                section = {
                    'start': float(section_measures[0].offset),
                    'end': float(section_measures[-1].offset + 
                               section_measures[-1].duration.quarterLength),
                    'measures': section_measures
                }
                sections.append(section)
        
        # Analyze similarity between sections
        for i in range(len(sections)):
            sections[i]['similarity'] = []
            for j in range(len(sections)):
                if i != j:
                    similarity = self._compute_section_similarity(
                        sections[i]['measures'],
                        sections[j]['measures']
                    )
                    sections[i]['similarity'].append({
                        'section': j,
                        'score': similarity
                    })
        
        return sections
    
    def _compute_section_similarity(
        self,
        section1: List[music21.stream.Measure],
        section2: List[music21.stream.Measure]
    ) -> float:
        """Compute similarity between two sections"""
        # Extract features for comparison
        def get_section_features(measures):
            features = []
            for m in measures:
                # Pitch sequence
                pitches = [
                    n.pitch.midi for n in m.flat.notes
                    if isinstance(n, music21.note.Note)
                ]
                
                # Rhythm sequence
                durations = [
                    n.duration.quarterLength for n in m.flat.notes
                ]
                
                features.append((pitches, durations))
            return features
        
        features1 = get_section_features(section1)
        features2 = get_section_features(section2)
        
        # Compare features
        similarity_scores = []
        for (p1, d1), (p2, d2) in zip(features1, features2):
            if p1 and p2:  # If both measures have notes
                # Pitch similarity
                pitch_sim = self._sequence_similarity(p1, p2)
                # Rhythm similarity
                rhythm_sim = self._sequence_similarity(d1, d2)
                similarity_scores.append((pitch_sim + rhythm_sim) / 2)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _sequence_similarity(
        self,
        seq1: List[float],
        seq2: List[float]
    ) -> float:
        """Compute similarity between two sequences"""
        # Dynamic Time Warping could be used here for more sophisticated comparison
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        differences = [
            abs(a - b) for a, b in zip(seq1[:min_len], seq2[:min_len])
        ]
        max_diff = max(max(seq1), max(seq2))
        similarity = 1.0 - (np.mean(differences) / max_diff)
        
        return similarity
    
    def _analyze_repetition(self, score: music21.stream.Score) -> Dict:
        """Analyze patterns of repetition"""
        patterns = {
            'measure_level': self._analyze_measure_repetition(score),
            'phrase_level': self._analyze_phrase_repetition(score)
        }
        return patterns
    
    def _analyze_measure_repetition(
        self,
        score: music21.stream.Score
    ) -> Dict:
        """Analyze repetition patterns at measure level"""
        measures = list(score.measures(0, None))
        patterns = defaultdict(list)
        
        for i, measure in enumerate(measures):
            # Create measure signature
            notes = list(measure.flat.notes)
            signature = []
            for note in notes:
                if isinstance(note, music21.note.Note):
                    signature.append((
                        note.pitch.midi,
                        note.duration.quarterLength
                    ))
            
            if signature:
                pattern_key = tuple(signature)
                patterns[pattern_key].append(i)
        
        # Filter for significant patterns
        significant_patterns = {
            str(pattern): occurrences
            for pattern, occurrences in patterns.items()
            if len(occurrences) > 1
        }
        
        return significant_patterns
    
    def _analyze_phrase_repetition(
        self,
        score: music21.stream.Score
    ) -> Dict:
        """Analyze repetition patterns at phrase level"""
        phrases = self._detect_phrases(score)
        patterns = defaultdict(list)
        
        for i, phrase in enumerate(phrases):
            # Create phrase signature
            signature = []
            for note in phrase['notes']:
                if isinstance(note, music21.note.Note):
                    # Include both pitch and rhythm information
                    signature.append((
                        note.pitch.midi,
                        note.duration.quarterLength
                    ))
            
            if signature:
                pattern_key = tuple(signature)
                patterns[pattern_key].append({
                    'index': i,
                    'start': phrase['start'],
                    'end': phrase['end']
                })
        
        # Filter and analyze patterns
        repetition_analysis = {
            'repeated_phrases': [],
            'phrase_variants': [],
            'common_endings': []
        }
        
        # Find repeated phrases
        for pattern, occurrences in patterns.items():
            if len(occurrences) > 1:
                repetition_analysis['repeated_phrases'].append({
                    'pattern': pattern,
                    'occurrences': occurrences
                })
        
        # Find phrase variants (similar but not identical)
        phrases_analyzed = set()
        for i, phrase1 in enumerate(phrases):
            if i in phrases_analyzed:
                continue
                
            variants = []
            for j, phrase2 in enumerate(phrases):
                if i != j and j not in phrases_analyzed:
                    similarity = self._compute_phrase_similarity(
                        phrase1['notes'],
                        phrase2['notes']
                    )
                    if similarity > 0.8:  # High similarity threshold
                        variants.append({
                            'index': j,
                            'similarity': similarity
                        })
            
            if variants:
                repetition_analysis['phrase_variants'].append({
                    'base_phrase': i,
                    'variants': variants
                })
                phrases_analyzed.update([i] + [v['index'] for v in variants])
        
        # Analyze common phrase endings
        endings = defaultdict(list)
        for phrase in phrases:
            if len(phrase['notes']) >= 3:
                # Look at last 3 notes
                ending_signature = []
                for note in phrase['notes'][-3:]:
                    if isinstance(note, music21.note.Note):
                        ending_signature.append((
                            note.pitch.midi,
                            note.duration.quarterLength
                        ))
                if ending_signature:
                    pattern_key = tuple(ending_signature)
                    endings[pattern_key].append(phrase)
        
        # Keep common endings
        for ending, phrases in endings.items():
            if len(phrases) > 1:
                repetition_analysis['common_endings'].append({
                    'pattern': ending,
                    'count': len(phrases)
                })
        
        return repetition_analysis
    
    def _compute_phrase_similarity(
        self,
        phrase1: List[music21.note.Note],
        phrase2: List[music21.note.Note]
    ) -> float:
        """Compute similarity between two phrases"""
        # Extract features
        def get_features(notes):
            pitches = []
            durations = []
            for note in notes:
                if isinstance(note, music21.note.Note):
                    pitches.append(note.pitch.midi)
                    durations.append(note.duration.quarterLength)
            return pitches, durations
        
        p1, d1 = get_features(phrase1)
        p2, d2 = get_features(phrase2)
        
        if not p1 or not p2:
            return 0.0
        
        # Compute pitch and rhythm similarity
        pitch_sim = self._sequence_similarity(p1, p2)
        rhythm_sim = self._sequence_similarity(d1, d2)
        
        # Weight pitch similarity higher than rhythm
        return 0.6 * pitch_sim + 0.4 * rhythm_sim