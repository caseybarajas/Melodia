# melodia/utils/midi.py

import mido
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import logging
from ..data.loader import MusicEvent

logger = logging.getLogger(__name__)

class MIDIProcessor:
    """Handles MIDI file processing and manipulation"""
    
    def __init__(self, ticks_per_beat: int = 480):
        self.ticks_per_beat = ticks_per_beat
        self.current_tempo = 500000  # Default tempo (120 BPM)
    
    def read_midi(self, file_path: Union[str, Path]) -> List[MusicEvent]:
        """Read MIDI file and convert to MusicEvents"""
        try:
            midi_file = mido.MidiFile(str(file_path))
            return self._parse_midi_file(midi_file)
        except Exception as e:
            logger.error(f"Error reading MIDI file {file_path}: {str(e)}")
            return []
    
    def write_midi(
        self,
        events: List[MusicEvent],
        file_path: Union[str, Path],
        program: int = 0
    ):
        """Write MusicEvents to MIDI file"""
        try:
            midi_file = self._create_midi_file(events, program)
            midi_file.save(str(file_path))
        except Exception as e:
            logger.error(f"Error writing MIDI file {file_path}: {str(e)}")
    
    def _parse_midi_file(self, midi_file: mido.MidiFile) -> List[MusicEvent]:
        """Parse MIDI file into MusicEvents"""
        events = []
        current_time = 0.0
        
        for track in midi_file.tracks:
            track_events = self._parse_track(track)
            events.extend(track_events)
        
        # Sort events by time
        events.sort(key=lambda x: x.time)
        return events
    
    def _parse_track(self, track: mido.MidiTrack) -> List[MusicEvent]:
        """Parse MIDI track into MusicEvents"""
        events = []
        current_time = 0.0
        active_notes = {}  # Keep track of active notes for duration calculation
        
        for msg in track:
            current_time += msg.time * self.current_tempo / (self.ticks_per_beat * 1000000)
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {
                    'start_time': current_time,
                    'velocity': msg.velocity
                }
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_time = active_notes[msg.note]['start_time']
                    velocity = active_notes[msg.note]['velocity']
                    duration = current_time - start_time
                    
                    events.append(MusicEvent(
                        type='note',
                        time=start_time,
                        pitch=msg.note,
                        velocity=velocity,
                        duration=duration
                    ))
                    
                    del active_notes[msg.note]
            
            elif msg.type == 'set_tempo':
                self.current_tempo = msg.tempo
                events.append(MusicEvent(
                    type='tempo',
                    time=current_time,
                    tempo=60000000 / msg.tempo
                ))
            
            elif msg.type == 'time_signature':
                events.append(MusicEvent(
                    type='time_signature',
                    time=current_time,
                    numerator=msg.numerator,
                    denominator=msg.denominator
                ))
            
            elif msg.type == 'key_signature':
                events.append(MusicEvent(
                    type='key_signature',
                    time=current_time,
                    key=msg.key
                ))
            
            elif msg.type == 'program_change':
                events.append(MusicEvent(
                    type='program',
                    time=current_time,
                    program=msg.program
                ))
        
        return events
    
    def _create_midi_file(
        self,
        events: List[MusicEvent],
        program: int
    ) -> mido.MidiFile:
        """Create MIDI file from MusicEvents"""
        midi_file = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = mido.MidiTrack()
        midi_file.tracks.append(track)
        
        # Add program change
        track.append(mido.Message('program_change', program=program, time=0))
        
        # Convert events to MIDI messages
        messages = []
        for event in events:
            if event.type == 'note':
                # Note on
                messages.append((
                    event.time,
                    mido.Message('note_on', note=int(event.pitch),
                               velocity=int(event.velocity or 64))
                ))
                # Note off
                messages.append((
                    event.time + event.duration,
                    mido.Message('note_off', note=int(event.pitch),
                               velocity=0)
                ))
            
            elif event.type == 'chord':
                # Handle chord as multiple notes
                if isinstance(event.pitch, list):
                    for pitch in event.pitch:
                        # Note on
                        messages.append((
                            event.time,
                            mido.Message('note_on', note=int(pitch),
                                       velocity=int(event.velocity or 64))
                        ))
                        # Note off
                        messages.append((
                            event.time + event.duration,
                            mido.Message('note_off', note=int(pitch),
                                       velocity=0)
                        ))
            
            elif event.type == 'tempo':
                tempo = int(60000000 / event.tempo)
                messages.append((
                    event.time,
                    mido.MetaMessage('set_tempo', tempo=tempo)
                ))
            
            elif event.type == 'time_signature':
                messages.append((
                    event.time,
                    mido.MetaMessage('time_signature',
                                   numerator=event.numerator,
                                   denominator=event.denominator)
                ))
            
            elif event.type == 'key_signature':
                messages.append((
                    event.time,
                    mido.MetaMessage('key_signature', key=event.key)
                ))
        
        # Sort messages by time
        messages.sort(key=lambda x: x[0])
        
        # Convert absolute times to delta times
        last_time = 0
        for time, msg in messages:
            delta_time = time - last_time
            delta_ticks = int(delta_time * self.ticks_per_beat * 1000000 / self.current_tempo)
            track.append(msg.copy(time=max(0, delta_ticks)))
            last_time = time
        
        return midi_file

class MIDIAugmenter:
    """Implements MIDI data augmentation techniques"""
    
    @staticmethod
    def transpose(
        events: List[MusicEvent],
        semitones: int,
        min_pitch: int = 0,
        max_pitch: int = 127
    ) -> List[MusicEvent]:
        """Transpose MIDI events by given number of semitones"""
        transposed = []
        
        for event in events:
            event_copy = event.__class__(**event.__dict__)
            
            if event.type == 'note':
                new_pitch = event.pitch + semitones
                if min_pitch <= new_pitch <= max_pitch:
                    event_copy.pitch = new_pitch
                    transposed.append(event_copy)
            else:
                transposed.append(event_copy)
        
        return transposed
    
    @staticmethod
    def time_stretch(
        events: List[MusicEvent],
        factor: float
    ) -> List[MusicEvent]:
        """Stretch or compress time by given factor"""
        stretched = []
        
        for event in events:
            event_copy = event.__class__(**event.__dict__)
            event_copy.time *= factor
            if hasattr(event_copy, 'duration') and event_copy.duration is not None:
                event_copy.duration *= factor
            stretched.append(event_copy)
        
        return stretched
    
    @staticmethod
    def velocity_scale(
        events: List[MusicEvent],
        factor: float,
        min_velocity: int = 1,
        max_velocity: int = 127
    ) -> List[MusicEvent]:
        """Scale note velocities by given factor"""
        scaled = []
        
        for event in events:
            event_copy = event.__class__(**event.__dict__)
            
            if event.type == 'note':
                new_velocity = int(event.velocity * factor)
                event_copy.velocity = max(min_velocity,
                                       min(max_velocity, new_velocity))
            scaled.append(event_copy)
        
        return scaled