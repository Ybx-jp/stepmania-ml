"""
StepMania chart parser for .sm and .ssc files.
Converts charts to tensor format for Phase 1 classification model training.

This parser focuses on:
- 16th note resolution alignment with audio features
- Binary encoding (steps + jumps only)
- Fixed BPM validation
- Audio feature synchronization
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TimingEvent:
    """Represents a timing event (BPM change, stop, etc.)"""
    beat: float
    value: float
    event_type: str  # 'bpm', 'stop', 'warp', etc.


@dataclass
class NoteData:
    """Represents note data for a single difficulty chart"""
    difficulty_name: str
    difficulty_value: int
    notes: str  # Raw note data string
    parsed_notes: Optional[np.ndarray] = None  # Tensor format (timesteps, 4)


@dataclass
class StepManiaChart:
    """Complete StepMania chart data aligned for Phase 1 training"""
    title: str
    artist: str
    audio_file: str
    bpm: float  # Primary BPM (must be fixed for Phase 1)
    offset: float
    sample_start: float
    sample_length: float
    timing_events: List[TimingEvent]
    note_data: List[NoteData]

    # Phase 1 specific properties
    song_length_seconds: float
    timesteps_total: int
    hop_length: int  # For audio feature alignment

    # Additional metadata
    genre: str = ""
    credit: str = ""


class StepManiaParser:
    """Parser for StepMania .sm and .ssc files with Phase 1 focus"""

    def __init__(self,
                 target_sample_rate: int = 22050,
                 timesteps_per_beat: int = 4,  # 16th note resolution
                 min_song_length: float = 90.0,
                 max_song_length: float = 120.0,
                 min_bpm: float = 60.0,
                 max_bpm: float = 200.0):

        self.target_sample_rate = target_sample_rate
        self.timesteps_per_beat = timesteps_per_beat
        self.min_song_length = min_song_length
        self.max_song_length = max_song_length
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def parse_file(self, file_path: str) -> Optional[StepManiaChart]:
        """
        Parse a .sm or .ssc file and return chart data.
        Returns None if chart doesn't meet Phase 1 requirements.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Chart file not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.sm', '.ssc']:
            raise ValueError(f"Unsupported file format: {file_ext}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        chart = self._parse_sm(content)

        # Validate for Phase 1 requirements
        if not self._validate_phase1_requirements(chart):
            return None

        # Calculate audio alignment parameters
        self._calculate_audio_alignment(chart)

        return chart

    def _parse_sm(self, content: str) -> StepManiaChart:
        """Parse .sm file content"""
        # Remove comments and normalize line endings
        content = re.sub(r'//.*?\n', '\n', content)
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Extract metadata fields
        metadata = {}
        field_pattern = r'#([A-Z]+):([^;]*);'
        matches = re.findall(field_pattern, content, re.DOTALL | re.IGNORECASE)

        for field, value in matches:
            field = field.upper()
            value = value.strip()
            metadata[field] = value

        # Parse primary BPM (must be fixed for Phase 1)
        bpm = self._extract_primary_bpm(metadata)

        # Parse timing events
        timing_events = self._parse_timing_events(metadata)

        # Parse note data
        note_data = self._parse_notes_sm(content)

        # Estimate song length (will be refined with actual audio)
        song_length = float(metadata.get('SAMPLELENGTH', '120'))

        # Create chart object
        chart = StepManiaChart(
            title=metadata.get('TITLE', ''),
            artist=metadata.get('ARTIST', ''),
            audio_file=metadata.get('MUSIC', ''),
            bpm=bpm,
            offset=float(metadata.get('OFFSET', '0')),
            sample_start=float(metadata.get('SAMPLESTART', '0')),
            sample_length=song_length,
            timing_events=timing_events,
            note_data=note_data,
            song_length_seconds=song_length,
            timesteps_total=0,  # Will be calculated
            hop_length=0,  # Will be calculated
            genre=metadata.get('GENRE', ''),
            credit=metadata.get('CREDIT', '')
        )

        return chart

    def _extract_primary_bpm(self, metadata: Dict[str, str]) -> float:
        """Extract primary BPM, ensuring it's fixed for Phase 1"""
        if 'BPMS' not in metadata:
            return 120.0  # Default BPM

        bpm_string = metadata['BPMS']
        bpm_pairs = bpm_string.split(',')

        # For Phase 1, we only support fixed BPM (single BPM value)
        if len(bpm_pairs) > 1:
            # Multiple BPM changes - not supported in Phase 1
            raise ValueError("Variable BPM not supported in Phase 1")

        if '=' in bpm_pairs[0]:
            _, bpm_str = bpm_pairs[0].split('=', 1)
            return float(bpm_str.strip())
        else:
            return 120.0

    def _validate_phase1_requirements(self, chart: StepManiaChart) -> bool:
        """Validate chart meets Phase 1 requirements"""
        # Check BPM range
        if not (self.min_bpm <= chart.bpm <= self.max_bpm):
            return False

        # Check song length
        if not (self.min_song_length <= chart.song_length_seconds <= self.max_song_length):
            return False

        # Check for multiple BPM changes (not allowed in Phase 1)
        bpm_events = [e for e in chart.timing_events if e.event_type == 'bpm']
        if len(bpm_events) > 1:
            return False

        # Check for valid difficulty charts in target range (1-10)
        valid_charts = [n for n in chart.note_data if 1 <= n.difficulty_value <= 10]
        if not valid_charts:
            return False

        return True

    def _calculate_audio_alignment(self, chart: StepManiaChart):
        """Calculate parameters for audio feature alignment"""
        # Calculate total timesteps for 16th note resolution
        total_beats = (chart.song_length_seconds * chart.bpm) / 60
        chart.timesteps_total = int(total_beats * self.timesteps_per_beat)

        # Calculate hop_length for librosa alignment
        # hop_length = sr * 60 / (BPM * timesteps_per_beat)
        chart.hop_length = int(self.target_sample_rate * 60 / (chart.bpm * self.timesteps_per_beat))

    def _parse_timing_events(self, metadata: Dict[str, str]) -> List[TimingEvent]:
        """Parse timing events from metadata"""
        events = []

        # Parse BPMs
        if 'BPMS' in metadata:
            bpm_string = metadata['BPMS']
            bpm_pairs = bpm_string.split(',')
            for pair in bpm_pairs:
                if '=' in pair:
                    beat_str, bpm_str = pair.split('=', 1)
                    try:
                        beat = float(beat_str.strip())
                        bpm = float(bpm_str.strip())
                        events.append(TimingEvent(beat, bpm, 'bpm'))
                    except ValueError:
                        continue

        # Parse stops (Phase 1: minimal support)
        if 'STOPS' in metadata and metadata['STOPS'].strip():
            stop_string = metadata['STOPS']
            stop_pairs = stop_string.split(',')
            for pair in stop_pairs:
                if '=' in pair:
                    beat_str, duration_str = pair.split('=', 1)
                    try:
                        beat = float(beat_str.strip())
                        duration = float(duration_str.strip())
                        events.append(TimingEvent(beat, duration, 'stop'))
                    except ValueError:
                        continue

        # Sort events by beat
        events.sort(key=lambda x: x.beat)
        return events

    def _parse_notes_sm(self, content: str) -> List[NoteData]:
        """Parse note data from .sm file"""
        note_data = []

        # Find all #NOTES sections
        notes_pattern = r'#NOTES:\s*([^;]*);'
        notes_matches = re.findall(notes_pattern, content, re.DOTALL | re.IGNORECASE)

        for notes_section in notes_matches:
            lines = notes_section.strip().split('\n')
            if len(lines) < 5:
                continue

            # Parse difficulty metadata
            dance_style = lines[0].strip()
            author = lines[1].strip()
            difficulty_name = lines[2].strip()
            difficulty_value = int(lines[3].strip()) if lines[3].strip().isdigit() else 0
            radar_values = lines[4].strip()

            # Only process single (4-panel) charts
            if dance_style.lower() != 'dance-single':
                continue

            # Extract note data (everything after the 5th line)
            notes_content = '\n'.join(lines[5:])

            note_data.append(NoteData(
                difficulty_name=difficulty_name,
                difficulty_value=difficulty_value,
                notes=notes_content
            ))

        return note_data

    def convert_to_tensor(self, chart: StepManiaChart, note_data: NoteData) -> np.ndarray:
        """
        Convert note data to tensor format: (timesteps_total, 4)
        Binary encoding: 0 = no step, 1 = step
        """
        # Initialize tensor with zeros
        chart_tensor = np.zeros((chart.timesteps_total, 4), dtype=np.float32)

        # Parse note measures
        measures = note_data.notes.split(',')
        current_beat = 0.0

        for measure in measures:
            measure = measure.strip()
            if not measure:
                continue

            # Split into lines (each line is a timestep within the measure)
            lines = [line.strip() for line in measure.split('\n') if line.strip()]

            if not lines:
                continue

            # Calculate beats per line in this measure
            beats_per_line = 4.0 / len(lines)  # 4 beats per measure

            for line_idx, line in enumerate(lines):
                if len(line) >= 4:
                    # Calculate beat position
                    beat_position = current_beat + (line_idx * beats_per_line)

                    # Convert to timestep index
                    timestep_idx = int(round(beat_position * self.timesteps_per_beat))

                    # Ensure timestep is within bounds
                    if 0 <= timestep_idx < chart.timesteps_total:
                        # Process each panel (only first 4 for single charts)
                        for panel_idx in range(4):
                            char = line[panel_idx]
                            if char == '1':  # Tap note
                                chart_tensor[timestep_idx, panel_idx] = 1.0
                            elif char == '2':  # Hold start -> convert to tap for Phase 1
                                chart_tensor[timestep_idx, panel_idx] = 1.0
                            # Ignore holds ends ('3'), mines ('M'), etc. for Phase 1

            # Move to next measure (4 beats)
            current_beat += 4.0

        return chart_tensor

    def validate_pattern_quality(self, chart_tensor: np.ndarray) -> bool:
        """
        Validate chart meets Phase 1 quality requirements:
        - Max 2 simultaneous steps (jump constraint)
        - Reasonable step density
        - No impossible patterns
        """
        # Check maximum simultaneous steps
        max_simultaneous = np.max(np.sum(chart_tensor, axis=1))
        if max_simultaneous > 2:
            return False

        # Check overall step density (not too sparse or too dense)
        total_steps = np.sum(chart_tensor)
        step_density = total_steps / chart_tensor.shape[0]

        if step_density < 0.01 or step_density > 0.5:  # Reasonable bounds
            return False

        return True

    def process_chart(self, file_path: str) -> Optional[Tuple[StepManiaChart, List[np.ndarray]]]:
        """
        Complete processing pipeline for Phase 1:
        Returns chart metadata and list of tensors for valid difficulties
        """
        try:
            # Parse chart
            chart = self.parse_file(file_path)
            if chart is None:
                return None

            # Convert all valid note data to tensors
            chart_tensors = []
            valid_note_data = []

            for note_data in chart.note_data:
                # Skip difficulties outside Phase 1 range
                if not (1 <= note_data.difficulty_value <= 10):
                    continue

                # Convert to tensor
                chart_tensor = self.convert_to_tensor(chart, note_data)

                # Validate quality
                if not self.validate_pattern_quality(chart_tensor):
                    continue

                # Store tensor in note_data for reference
                note_data.parsed_notes = chart_tensor

                chart_tensors.append(chart_tensor)
                valid_note_data.append(note_data)

            if not chart_tensors:
                return None

            # Update chart with only valid note data
            chart.note_data = valid_note_data

            return chart, chart_tensors

        except Exception as e:
            # Log error and skip problematic charts
            print(f"Error processing {file_path}: {e}")
            return None

    def get_audio_alignment_params(self, chart: StepManiaChart) -> Dict[str, any]:
        """Get parameters needed for audio feature extraction alignment"""
        return {
            'sample_rate': self.target_sample_rate,
            'hop_length': chart.hop_length,
            'n_fft': chart.hop_length * 4,  # Common ratio
            'expected_frames': chart.timesteps_total
        }