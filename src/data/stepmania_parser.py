"""
StepMania chart parser for .sm and .ssc files.
Converts charts to tensor format for classification model training.

This parser focuses on:
- 16th note resolution alignment with audio features
- Binary encoding (steps + jumps only)
- Variable BPM support for groove radar calculations
- Hold arrow tracking for freeze calculation
- Audio feature synchronization
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import librosa

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
                 config: Optional[Dict] = None,
                 target_sample_rate: int = 22050,
                 timesteps_per_beat: int = 4,  # 16th note resolution
                 min_song_length: float = 75.0,
                 max_song_length: float = 130.0,
                 min_bpm: float = 60.0,
                 max_bpm: float = 200.0,
                 max_simultaneous: int = 2,
                 gimmick_max_bpm: Optional[float] = None,
                 gimmick_min_bpm: float = 15.0,
                 round_quantize: bool = False):
        """
        Initialize parser with optional config dict.

        Args:
            config: Optional dict from data_config.yaml['data']['stepmania']
                   Keys: min_song_length, max_song_length, min_bpm, max_bpm
            max_simultaneous: reject difficulties whose tensor has any frame with more than this many
                   simultaneously-occupied panels. Default 2 (the stale Phase-1 jump constraint, which
                   excludes 55% of real Hard charts — see notes/constraint_relaxation_roadmap.md). Pass 4
                   to admit hands/quads (the typed model's 15-way pattern head supports them).
            gimmick_max_bpm: if set, reject any chart with a RAW BPM event above this (or below
                   gimmick_min_bpm). The min/max_bpm filter above uses the duration-weighted AVERAGE,
                   which is blind to a brief speed-gimmick spike (e.g. #BPMS=...=2467 for scroll flair) —
                   a sane average sails through, then the single-hop 16th grid (hop = sr·60/(avg_bpm·4),
                   ONE hop per song) mis-grids that section. Default None (guard OFF → training byte-
                   identical). Paired with the widened INFERENCE bounds; see StepManiaParser.for_inference()
                   and notes/constraint_relaxation_roadmap.md (the cheap decoupled reach win).
        """
        self.target_sample_rate = target_sample_rate
        self.timesteps_per_beat = timesteps_per_beat
        self.max_simultaneous = max_simultaneous
        self.gimmick_max_bpm = gimmick_max_bpm
        self.gimmick_min_bpm = gimmick_min_bpm
        # data-layer-v2 (notes/data_layer_v2_scope.md): round-to-nearest on a FINER grid instead of floor-to-16th.
        # On the 48th grid (timesteps_per_beat=12) a triplet at beat 1/3 -> round(12/3)=4 -> 4/12 = 1/3 EXACTLY
        # (displacement 0), vs floor(4/3)=1 -> 1/4 (the 0.083-beat / 33 ms triplet shear). Default False keeps the
        # DEPLOYED 16th path (timesteps_per_beat=4, floor) byte-identical. See for_v2().
        self.round_quantize = round_quantize

        # Use config values if provided, otherwise use defaults
        if config:
            self.min_song_length = config.get('min_song_length', min_song_length)
            self.max_song_length = config.get('max_song_length', max_song_length)
            self.min_bpm = config.get('min_bpm', min_bpm)
            self.max_bpm = config.get('max_bpm', max_bpm)
        else:
            self.min_song_length = min_song_length
            self.max_song_length = max_song_length
            self.min_bpm = min_bpm
            self.max_bpm = max_bpm

    @classmethod
    def for_inference(cls, **kwargs) -> "StepManiaParser":
        """Parser for the INFERENCE / export path with WIDENED gates (the cheap decoupled reach win,
        notes/constraint_relaxation_roadmap.md). generate() itself is filter-free (it consumes precomputed
        audio + a scalar bpm and validates neither), so the only thing keeping the export/playtest path off
        an out-of-band real song is THIS parser's dataset-build validation. Widen it so we reach songs
        generate() can already chart — WITHOUT touching training (the training path keeps the default
        narrow [60,200]/[75,130] gates on the clean single-BPM 16th grid).

        Widened vs default: BPM avg [40, 320] (was [60, 200]); song length [30, 600]s (was [75, 130] — the
        generate() context cap truncates the long tail, so the upper bound is only a discovery gate). The
        gimmick guard is ON (gimmick_max_bpm=400) precisely BECAUSE the avg band is now wide enough to admit
        variable-BPM charts whose brief scroll-gimmick spikes (2467/1431/441) would feed the single-hop grid
        garbage. This is PURE reach and independent of the data-layer-v2 grid refactor.
        """
        defaults = dict(min_bpm=40.0, max_bpm=320.0, min_song_length=30.0, max_song_length=600.0,
                        gimmick_max_bpm=400.0)
        defaults.update(kwargs)  # caller overrides win
        return cls(**defaults)

    @classmethod
    def for_v2(cls, subdiv: int = 12, **kwargs) -> "StepManiaParser":
        """Parser for the data-layer-v2 FINER grid (notes/data_layer_v2_scope.md). subdiv=12 = the 48th grid
        (LCM of duple-16th and triplet) that resolves the confirmed triplet tax: `timesteps_per_beat=subdiv` +
        round-to-nearest quantization so triplets land EXACTLY (displacement -> 0) instead of floored to the 16th
        grid. This is the finer-SUBDIVISION half (phase 2a); the variable-BPM audio re-grid (phase 2b, TimingMap
        frame times) is separate. A version bump — keep the deployed 16th parser as the default."""
        defaults = dict(timesteps_per_beat=subdiv, round_quantize=True)
        defaults.update(kwargs)
        return cls(**defaults)

    def _beat_to_ts(self, beat_position: float) -> int:
        """Quantize a beat position to a grid timestep. Floor on the legacy 16th grid (byte-identical), round on
        the v2 finer grid (round_quantize) so triplets snap to their exact 48th cell. See __init__/for_v2()."""
        scaled = beat_position * self.timesteps_per_beat
        return int(np.round(scaled)) if self.round_quantize else int(np.floor(scaled))

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

        chart = self._parse_sm(content, file_path)

        # Validate for Phase 1 requirements
        if not self._validate_phase1_requirements(chart):
            return None

        # Calculate audio alignment parameters
        self._calculate_audio_alignment(chart)

        return chart

    def _get_audio_duration(self, chart_file_path: str, audio_filename: str) -> float:
        """Get real audio duration from audio file"""

        chart_dir = os.path.dirname(chart_file_path)

        if not audio_filename:
            raise ValueError("No audio filename provided in chart metadata")

        audio_path = os.path.join(chart_dir, audio_filename)
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        return librosa.get_duration(path=audio_path)

    def _parse_sm(self, content: str, file_path: str) -> StepManiaChart:
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

        # Get real song length from audio file
        song_length = self._get_audio_duration(file_path, metadata.get('MUSIC', ''))

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
        """Extract primary BPM (first BPM value in the chart)"""
        if not metadata['BPMS']:
            raise ValueError("BPMS metadata is missing")
        bpm_string = metadata['BPMS']
        bpm_pairs = bpm_string.split(',')

        # Return the first BPM value as the primary BPM
        if '=' in bpm_pairs[0]:
            _, bpm_str = bpm_pairs[0].split('=', 1)
            return float(bpm_str.strip())
        else:
            raise ValueError("Error parsing BPMS value")

    def _validate_phase1_requirements(self, chart: StepManiaChart) -> bool:
        """Validate chart meets basic requirements"""
        # Check primary BPM range (use average for variable BPM charts).
        # A chart with no BPM events cannot be used -> reject (return False) rather
        # than letting the ValueError escape this bool-typed validator.
        try:
            avg_bpm = self.compute_average_bpm(chart.timing_events, chart.song_length_seconds)
        except ValueError:
            print(f"{chart.title} failed bpm requirement (no BPM events)")
            return False
        if not (self.min_bpm <= avg_bpm <= self.max_bpm):
            print(f"{chart.title} failed bpm requirement (avg_bpm={avg_bpm:.1f})")
            return False

        # Gimmick guard (INFERENCE-only, off unless gimmick_max_bpm is set): the avg-BPM filter above is
        # blind to a brief speed-gimmick spike (a sane average hides it), but the single-hop 16th grid
        # mis-grids that section. Reject on any RAW event outside the sane-tempo band. See for_inference().
        if self.gimmick_max_bpm is not None:
            raw_bpms = [e.value for e in chart.timing_events if e.event_type == 'bpm']
            bad = [v for v in raw_bpms if v > self.gimmick_max_bpm or v < self.gimmick_min_bpm]
            if bad:
                print(f"{chart.title} failed gimmick guard (raw BPM events {sorted(set(bad))} outside "
                      f"[{self.gimmick_min_bpm:.0f}, {self.gimmick_max_bpm:.0f}])")
                return False

        # Check song length
        if not (self.min_song_length <= chart.song_length_seconds <= self.max_song_length):
            print(f"{chart.title} failed song length requirement")
            return False

        # Variable BPM charts are now allowed for groove radar calculations

        # Check for valid difficulty charts (any dance-single charts with note data)
        valid_charts = [n for n in chart.note_data if n.difficulty_name]
        if not valid_charts:
            print(f"{chart.title} failed valid chart requirement (no dance-single charts)")
            return False

        return True

    def _calculate_audio_alignment(self, chart: StepManiaChart):
        """Calculate parameters for audio feature alignment"""
        # Use average BPM for variable tempo charts
        avg_bpm = self.compute_average_bpm(chart.timing_events, chart.song_length_seconds)

        # Calculate total timesteps for 16th note resolution
        total_beats = (chart.song_length_seconds * avg_bpm) / 60
        chart.timesteps_total = int(total_beats * self.timesteps_per_beat)

        # Calculate hop_length for librosa alignment using average BPM
        # hop_length = sr * 60 / (BPM * timesteps_per_beat)
        chart.hop_length = int(self.target_sample_rate * 60 / (avg_bpm * self.timesteps_per_beat))

    def compute_average_bpm(self, timing_events: List[TimingEvent],
                            song_length_seconds: float) -> float:
        """
        Compute weighted average BPM from timing events.

        For variable BPM charts, weights each BPM by its duration.
        For fixed BPM charts, returns the single BPM value.

        Args:
            timing_events: List of TimingEvent objects
            song_length_seconds: Total song duration in seconds

        Returns:
            Weighted average BPM
        """
        bpm_events = [e for e in timing_events if e.event_type == 'bpm']

        if not bpm_events:
            raise ValueError("No BPM events found")

        if len(bpm_events) == 1:
            return bpm_events[0].value

        # Sort by beat position
        bpm_events = sorted(bpm_events, key=lambda e: e.beat)

        # Calculate total beats in the song (approximate using first BPM)
        total_beats = song_length_seconds * bpm_events[0].value / 60

        # Calculate weighted average
        total_weighted_bpm = 0.0
        for i, event in enumerate(bpm_events):
            # Duration in beats until next BPM change (or end of song)
            if i + 1 < len(bpm_events):
                duration_beats = bpm_events[i + 1].beat - event.beat
            else:
                duration_beats = total_beats - event.beat

            duration_beats = max(0, duration_beats)
            total_weighted_bpm += event.value * duration_beats

        if total_beats > 0:
            return total_weighted_bpm / total_beats
        else:
            return bpm_events[0].value

    def compute_bpm_delta(self, timing_events: List[TimingEvent]) -> float:
        """
        Compute total BPM delta for Chaos calculation.

        Total BPM Delta is the sum of absolute BPM changes throughout the song.
        For gradual BPM changes, uses the difference between highest and lowest.

        Args:
            timing_events: List of TimingEvent objects

        Returns:
            Total BPM delta (sum of all BPM changes)
        """
        bpm_events = [e for e in timing_events if e.event_type == 'bpm']
        stop_events = [e for e in timing_events if e.event_type == 'stop']

        if len(bpm_events) <= 1 and not stop_events:
            return 0.0

        # Sort by beat position
        bpm_events = sorted(bpm_events, key=lambda e: e.beat)

        total_delta = 0.0

        # Sum absolute differences between consecutive BPM values
        for i in range(1, len(bpm_events)):
            delta = abs(bpm_events[i].value - bpm_events[i - 1].value)
            total_delta += delta

        # Stops also contribute - we use the BPM at the stop position
        # (effectively a "pause" which affects rhythm perception)
        # For simplicity, we'll count stops as contributing their duration * current_bpm
        # This is a simplification of the DDR formula

        return total_delta

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
            difficulty_value = int(lines[3].strip().rstrip(':')) if lines[3].strip().rstrip(':').isdigit() else 0
            radar_values = lines[4].strip()

            # Only process single (4-panel) charts
            if dance_style.lower().rstrip(':') != 'dance-single':
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
                    timestep_idx = self._beat_to_ts(beat_position)

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

    def convert_to_tensor_extended(self, chart: StepManiaChart,
                                    note_data: NoteData) -> Tuple[np.ndarray, Dict]:
        """
        Convert note data to tensor format with hold arrow tracking.

        Returns both the chart tensor and hold information for freeze calculation.

        Args:
            chart: StepManiaChart metadata
            note_data: NoteData for a specific difficulty

        Returns:
            Tuple of:
            - chart_tensor: Binary chart encoding (timesteps_total, 4)
            - hold_info: Dict with hold arrow data:
                - 'holds': List of (panel_idx, start_beat, end_beat) tuples
                - 'total_hold_beats': Total hold arrow length in beats
                - 'note_beats': List of (beat_position, panel_idx, note_type) for all notes
        """
        # Initialize tensor with zeros
        chart_tensor = np.zeros((chart.timesteps_total, 4), dtype=np.float32)

        # Track hold information
        holds = []  # List of completed holds: (panel_idx, start_beat, end_beat)
        active_holds = {}  # panel_idx -> start_beat (for holds in progress)
        note_beats = []  # List of (beat_position, panel_idx, note_type) for chaos calculation

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
                    timestep_idx = self._beat_to_ts(beat_position)

                    # Ensure timestep is within bounds
                    if 0 <= timestep_idx < chart.timesteps_total:
                        # Process each panel (only first 4 for single charts)
                        for panel_idx in range(4):
                            char = line[panel_idx]
                            if char == '1':  # Tap note
                                chart_tensor[timestep_idx, panel_idx] = 1.0
                                note_beats.append((beat_position, panel_idx, 'tap'))
                            elif char == '2':  # Hold start
                                chart_tensor[timestep_idx, panel_idx] = 1.0
                                active_holds[panel_idx] = beat_position
                                note_beats.append((beat_position, panel_idx, 'hold_start'))
                            elif char == '3':  # Hold end
                                if panel_idx in active_holds:
                                    start_beat = active_holds.pop(panel_idx)
                                    holds.append((panel_idx, start_beat, beat_position))
                            # Ignore mines ('M'), etc.

            # Move to next measure (4 beats)
            current_beat += 4.0

        # Handle any unclosed holds (extend to end of song)
        total_beats = chart.timesteps_total / self.timesteps_per_beat
        for panel_idx, start_beat in active_holds.items():
            holds.append((panel_idx, start_beat, total_beats))

        # Calculate total hold length in beats
        total_hold_beats = sum(end - start for _, start, end in holds)

        hold_info = {
            'holds': holds,
            'total_hold_beats': total_hold_beats,
            'note_beats': note_beats,
            'song_length_beats': total_beats
        }

        return chart_tensor, hold_info

    # StepMania note chars -> typed symbol id. Mines ('M') and others -> 0 (excluded).
    TYPED_SYMBOLS = {'1': 1, '2': 2, '3': 3, '4': 4}  # tap, hold-head, tail, roll-head

    def convert_to_tensor_typed(self, chart: StepManiaChart, note_data: NoteData) -> np.ndarray:
        """Convert note data to a TYPED tensor: (timesteps_total, 4), int8.

        Each cell is a symbol: 0=none, 1=tap, 2=hold-head, 3=hold/roll-tail, 4=roll-head.
        Lossless for taps/holds/rolls (the rows between a hold head and its tail are 0
        in .sm; the bar is implied). Mines and unknown chars map to 0.

        Additive alternative to convert_to_tensor (binary) — the frozen classifier path
        is unchanged; this feeds the typed generator.
        """
        arr = np.zeros((chart.timesteps_total, 4), dtype=np.int8)
        current_beat = 0.0
        for measure in note_data.notes.split(','):
            lines = [line.strip() for line in measure.strip().split('\n') if line.strip()]
            if not lines:
                continue
            beats_per_line = 4.0 / len(lines)
            for line_idx, line in enumerate(lines):
                if len(line) >= 4:
                    beat_position = current_beat + (line_idx * beats_per_line)
                    ts = self._beat_to_ts(beat_position)
                    if 0 <= ts < chart.timesteps_total:
                        for panel_idx in range(4):
                            sym = self.TYPED_SYMBOLS.get(line[panel_idx], 0)
                            # Collision-safe (sticky), matching convert_to_tensor_extended: when a sub-16th note
                            # quantizes onto an already-occupied cell, do NOT let a later line's EMPTY panel
                            # overwrite an existing note with 0 (that silently dropped sub-16th notes from the
                            # typed path while the binary/radar path kept them).
                            if sym:
                                arr[ts, panel_idx] = sym
            current_beat += 4.0
        return arr

    def validate_pattern_quality(self, chart_tensor: np.ndarray) -> bool:
        """
        Validate chart meets quality requirements:
        - At most self.max_simultaneous simultaneously-occupied panels (default 2; set 4 to allow hands)
        - Reasonable step density
        - No impossible patterns
        """
        # Check maximum simultaneous steps
        max_simultaneous = np.max(np.sum(chart_tensor, axis=1))
        if max_simultaneous > self.max_simultaneous:
            return False

        active_timesteps = np.sum(np.sum(chart_tensor, axis=1) > 0)
        active_ratio = active_timesteps / chart_tensor.shape[0]

        if active_ratio < 0.01 or active_ratio > 0.4:  # Reasonable bounds
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
                # Convert to tensor (difficulty name filtering done in dataset)
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