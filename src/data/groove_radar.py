"""
Groove Radar calculator implementing DDR-style radar values.

Calculates the 5 groove radar values used in Dance Dance Revolution:
- Stream: Average step density (steps per minute)
- Voltage: Peak intensity (based on peak density and BPM)
- Air: Jump density (jumps per minute)
- Freeze: Hold arrow rate
- Chaos: Rhythm complexity (note quantization and BPM changes)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

from .stepmania_parser import TimingEvent


@dataclass
class GrooveRadar:
    """DDR-style groove radar values"""
    stream: float   # Step density (0-200+ range)
    voltage: float  # Peak intensity (0-200+ range)
    air: float      # Jump density (0-200+ range)
    freeze: float   # Hold arrow rate (0-100+ range)
    chaos: float    # Rhythm complexity (0-200+ range)

    def to_vector(self) -> np.ndarray:
        """
        Return as normalized 5D vector for similarity computation.

        Normalizes each value to roughly [0, 1] range using typical
        maximum values from DDR charts.
        """
        return np.array([
            min(self.stream / 200.0, 1.0),
            min(self.voltage / 200.0, 1.0),
            min(self.air / 200.0, 1.0),
            min(self.freeze / 100.0, 1.0),
            min(self.chaos / 200.0, 1.0)
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Return as dictionary for logging/debugging"""
        return {
            'stream': self.stream,
            'voltage': self.voltage,
            'air': self.air,
            'freeze': self.freeze,
            'chaos': self.chaos
        }


class GrooveRadarCalculator:
    """
    Calculator for DDR groove radar values.

    Implements the official DDR formulas for calculating groove radar
    values from chart data.
    """

    def __init__(self, timesteps_per_beat: int = 4, play_style: str = 'single'):
        """
        Initialize the groove radar calculator.

        Args:
            timesteps_per_beat: Resolution of the chart (4 = 16th notes)
            play_style: 'single' (4-panel) or 'double' (8-panel)
        """
        self.timesteps_per_beat = timesteps_per_beat
        self.play_style = play_style

        # Note color values for chaos calculation based on beat position
        # For 4 timesteps per beat (16th note resolution):
        # Position 0: On beat (red/4th) = 0 (no chaos contribution)
        # Position 2: 8th note (blue) = 0.5
        # Position 1, 3: 16th note (yellow) = 1.0
        # Note: We use simplified values since we have 16th note resolution
        self.note_color_values = self._build_color_values()

    def _build_color_values(self) -> Dict[int, float]:
        """
        Build note color values based on beat position.

        DDR assigns color values based on note quantization:
        - 4th notes (red): 0
        - 8th notes (blue): 0.5
        - 16th notes (yellow): 1.0
        - 12th/24th/32nd/etc (green): 1.25
        """
        # For 4 timesteps per beat (16th note resolution):
        return {
            0: 0.0,   # Quarter note (on beat) - red
            1: 1.0,   # 16th note - yellow
            2: 0.5,   # 8th note - blue
            3: 1.0,   # 16th note - yellow
        }

    def calculate(self,
                  chart_tensor: np.ndarray,
                  hold_info: Dict,
                  timing_events: List[TimingEvent],
                  song_length_seconds: float,
                  avg_bpm: float) -> GrooveRadar:
        """
        Calculate groove radar from chart data.

        Args:
            chart_tensor: Binary chart encoding (timesteps, 4)
            hold_info: Dict from convert_to_tensor_extended with hold data
            timing_events: List of TimingEvent objects
            song_length_seconds: Duration in seconds
            avg_bpm: Average BPM of the song

        Returns:
            GrooveRadar with all 5 values calculated
        """
        stream = self._calculate_stream(chart_tensor, song_length_seconds)
        voltage = self._calculate_voltage(chart_tensor, avg_bpm)
        air = self._calculate_air(chart_tensor, song_length_seconds)
        freeze = self._calculate_freeze(hold_info)
        chaos = self._calculate_chaos(chart_tensor, hold_info, timing_events,
                                       song_length_seconds)

        return GrooveRadar(
            stream=stream,
            voltage=voltage,
            air=air,
            freeze=freeze,
            chaos=chaos
        )

    def _calculate_stream(self, chart_tensor: np.ndarray,
                          song_length_seconds: float) -> float:
        """
        Calculate STREAM value.

        Formula: Average Step Density = 60 × Steps ÷ Song Length (sec)

        If Average Step Density <= 300: STREAM = Average Step Density ÷ 3
        If Average Step Density > 300:
            Single: STREAM = (Average Step Density - 139) × 100 ÷ 161
            Double: STREAM = (Average Step Density - 183) × 100 ÷ 117
        """
        # Count total steps (sum of all notes)
        total_steps = chart_tensor.sum()

        # Calculate average step density (steps per minute)
        avg_step_density = int(60 * total_steps / max(song_length_seconds, 1.0))

        # Apply DDR formula
        if avg_step_density <= 300:
            stream = avg_step_density / 3.0
        else:
            if self.play_style == 'single':
                stream = (avg_step_density - 139) * 100 / 161
            else:  # double
                stream = (avg_step_density - 183) * 100 / 117

        return stream

    def _calculate_voltage(self, chart_tensor: np.ndarray,
                           avg_bpm: float) -> float:
        """
        Calculate VOLTAGE value.

        Peak Density = max notes in 4 consecutive beats
        Average Peak Density = Average BPM × Peak Density ÷ 4

        If Average Peak Density <= 600: VOLTAGE = Average Peak Density ÷ 6
        If Average Peak Density > 600:
            VOLTAGE = (Average Peak Density + 594) × 100 ÷ 1194
        """
        # Peak density: max notes in 4 consecutive beats (16 timesteps at 4tpb)
        window_size = 4 * self.timesteps_per_beat
        notes_per_timestep = chart_tensor.sum(axis=1)

        if len(notes_per_timestep) < window_size:
            peak_density = float(notes_per_timestep.sum())
        else:
            # Efficient sliding window using convolution
            kernel = np.ones(window_size)
            windowed_sums = np.convolve(notes_per_timestep, kernel, mode='valid')
            peak_density = float(windowed_sums.max())

        # Calculate average peak density
        avg_peak_density = int(avg_bpm * peak_density / 4)

        # Apply DDR formula
        if avg_peak_density <= 600:
            voltage = avg_peak_density / 6.0
        else:
            voltage = (avg_peak_density + 594) * 100 / 1194

        return voltage

    def _calculate_air(self, chart_tensor: np.ndarray,
                       song_length_seconds: float) -> float:
        """
        Calculate AIR value.

        Formula: Average Air Density = 60 × Jumps ÷ Song Length (sec)

        If Average Air Density <= 55: AIR = Average Air Density × 20 ÷ 11
        If Average Air Density > 55:
            Single: AIR = (Average Air Density + 36) × 100 ÷ 91
            Double: AIR = (Average Air Density + 35) × 10 ÷ 9
        """
        # Count jumps (timesteps with 2+ simultaneous notes)
        notes_per_timestep = chart_tensor.sum(axis=1)
        total_jumps = (notes_per_timestep >= 2).sum()

        # Calculate average air density (jumps per minute)
        avg_air_density = int(60 * total_jumps / max(song_length_seconds, 1.0))

        # Apply DDR formula
        if avg_air_density <= 55:
            air = avg_air_density * 20 / 11
        else:
            if self.play_style == 'single':
                air = (avg_air_density + 36) * 100 / 91
            else:  # double
                air = (avg_air_density + 35) * 10 / 9

        return air

    def _calculate_freeze(self, hold_info: Dict) -> float:
        """
        Calculate FREEZE value.

        Formula: FA Rate = 10,000 × FA Total Length (beats) ÷ Song Length (beats)

        If FA Rate <= 3,500: FREEZE = FA Rate ÷ 35
        If FA Rate > 3,500:
            Single: FREEZE = (FA Rate + 2484) × 100 ÷ 5,984
            Double: FREEZE = (FA Rate + 2246) × 100 ÷ 5,746
        """
        total_hold_beats = hold_info.get('total_hold_beats', 0)
        song_length_beats = hold_info.get('song_length_beats', 1)

        # Calculate FA rate
        fa_rate = int(10000 * total_hold_beats / max(song_length_beats, 1))

        # Apply DDR formula
        if fa_rate <= 3500:
            freeze = fa_rate / 35.0
        else:
            if self.play_style == 'single':
                freeze = (fa_rate + 2484) * 100 / 5984
            else:  # double
                freeze = (fa_rate + 2246) * 100 / 5746

        return freeze

    def _calculate_chaos(self,
                         chart_tensor: np.ndarray,
                         hold_info: Dict,
                         timing_events: List[TimingEvent],
                         song_length_seconds: float) -> float:
        """
        Calculate CHAOS value.

        CHAOS is determined by:
        1. Total Chaos Base Value: Sum of note chaos values based on:
           - Note quantization (position within beat)
           - Interval from last note
           - Note color value
           - Number of arrows

        2. Total BPM Delta: Sum of BPM changes

        3. Average BPM Delta = 60 × Total BPM Delta ÷ Song Length (sec)

        4. Chaos Degree = Total Chaos Base Value × (1 + (Average BPM Delta ÷ 1,500))

        5. Unit Chaos Degree = Chaos Degree × 100 ÷ Song Length (sec)

        If Unit Chaos Degree <= 2,000: CHAOS = Unit Chaos Degree ÷ 20
        If Unit Chaos Degree > 2,000:
            Single: CHAOS = (Unit Chaos Degree + 21,605) × 100 ÷ 23,605
            Double: CHAOS = (Unit Chaos Degree + 16,628) × 100 ÷ 18,628
        """
        # Calculate Total Chaos Base Value
        total_chaos_base = self._calculate_chaos_base_value(chart_tensor, hold_info)

        # Calculate Total BPM Delta
        total_bpm_delta = self._compute_bpm_delta(timing_events)

        # Calculate Average BPM Delta (per minute)
        avg_bpm_delta = 60 * total_bpm_delta / max(song_length_seconds, 1)

        # Calculate Chaos Degree
        chaos_degree = total_chaos_base * (1 + (avg_bpm_delta / 1500))

        # Calculate Unit Chaos Degree
        unit_chaos_degree = int(chaos_degree * 100 / max(song_length_seconds, 1))

        # Apply DDR formula
        if unit_chaos_degree <= 2000:
            chaos = unit_chaos_degree / 20.0
        else:
            if self.play_style == 'single':
                chaos = (unit_chaos_degree + 21605) * 100 / 23605
            else:  # double
                chaos = (unit_chaos_degree + 16628) * 100 / 18628

        return chaos

    def _calculate_chaos_base_value(self, chart_tensor: np.ndarray,
                                     hold_info: Dict) -> float:
        """
        Calculate Total Chaos Base Value from note data.

        For each note:
        Chaos Base Value = (Note Quantization ÷ Interval from Last Note)
                          × Note Color Value × Number of Arrows

        Note: The first note's Chaos Base Value is always 0.
        """
        note_beats = hold_info.get('note_beats', [])

        if len(note_beats) <= 1:
            return 0.0

        # Sort notes by beat position
        sorted_notes = sorted(note_beats, key=lambda x: x[0])

        total_chaos = 0.0
        last_beat = sorted_notes[0][0]

        for i, (beat_pos, panel_idx, note_type) in enumerate(sorted_notes):
            if i == 0:
                # First note has chaos value of 0
                continue

            # Calculate interval from last note (in beats)
            interval = beat_pos - last_beat
            if interval <= 0:
                # Notes at same position - use a small interval
                interval = 0.0625  # 1/16th beat minimum

            # Determine note quantization (position within beat)
            beat_fraction = beat_pos % 1.0
            # Map to timestep position within beat
            timestep_in_beat = int(round(beat_fraction * self.timesteps_per_beat)) % self.timesteps_per_beat

            # Get note color value
            color_value = self.note_color_values.get(timestep_in_beat, 1.0)

            # Count number of arrows at this position
            # (simplified: we count per-note, not per-timestep)
            num_arrows = 1
            if note_type == 'hold_start':
                num_arrows = 1  # Hold starts count as 1

            # Calculate chaos base value for this note
            # Note quantization is approximated by the color value weight
            note_quantization = self.timesteps_per_beat / max(
                beat_fraction * self.timesteps_per_beat if beat_fraction > 0 else self.timesteps_per_beat, 1)

            chaos_value = (note_quantization / interval) * color_value * num_arrows
            total_chaos += chaos_value

            last_beat = beat_pos

        return total_chaos

    def _compute_bpm_delta(self, timing_events: List[TimingEvent]) -> float:
        """
        Compute total BPM delta from timing events.

        Total BPM Delta is the sum of absolute BPM changes throughout the song.
        """
        bpm_events = [e for e in timing_events if e.event_type == 'bpm']

        if len(bpm_events) <= 1:
            return 0.0

        # Sort by beat position
        bpm_events = sorted(bpm_events, key=lambda e: e.beat)

        total_delta = 0.0
        for i in range(1, len(bpm_events)):
            delta = abs(bpm_events[i].value - bpm_events[i - 1].value)
            total_delta += delta

        return total_delta


def calculate_groove_radar_from_chart(chart_tensor: np.ndarray,
                                       hold_info: Dict,
                                       timing_events: List[TimingEvent],
                                       song_length_seconds: float,
                                       avg_bpm: float,
                                       play_style: str = 'single') -> GrooveRadar:
    """
    Convenience function to calculate groove radar from chart data.

    Args:
        chart_tensor: Binary chart encoding (timesteps, 4)
        hold_info: Dict from convert_to_tensor_extended
        timing_events: List of TimingEvent objects
        song_length_seconds: Duration in seconds
        avg_bpm: Average BPM of the song
        play_style: 'single' or 'double'

    Returns:
        GrooveRadar with all 5 values
    """
    calculator = GrooveRadarCalculator(play_style=play_style)
    return calculator.calculate(
        chart_tensor=chart_tensor,
        hold_info=hold_info,
        timing_events=timing_events,
        song_length_seconds=song_length_seconds,
        avg_bpm=avg_bpm
    )
