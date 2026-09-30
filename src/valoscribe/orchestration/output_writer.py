"""Output writer for frame states and event logs."""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Optional

from valoscribe.orchestration.phase_detector import Phase
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class OutputWriter:
    """
    Writes game state and events to files.

    Creates two output files:
    1. frame_states.csv - Frame-by-frame game state (CSV format)
    2. event_log.jsonl - Discrete game events (JSON Lines format)

    The writer handles formatting, headers, and incremental writing.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize output writer.

        Args:
            output_dir: Directory to write output files to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frame_states_path = self.output_dir / "frame_states.csv"
        self.event_log_path = self.output_dir / "event_log.jsonl"

        # File handles and writers
        self.frame_states_file: Optional[object] = None
        self.event_log_file: Optional[object] = None
        self.frame_states_writer: Optional[csv.DictWriter] = None

        # Track if headers have been written
        self.frame_states_initialized = False
        self.event_log_initialized = False

        #position output writer intializating - added
        self.positions_path = self.output_dir / "minimap_positions.csv"
        self.positions_file: Optional[object] = None
        self.positions_writer: Optional[csv.DictWriter] = None
        self.positions_initialized = False
        self.POSITION_COLUMNS = ["timestamp", "round_number", "player_index", "name", "team",
        "agent", "side", "x_norm", "y_norm", "region", "zone", "confidence",
        ]

        log.info(f"OutputWriter initialized: {output_dir}")

    def _get_frame_state_columns(self) -> list[str]:
        """
        Get column names for frame_states.csv.

        Returns:
            List of column names
        """
        columns = [
            "timestamp",
            "frame_number",
            "phase",
            "round_number",
            "score_team1",
            "score_team2",
            "game_timer",
            "spike_timer",
            "post_round_timer",
        ]

        # Add columns for all 10 players
        for player_idx in range(10):
            prefix = f"player_{player_idx}_"
            columns.extend([
                f"{prefix}name",
                f"{prefix}team",
                f"{prefix}agent",
                f"{prefix}alive",
                f"{prefix}health",
                f"{prefix}armor",
                f"{prefix}ability_1",
                f"{prefix}ability_2",
                f"{prefix}ability_3",
                f"{prefix}ultimate_charges",
                f"{prefix}ultimate_full",
            ])

        return columns


    def write_frame_state(
        self,
        timestamp: float,
        frame_number: int,
        phase: Phase,
        round_number: Optional[int],
        scores: tuple[int, int],
        player_states: list[dict],
        timers: Optional[dict[str, Optional[float]]] = None,
    ) -> None:
        """
        Write a single frame state to CSV.

        Args:
            timestamp: Frame timestamp in seconds
            frame_number: Frame number
            phase: Current game phase
            round_number: Current round number (None if not started)
            scores: Tuple of (team1_score, team2_score)
            player_states: List of 10 player state dicts with metadata
            timers: Optional dict with game_timer, spike_timer, post_round_timer
        """
        # Initialize file if needed
        if not self.frame_states_initialized:
            self.frame_states_file = open(self.frame_states_path, "w", newline="")
            columns = self._get_frame_state_columns()
            self.frame_states_writer = csv.DictWriter(
                self.frame_states_file, fieldnames=columns
            )
            self.frame_states_writer.writeheader()
            self.frame_states_initialized = True
            log.debug(f"Initialized frame_states.csv with {len(columns)} columns")

        # Default timers to None if not provided
        if timers is None:
            timers = {
                "game_timer": None,
                "spike_timer": None,
                "post_round_timer": None,
            }

        # Build row
        row = {
            "timestamp": f"{timestamp:.3f}",
            "frame_number": frame_number,
            "phase": phase.name,
            "round_number": round_number if round_number is not None else "",
            "score_team1": scores[0],
            "score_team2": scores[1],
            "game_timer": f"{timers['game_timer']:.3f}" if timers['game_timer'] is not None else "",
            "spike_timer": f"{timers['spike_timer']:.3f}" if timers['spike_timer'] is not None else "",
            "post_round_timer": f"{timers['post_round_timer']:.3f}" if timers['post_round_timer'] is not None else "",
        }

        # Add player data
        for player_idx, player_state in enumerate(player_states):
            prefix = f"player_{player_idx}_"
            metadata = player_state.get("metadata", {})

            row[f"{prefix}name"] = metadata.get("name", "")
            row[f"{prefix}team"] = metadata.get("team", "")
            row[f"{prefix}agent"] = metadata.get("agent", "")

            state = player_state.get("current_state", {})
            row[f"{prefix}alive"] = state.get("alive", "")
            row[f"{prefix}health"] = state.get("health", "")
            row[f"{prefix}armor"] = state.get("armor", "")
            row[f"{prefix}ability_1"] = state.get("ability_1", "")
            row[f"{prefix}ability_2"] = state.get("ability_2", "")
            row[f"{prefix}ability_3"] = state.get("ability_3", "")

            # Ultimate
            ult = state.get("ultimate")
            if ult:
                row[f"{prefix}ultimate_charges"] = ult.get("charges", "")
                row[f"{prefix}ultimate_full"] = ult.get("is_full", "")
            else:
                row[f"{prefix}ultimate_charges"] = ""
                row[f"{prefix}ultimate_full"] = ""

        self.frame_states_writer.writerow(row)

    def write_event(self, event: dict) -> None:
        """
        Write a single event to JSONL file.

        Args:
            event: Event dictionary with 'type', 'timestamp', and other data
        """
        # Initialize file if needed
        if not self.event_log_initialized:
            self.event_log_file = open(self.event_log_path, "w")
            self.event_log_initialized = True
            log.debug("Initialized event_log.jsonl")

        # Write event as a single JSON line
        json.dump(event, self.event_log_file)
        self.event_log_file.write("\n")

    def write_events(self, events: list[dict]) -> None:
        """
        Write multiple events to JSONL file.

        Args:
            events: List of event dictionaries
        """
        for event in events:
            self.write_event(event)

    def flush(self) -> None:
        """Flush file buffers to disk."""
        if self.frame_states_file:
            self.frame_states_file.flush()
        if self.event_log_file:
            self.event_log_file.flush()
        if self.positions_file:
            self.positions_file.flush()   # added flush() for position


    def close(self) -> None:
        """Close output files."""
        if self.frame_states_file:
            self.frame_states_file.close()
            self.frame_states_file = None
            log.info(f"Closed frame_states.csv: {self.frame_states_path}")

        if self.event_log_file:
            self.event_log_file.close()
            self.event_log_file = None
            log.info(f"Closed event_log.jsonl: {self.event_log_path}")
        if self.positions_file:          # added close() for position
            self.positions_file.close()
            self.positions_file = None


    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        """String representation."""
        return f"OutputWriter(output_dir={self.output_dir})"

    #initialise and update positions - added 
    def write_positions(self, rows: list[dict]) -> None:
        """Append minimap position rows (long format, one row per detected player)."""
        if not self.positions_initialized:
            self.positions_file = open(self.positions_path, "w", newline="")
            self.positions_writer = csv.DictWriter(
                self.positions_file, fieldnames=self.POSITION_COLUMNS
            )
            self.positions_writer.writeheader()
            self.positions_initialized = True

        self.positions_writer.writerows(rows)
