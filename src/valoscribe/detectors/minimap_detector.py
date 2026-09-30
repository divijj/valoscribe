"""
Template-based minimap detector for Valorant HUD.

Finds agent icons on the broadcast minimap and returns their pixel
positions within the minimap crop, with attack/defense side.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.types.detections import MinimapDetection
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class MinimapDetector:
    """
    Detects agent icons on the minimap via template matching.
    1. crop the minimap;
    2. template-match the agents currently allowed;
    3. classify ring color as attack or defense;
    4. apply non-max suppression.
    """

    def __init__(
        self,
        cropper: Cropper,
        templates_dir: Optional[Path] = None,
        min_confidence: float = 0.7,
    ):
        self.cropper = cropper
        self.min_confidence = min_confidence

        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "templates" / "minimap_agents"
        self.templates_dir = Path(templates_dir)

        self.templates = self._load_templates()   # {(agent, side): image}
        self.agent_filter: Optional[set[str]] = None

        log.info(f"MinimapDetector initialized with {len(self.templates)} templates")

    def _load_templates(self) -> dict[tuple[str, str], np.ndarray]:
        """Load agent icon templates from attack/ and defense/ subdirectories."""
        templates: dict[tuple[str, str], np.ndarray] = {}
        for side in ("attack", "defense"):
            side_dir = self.templates_dir / side
            if not side_dir.exists():
                log.warning(f"Minimap template dir missing: {side_dir}")
                continue
            for path in side_dir.glob("*.png"):
                img = cv2.imread(str(path))
                if img is not None:
                    templates[(path.stem, side)] = img
        return templates

    def set_agent_filter(self, agents: list[str]) -> None:
        """Restrict matching to the agents in this game (mirrors other detectors)."""
        self.agent_filter = {a.lower() for a in agents}

    def detect(self, frame: np.ndarray) -> list[MinimapDetection]:
        """
        Detect all agent icons on the minimap.

        Returns detections keyed by agent + side; resolving those to players
        is the GameStateManager's job.
        """
        minimap = self.cropper.crop_minimap(frame)
        detections: list[MinimapDetection] = []

        for (agent, side), template in self.templates.items():
            if self.agent_filter and agent not in self.agent_filter:
                continue

            result = cv2.matchTemplate(minimap, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= self.min_confidence:
                h, w = template.shape[:2]
                detections.append(MinimapDetection(
                    agent=agent,
                    side=side,
                    x_px=max_loc[0] + w / 2,   # centre, not top-left
                    y_px=max_loc[1] + h / 2,
                    confidence=float(max_val),
                ))

    # todo: handle overlapping icons when players stack.
    
        """
        
        """
    # todo: consider masking the icon circle so background doesn't drive the match.
    
        return detections

    def __repr__(self) -> str:
        return f"MinimapDetector(templates={len(self.templates)}, min_conf={self.min_confidence})"
