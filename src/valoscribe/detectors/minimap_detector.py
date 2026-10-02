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
        min_confidence: float = 0.6,
        use_mask: bool = True,
        min_separation: Optional[float] = None,
    ):
        self.cropper = cropper
        self.min_confidence = min_confidence
        self.use_mask = use_mask
        self.min_separation = min_separation

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

    @staticmethod
    def _circular_mask(template: np.ndarray) -> np.ndarray:
        """Circular mask covering the icon, excluding the square's corners."""
        h, w = template.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 1, 255, -1)
        return mask

    def set_agent_filter(self, agents: list[str]) -> None:
        """Restrict matching to the agents in this game (mirrors other detectors)."""
        self.agent_filter = {a.lower() for a in agents}

    def _score_map(self, minimap: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
        Confidence map in [0, 1], one value per candidate top-left position.

        With use_mask, matching uses TM_SQDIFF_NORMED with a circular mask so the
        square's corners (which hold map terrain, not icon) are ignored; the
        squared difference is inverted into a confidence. Without it, plain
        TM_CCOEFF_NORMED is used.
        """
        if self.use_mask:
            mask = self._circular_mask(template)
            result = cv2.matchTemplate(minimap, template, cv2.TM_SQDIFF_NORMED, mask=mask)
            result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
            return 1.0 - np.clip(result, 0.0, 1.0)

        result = cv2.matchTemplate(minimap, template, cv2.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result, 0.0, 1.0)

    def detect(self, frame: np.ndarray) -> list[MinimapDetection]:
        """
        Detect all agent icons on the minimap.

        Returns detections keyed by agent + side; resolving those to players is
        the GameStateManager's job. Each (agent, side) can appear at most once,
        since a team cannot run duplicate agents.
        """
        minimap = self.cropper.crop_minimap(frame)
        if minimap.size == 0 or not self.templates:
            return []

        candidates = self._collect_candidates(minimap)
        return self._suppress_overlaps(candidates)

    def _collect_candidates(self, minimap: np.ndarray) -> list[tuple]:
        """Every (confidence, agent, side, centre_x, centre_y) above threshold."""
        candidates: list[tuple] = []

        for (agent, side), template in self.templates.items():
            if self.agent_filter and agent not in self.agent_filter:
                continue

            th, tw = template.shape[:2]
            if minimap.shape[0] < th or minimap.shape[1] < tw:
                continue

            scores = self._score_map(minimap, template)
            ys, xs = np.where(scores >= self.min_confidence)

            for y, x in zip(ys, xs):
                candidates.append((
                    float(scores[y, x]),
                    agent,
                    side,
                    float(x) + tw / 2,   # centre, not top-left
                    float(y) + th / 2,
                ))

        return candidates

    def _suppress_overlaps(self, candidates: list[tuple]) -> list[MinimapDetection]:
        """
        Greedy non-maximum suppression over all templates at once.

        Highest confidence wins. A candidate is dropped when its (agent, side)
        is already placed, or when it sits within min_separation of an accepted
        detection - the same icon matching at neighbouring offsets.
        """
        detections: list[MinimapDetection] = []
        placed: set[tuple[str, str]] = set()

        for confidence, agent, side, cx, cy in sorted(candidates, key=lambda c: -c[0]):
            if (agent, side) in placed:
                continue

            separation = self.min_separation or self._default_separation(agent, side)
            if any((cx - d.x_px) ** 2 + (cy - d.y_px) ** 2 < separation ** 2 for d in detections):
                continue

            detections.append(MinimapDetection(
                agent=agent,
                side=side,
                x_px=cx,
                y_px=cy,
                confidence=confidence,
            ))
            placed.add((agent, side))

        return detections

    # Fraction of an icon width used to separate DIFFERENT agents. Kept small
    # because stacked players sit only a few pixels apart; duplicate matches of
    # the SAME icon are already handled by the (agent, side) dedupe above.
    SEPARATION_RATIO = 0.3

    def _default_separation(self, agent: str, side: str) -> float:
        """Minimum centre distance between two different agents' detections."""
        template = self.templates[(agent, side)]
        return template.shape[1] * self.SEPARATION_RATIO

    def __repr__(self) -> str:
        return (
            f"MinimapDetector(templates={len(self.templates)}, "
            f"min_conf={self.min_confidence}, mask={self.use_mask})"
        )