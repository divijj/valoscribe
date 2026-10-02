from __future__ import annotations
from pathlib import Path
import json
import pytest
import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
from valoscribe.detectors.minimap_detector import MinimapDetector


@pytest.fixture
def cropper():
    return Cropper()


def _make_icon(colour: tuple[int, int, int], size: int = 20) -> np.ndarray:
    """Build a fake agent icon: a filled circle with a coloured ring."""
    icon = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(icon, (size // 2, size // 2), size // 2 - 1, colour, -1)
    cv2.circle(icon, (size // 2, size // 2), size // 2 - 1, (255, 255, 255), 1)
    return icon


@pytest.fixture
def template_dir(tmp_path):
    """Write two synthetic templates: jett (attack) and sova (defense)."""
    for side, colour in (("attack", (0, 0, 200)), ("defense", (200, 180, 0))):
        side_dir = tmp_path / side
        side_dir.mkdir(parents=True)
        agent = "jett" if side == "attack" else "sova"
        cv2.imwrite(str(side_dir / f"{agent}.png"), _make_icon(colour))
    return tmp_path


@pytest.fixture
def detector(cropper, template_dir):
    """Detector loaded with the synthetic templates."""
    return MinimapDetector(cropper, templates_dir=template_dir, min_confidence=0.7)


def _frame_with_icon(cropper, icon: np.ndarray, x: int, y: int) -> np.ndarray:
    """Build a 1080p frame with an icon pasted at (x, y) inside the minimap region."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    region = cropper.regions["minimap"]
    h, w = icon.shape[:2]
    top = region["y"] + y
    left = region["x"] + x
    frame[top:top + h, left:left + w] = icon
    return frame


class TestMinimapDetector:
    """Tests for MinimapDetector class."""

    def test_init_with_missing_templates(self, cropper, tmp_path):
        """Must not crash when the template dir is absent."""
        detector = MinimapDetector(cropper, templates_dir=tmp_path / "does_not_exist")

        assert detector.templates == {}
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert detector.detect(frame) == []

    def test_loads_templates_from_both_sides(self, detector):
        """Templates are keyed by (agent, side)."""
        assert ("jett", "attack") in detector.templates
        assert ("sova", "defense") in detector.templates

    def test_detects_synthetic_icon(self, detector, cropper):
        """Paste a template into a blank minimap; expect it at the right centre."""
        icon = detector.templates[("jett", "attack")]
        frame = _frame_with_icon(cropper, icon, x=100, y=80)

        detections = detector.detect(frame)
        jett = [d for d in detections if d.agent == "jett"]

        assert len(jett) == 1
        assert jett[0].side == "attack"
        # Detection reports the icon CENTRE, not its top-left corner
        assert jett[0].x_px == pytest.approx(100 + icon.shape[1] / 2, abs=2)
        assert jett[0].y_px == pytest.approx(80 + icon.shape[0] / 2, abs=2)
        assert jett[0].confidence >= 0.7

    def test_agent_filter_limits_matching(self, detector, cropper):
        """Filtered-out agents are never returned."""
        icon = detector.templates[("jett", "attack")]
        frame = _frame_with_icon(cropper, icon, x=100, y=80)

        detector.set_agent_filter(["sova"])
        detections = detector.detect(frame)

        assert all(d.agent != "jett" for d in detections)

    def test_agent_filter_is_case_insensitive(self, detector, cropper):
        """Metadata agent names may arrive capitalised."""
        icon = detector.templates[("jett", "attack")]
        frame = _frame_with_icon(cropper, icon, x=100, y=80)

        detector.set_agent_filter(["Jett"])
        detections = detector.detect(frame)

        assert any(d.agent == "jett" for d in detections)

    def test_below_threshold_returns_nothing(self, cropper, template_dir):
        """A blank minimap yields no detections at a high threshold."""
        detector = MinimapDetector(cropper, templates_dir=template_dir, min_confidence=0.99)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        assert detector.detect(frame) == []

    def test_detections_are_within_minimap_bounds(self, detector, cropper):
        """Coordinates are relative to the minimap crop, not the full frame."""
        icon = detector.templates[("jett", "attack")]
        frame = _frame_with_icon(cropper, icon, x=100, y=80)
        region = cropper.regions["minimap"]

        for det in detector.detect(frame):
            assert 0 <= det.x_px <= region["width"]
            assert 0 <= det.y_px <= region["height"]