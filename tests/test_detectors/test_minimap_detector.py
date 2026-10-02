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
    """Write synthetic templates for three agents on both sides.

    Attack and defense use clearly different colours, the way the real
    minimap uses team-coloured rings.
    """
    palettes = {
        "attack": {"jett": (0, 0, 200), "sova": (0, 90, 220), "omen": (0, 40, 160)},
        "defense": {"jett": (200, 180, 0), "sova": (220, 120, 0), "omen": (160, 200, 40)},
    }
    for side, agents in palettes.items():
        side_dir = tmp_path / side
        side_dir.mkdir(parents=True)
        for agent, colour in agents.items():
            cv2.imwrite(str(side_dir / f"{agent}.png"), _make_icon(colour))
    return tmp_path


@pytest.fixture
def detector(cropper, template_dir):
    """Detector loaded with the synthetic templates."""
    return MinimapDetector(cropper, templates_dir=template_dir, min_confidence=0.7)


def _frame_with_icons(cropper, placements: list[tuple[np.ndarray, int, int]]) -> np.ndarray:
    """Build a 1080p frame with icons pasted at (x, y) inside the minimap region.

    The minimap area is filled with noise so matches have to beat a textured
    background rather than flat black.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    region = cropper.regions["minimap"]

    rng = np.random.default_rng(0)
    noise = rng.integers(20, 70, (region["height"], region["width"], 3), dtype=np.uint8)
    frame[region["y"]:region["y"] + region["height"],
          region["x"]:region["x"] + region["width"]] = noise

    for icon, x, y in placements:
        h, w = icon.shape[:2]
        frame[region["y"] + y:region["y"] + y + h,
              region["x"] + x:region["x"] + x + w] = icon
    return frame


def _frame_with_icon(cropper, icon: np.ndarray, x: int, y: int) -> np.ndarray:
    """Single-icon convenience wrapper."""
    return _frame_with_icons(cropper, [(icon, x, y)])


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

    def test_detects_two_icons_same_frame(self, detector, cropper):
        """Two different agents are both reported (NMS must not collapse them)."""
        jett = detector.templates[("jett", "attack")]
        sova = detector.templates[("sova", "attack")]
        frame = _frame_with_icons(cropper, [(jett, 60, 60), (sova, 200, 150)])

        detections = detector.detect(frame)
        found = {d.agent: d for d in detections}

        assert {"jett", "sova"} <= set(found)
        assert found["jett"].x_px == pytest.approx(60 + jett.shape[1] / 2, abs=2)
        assert found["sova"].x_px == pytest.approx(200 + sova.shape[1] / 2, abs=2)

    def test_same_agent_reported_once(self, detector, cropper):
        """One icon must not yield several detections at neighbouring offsets."""
        jett = detector.templates[("jett", "attack")]
        frame = _frame_with_icon(cropper, jett, x=120, y=140)

        jett_hits = [d for d in detector.detect(frame) if d.agent == "jett"]

        assert len(jett_hits) == 1

    def test_mirror_comp_separated_by_side(self, detector, cropper):
        """The same agent on both teams resolves to two detections, one per side."""
        attack_jett = detector.templates[("jett", "attack")]
        defense_jett = detector.templates[("jett", "defense")]
        frame = _frame_with_icons(cropper, [(attack_jett, 50, 50), (defense_jett, 260, 260)])

        jett_hits = {d.side: d for d in detector.detect(frame) if d.agent == "jett"}

        assert set(jett_hits) == {"attack", "defense"}
        assert jett_hits["attack"].x_px < jett_hits["defense"].x_px

    def test_nearby_agents_both_detected(self, detector, cropper):
        """Players standing close together stay separate detections."""
        jett = detector.templates[("jett", "attack")]
        sova = detector.templates[("sova", "attack")]
        gap = jett.shape[1] + 2        # adjacent icons, not overlapping
        frame = _frame_with_icons(cropper, [(jett, 100, 100), (sova, 100 + gap, 100)])

        agents = {d.agent for d in detector.detect(frame)}

        assert {"jett", "sova"} <= agents

    def test_mask_mode_ignores_corner_background(self, cropper, template_dir):
        """With masking on, background in the square's corners must not break the match."""
        detector = MinimapDetector(cropper, templates_dir=template_dir, use_mask=True)
        jett = detector.templates[("jett", "attack")]

        noisy = jett.copy()
        noisy[0:3, 0:3] = 255          # corner pixels the circular mask excludes
        noisy[-3:, -3:] = 255
        frame = _frame_with_icon(cropper, noisy, x=80, y=80)

        assert any(d.agent == "jett" for d in detector.detect(frame))

    def test_unmasked_mode_still_works(self, cropper, template_dir):
        """The CCOEFF path remains available for comparison on real footage."""
        detector = MinimapDetector(cropper, templates_dir=template_dir, use_mask=False)
        jett = detector.templates[("jett", "attack")]
        frame = _frame_with_icon(cropper, jett, x=90, y=70)

        assert any(d.agent == "jett" for d in detector.detect(frame))