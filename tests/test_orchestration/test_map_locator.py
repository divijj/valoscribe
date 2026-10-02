from __future__ import annotations
from pathlib import Path
import json
import pytest
import cv2
import numpy as np

from valoscribe.orchestration.map_locator import MapLocator


# Landmarks chosen so the transform is a simple scale: pixel / 400 -> normalized
LANDMARKS = [
    {"name": "top-left", "minimap_px": [0, 0], "normalized": [0.0, 0.0]},
    {"name": "top-right", "minimap_px": [400, 0], "normalized": [1.0, 0.0]},
    {"name": "bottom-left", "minimap_px": [0, 400], "normalized": [0.0, 1.0]},
]

GRAPH = {
    "map": "testmap",
    "regions": {
        "1": {"name": "A Main", "zone": "A", "parent": "A"},
        "2": {"name": "A Site", "zone": "A", "parent": "A"},
        "3": {"name": "B Site", "zone": "B", "parent": "B"},
    },
    "edges": [
        {"from": "A Main", "to": "A Site", "type": "open", "bidirectional": True},
        {"from": "A Site", "to": "B Site", "type": "drop", "bidirectional": False},
    ],
}


@pytest.fixture
def maps_dir(tmp_path):
    """Build a synthetic config/maps/testmap/ directory."""
    map_dir = tmp_path / "testmap"
    map_dir.mkdir(parents=True)

    (map_dir / "calibration.json").write_text(json.dumps({"map": "testmap", "landmarks": LANDMARKS}))
    (map_dir / "graph.json").write_text(json.dumps(GRAPH))

    # 100x100 label mask: left half = region 1, top-right = region 2,
    # bottom-right = region 3, with a 10px unlabeled (0) border on the right edge
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :50] = 1
    mask[:50, 50:90] = 2
    mask[50:, 50:90] = 3
    cv2.imwrite(str(map_dir / "regions.png"), mask)

    return tmp_path


@pytest.fixture
def locator(maps_dir):
    """MapLocator for the synthetic map."""
    return MapLocator("testmap", maps_dir=maps_dir)


class TestMapLocator:
    """Tests for MapLocator class."""

    def test_map_name_is_lowercased(self, maps_dir):
        """VLR metadata may capitalise map names."""
        assert MapLocator("TestMap", maps_dir=maps_dir).map_name == "testmap"

    def test_missing_map_raises(self, tmp_path):
        """A map with no data directory fails loudly, not silently."""
        with pytest.raises(FileNotFoundError):
            MapLocator("nonexistent", maps_dir=tmp_path)

    def test_to_normalized_at_landmarks(self, locator):
        """Landmark pixels map to their declared normalized coords."""
        for landmark in LANDMARKS:
            x, y = locator.to_normalized(*landmark["minimap_px"])
            assert x == pytest.approx(landmark["normalized"][0], abs=1e-6)
            assert y == pytest.approx(landmark["normalized"][1], abs=1e-6)

    def test_to_normalized_interpolates(self, locator):
        """Points between landmarks scale linearly."""
        x, y = locator.to_normalized(200, 100)
        assert x == pytest.approx(0.5, abs=1e-6)
        assert y == pytest.approx(0.25, abs=1e-6)

    def test_region_at_known_point(self, locator):
        """Each quadrant of the mask resolves to its region name."""
        assert locator.region_at(0.25, 0.25) == "A Main"
        assert locator.region_at(0.70, 0.25) == "A Site"
        assert locator.region_at(0.70, 0.75) == "B Site"

    def test_region_at_unlabeled_returns_none(self, locator):
        """Mask value 0 means unlabeled, not region id 0."""
        assert locator.region_at(0.95, 0.5) is None

    def test_region_at_out_of_bounds_returns_none(self, locator):
        """Coordinates outside the map never raise."""
        assert locator.region_at(1.5, 0.5) is None
        assert locator.region_at(-0.1, 0.5) is None
        assert locator.region_at(0.5, 2.0) is None

    def test_zone_of(self, locator):
        """Regions roll up into zones."""
        assert locator.zone_of("A Main") == "A"
        assert locator.zone_of("B Site") == "B"
        assert locator.zone_of("Nonexistent") is None

    def test_are_adjacent_symmetric_for_bidirectional_edges(self, locator):
        """Bidirectional edges connect both ways."""
        assert locator.are_adjacent("A Main", "A Site")
        assert locator.are_adjacent("A Site", "A Main")

    def test_are_adjacent_one_way_for_directed_edges(self, locator):
        """A drop is traversable in one direction only."""
        assert locator.are_adjacent("A Site", "B Site")
        assert not locator.are_adjacent("B Site", "A Site")

    def test_are_adjacent_false_for_distant_regions(self, locator):
        """Regions with no edge between them are not adjacent."""
        assert not locator.are_adjacent("A Main", "B Site")

    def test_are_adjacent_true_for_same_region(self, locator):
        """Staying put is always valid (no movement to validate)."""
        assert locator.are_adjacent("A Main", "A Main")

    def test_are_adjacent_unknown_region_is_false(self, locator):
        """An unknown region name must not raise."""
        assert not locator.are_adjacent("Nowhere", "A Site")