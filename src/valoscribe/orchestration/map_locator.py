"""
Maps minimap pixel coordinates to normalized map coordinates, callout
regions, and zones, using per-map data under config/maps/<map>/.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class MapLocator:
    """Per-map geometry: pixel -> normalized coords -> region -> zone."""

    def __init__(self, map_name: str, maps_dir: Optional[Path] = None):
        """
        Args:
            map_name: Map name from VLR metadata (e.g. "ascent"); case-insensitive
            maps_dir: Override for config/maps/
        """
        self.map_name = map_name.lower()

        if maps_dir is None:
            maps_dir = Path(__file__).parent.parent / "config" / "maps"
        self.map_dir = Path(maps_dir) / self.map_name

        self.transform = self._load_calibration()   # 2x3 affine matrix
        self.region_mask = self._load_region_mask() # HxW uint8/uint16 of region ids
        self.graph = self._load_graph()             # parsed graph.json

        # id (as str) -> {"name", "zone", "parent"}
        self.regions: dict[str, dict] = self.graph["regions"]
        # region name -> set of adjacent region names
        self.adjacency: dict[str, set[str]] = self._build_adjacency()

        log.info(f"MapLocator loaded for {self.map_name}: {len(self.regions)} regions")

    def _load_calibration(self) -> np.ndarray:
        """Build the affine transform from 3+ landmark pairs in calibration.json."""
        data = json.loads((self.map_dir / "calibration.json").read_text())
        src = np.float32([p["minimap_px"] for p in data["landmarks"][:3]])
        dst = np.float32([p["normalized"] for p in data["landmarks"][:3]])
        return cv2.getAffineTransform(src, dst)
        # TODO: if a map's minimap rotates between sides, support a per-side transform.

    def _load_region_mask(self) -> np.ndarray:
        mask = cv2.imread(str(self.map_dir / "regions.png"), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Region mask missing for {self.map_name}")
        return mask

    def _load_graph(self) -> dict:
        return json.loads((self.map_dir / "graph.json").read_text())

    def _build_adjacency(self) -> dict[str, set[str]]:
        """Flatten graph.json edges into a name -> neighbours lookup."""
        adj: dict[str, set[str]] = {r["name"]: set() for r in self.regions.values()}
        for edge in self.graph["edges"]:
            adj[edge["from"]].add(edge["to"])
            if edge.get("bidirectional", True):
                adj[edge["to"]].add(edge["from"])
        return adj

    def to_normalized(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Minimap crop pixels -> normalized map coords in [0, 1]."""
        pt = np.array([x_px, y_px, 1.0])
        x, y = self.transform @ pt
        return float(x), float(y)

    def region_at(self, x_norm: float, y_norm: float) -> Optional[str]:
        """Region name at a normalized coordinate, or None if out of bounds/unlabeled."""
        h, w = self.region_mask.shape[:2]
        col, row = int(x_norm * w), int(y_norm * h)
        if not (0 <= col < w and 0 <= row < h):
            return None
        region_id = int(self.region_mask[row, col])
        if region_id == 0:   # 0 = unlabeled
            return None
        return self.regions[str(region_id)]["name"]

    def zone_of(self, region: str) -> Optional[str]:
        for r in self.regions.values():
            if r["name"] == region:
                return r.get("zone")
        return None

    def are_adjacent(self, region_a: str, region_b: str) -> bool:
        """True if the two regions are directly connected (or identical)."""
        if region_a == region_b:
            return True
        return region_b in self.adjacency.get(region_a, set())

    def __repr__(self) -> str:
        return f"MapLocator(map={self.map_name}, regions={len(self.regions)})"
