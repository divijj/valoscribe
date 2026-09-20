"""
detect player locs from minimap i guess


"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.detectors.cropper import Cropper
#valoscribe.types.*minimapinfo* add this later
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)

class MinimapDetector:
    """
    locate minimap then blob locate player locs and region

    """

    def __init__(
        self,
        cropper: Cropper,
        brightness_threshold: int = 180,
        min_blob_area: int = 10,
        max_blob_area: int = 500,
        min_circularity: float = 0.3,
    ):
        """
        Initialize minimap detector.

        Args:
            cropper: Cropper instance for extracting HUD regions
            brightness_threshold: Minimum brightness to consider a pixel as part of a blob (0-255)
            min_blob_area: Minimum area in pixels for a valid blob
            max_blob_area: Maximum area in pixels for a valid blob
            min_circularity: Minimum circularity (0-1) for blob validation (0.3 = fairly lenient)
        """
        self.cropper = cropper
        self.brightness_threshold = brightness_threshold
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.min_circularity = min_circularity

        log.info(
            f"Ability detector initialized (brightness_threshold: {brightness_threshold}, "
            f"blob_area: {min_blob_area}-{max_blob_area}, min_circularity: {min_circularity})"
        )


    def detect_player(
        self,
        frame: np.ndarray,
        player_index: int,
        ability_name: str,
        side: str = "left",
    ) -> Optional[AbilityInfo]:
        """
        Detect players from both teams

        Args:
            frame: Input frame (1080p)
            player_index: Player index (0-4)
            player_side: team (attack/defense)
        
        Returns:
            PlayerInfo if successfully detected, (x,y) coords and localised map region, None otherwise
        """
        # Get player info crops (includes all ability regions)
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            log.warning(f"Player index {player_index} out of range (max: {len(player_crops) - 1})")
            return None

        player_crop_data = player_crops[player_index]

        # Check side
        if player_crop_data["side"] != side:
            log.warning(
                f"Player {player_index} is on {player_crop_data['side']} side, "
                f"but {side} was requested"
            )
            return None

        # Get specific ability crop
        if ability_name not in player_crop_data:
            log.warning(f"Ability '{ability_name}' not found in player crop data")
            return None

        ability_crop = player_crop_data[ability_name]

        if ability_crop.size == 0:
            log.warning(f"Ability crop is empty for player {player_index} {ability_name}")
            return None

        # Detect blobs
        charges, total_blobs = self._count_blobs(ability_crop)

        log.debug(
            f"Player {player_index} {ability_name}: {charges} charges "
            f"({total_blobs} total blobs detected)"
        )

        return AbilityInfo(
            charges=charges,
            total_blobs_detected=total_blobs,
        )

    def _count_blobs(self, ability_crop: np.ndarray) -> tuple[int, int]:
        """
        Count bright blobs in a player crop.

        Args:
            ability_crop: Cropped ability region

        Returns:
            Tuple of (filtered_blob_count, total_blob_count)
        """
        # Preprocess
        preprocessed = self._preprocess_crop(ability_crop)

        # Find connected components (blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            preprocessed, connectivity=8
        )

        # Filter blobs by area and circularity
        valid_blobs = []

        for i in range(1, num_labels):  # Skip label 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]

            # Check area constraint
            if not (self.min_blob_area <= area <= self.max_blob_area):
                continue

            # Check circularity constraint
            # Circularity = 4π × area / perimeter²
            # Perfect circle = 1.0, lower values = less circular
            blob_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = contours[0]
                perimeter = cv2.arcLength(contour, True)

                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    if circularity >= self.min_circularity:
                        valid_blobs.append(i)

        return len(valid_blobs), num_labels - 1  # -1 to exclude background

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess player crop for blob detection.

        Creates binary image where bright pixels (player blobs) are red/blue.

        Args:
            crop: Cropped ability region

        Returns:
            Binary preprocessed image
        """
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Apply threshold to get bright pixels
        # Bright colored dots = white, dark background = black
        _, binary = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        return binary

    def detect_with_debug(
        self,
        frame: np.ndarray,
        player_index: int,
        ability_name: str,
        side: str = "left",
    ) -> tuple[Optional[AbilityInfo], np.ndarray, dict]:
        """
        Detect player and return debug visualizations.

        Args:
            frame: Input frame (1080p)
            player_loc: Player location - x,y confidence
            player_info: Name of player-agent confidence
            side: Which side 

        Returns:
            Tuple of (AbilityInfo or None, preprocessedcrop, debug_info)
        """
        # Get player info crops
        player_crops = self.cropper.crop_player_info(frame)

        if player_index >= len(player_crops):
            return None, np.array([]), {}

        player_crop_data = player_crops[player_index]

        if ability_name not in player_crop_data:
            return None, np.array([]), {}

        ability_crop = player_crop_data[ability_name]

        if ability_crop.size == 0:
            return None, np.array([]), {}

        # Preprocess
        preprocessed = self._preprocess_crop(ability_crop)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            preprocessed, connectivity=8
        )

        # Create debug info
        debug_info = {
            "total_blobs": num_labels - 1,
            "blob_stats": [],
        }

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            debug_info["blob_stats"].append({
                "label": i,
                "area": int(area),
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
            })

        # Run detection
        ability_info = self.detect_ability(frame, player_index, ability_name, side)

        return ability_info, preprocessed, debug_info
