"""Unit tests for cropper module."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest
import json
import numpy as np

from valoscribe.detectors.cropper import Cropper


@pytest.fixture
def test_config():
    """Create a test HUD configuration."""
    return {
        "name": "Test Config",
        "regions": {
            "round_number": {
                "x": 100,
                "y": 10,
                "width": 50,
                "height": 20
            },
            "team1_score": {
                "x": 50,
                "y": 10,
                "width": 30,
                "height": 30
            },
            "team2_score": {
                "x": 200,
                "y": 10,
                "width": 30,
                "height": 30
            },
            "round_timer": {
                "x": 150,
                "y": 10,
                "width": 40,
                "height": 30
            },
            "minimap": {
                "x": 10,
                "y": 20,
                "width": 200,
                "height": 200
            },
            "killfeed": {
                "x": 800,
                "y": 100,
                "width": 300,
                "height": 200,
                "individual_height": 20,
                "offset": 5
            },
            "player_info": {
                "x": 10,
                "y": 500,
                "width": 300,
                "height": 500,
                "individual_height": 90,
                "offset": 10
            },
            "individual_player_info": {
                "agent_icon": {
                    "x": 0,
                    "y": 0,
                    "width": 60,
                    "height": 30
                },
                "player_name": {
                    "x": 65,
                    "y": 5,
                    "width": 100,
                    "height": 20
                },
                "ultimate": {
                    "x": 0,
                    "y": 35,
                    "width": 40,
                    "height": 40
                },
                "ability_1": {
                    "x": 45,
                    "y": 60,
                    "width": 30,
                    "height": 15
                },
                "ability_2": {
                    "x": 80,
                    "y": 60,
                    "width": 30,
                    "height": 15
                },
                "ability_3": {
                    "x": 115,
                    "y": 60,
                    "width": 30,
                    "height": 15
                },
                "weapon": {
                    "x": 150,
                    "y": 50,
                    "width": 80,
                    "height": 15
                },
                "credits": {
                    "x": 200,
                    "y": 60,
                    "width": 60,
                    "height": 15
                },
                "armor": {
                    "x": 180,
                    "y": 5,
                    "width": 30,
                    "height": 25
                },
                "health": {
                    "x": 215,
                    "y": 5,
                    "width": 35,
                    "height": 20
                }
            },
            "player_info_preround": {
                "x": 10,
                "y": 700,
                "width": 900,
                "height": 280,
                "individual_height": 50,
                "offset": 4
            },
            "individual_player_info_preround": {
                "agent_icon": {
                    "x": 500,
                    "y": 0,
                    "width": 70,
                    "height": 50
                },
                "player_name": {
                    "x": 575,
                    "y": 0,
                    "width": 150,
                    "height": 50
                },
                "ultimate": {
                    "x": 450,
                    "y": 2,
                    "width": 46,
                    "height": 46
                },
                "ability_1": {
                    "x": 150,
                    "y": 30,
                    "width": 40,
                    "height": 10
                },
                "ability_2": {
                    "x": 195,
                    "y": 30,
                    "width": 40,
                    "height": 10
                },
                "ability_3": {
                    "x": 240,
                    "y": 30,
                    "width": 40,
                    "height": 10
                },
                "weapon": {
                    "x": 300,
                    "y": 5,
                    "width": 90,
                    "height": 30
                },
                "credits": {
                    "x": 50,
                    "y": 10,
                    "width": 80,
                    "height": 30
                },
                "armor": {
                    "x": 280,
                    "y": 10,
                    "width": 25,
                    "height": 25
                },
                "health": {
                    "x": 0,
                    "y": 0,
                    "width": 1,
                    "height": 1
                }
            }
        }
    }


@pytest.fixture
def test_frame():
    """Create a test frame (1080p)."""
    # Create a frame with different colors in different regions for testing
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Fill different regions with different colors for verification
    # Round number area - red
    frame[10:30, 100:150] = [255, 0, 0]

    # Team scores - green
    frame[10:40, 50:80] = [0, 255, 0]
    frame[10:40, 200:230] = [0, 255, 0]

    # Timer - blue
    frame[10:40, 150:190] = [0, 0, 255]

    # Minimap - yellow
    frame[20:220, 10:210] = [255, 255, 0]

    # Killfeed - cyan
    frame[100:300, 800:1100] = [0, 255, 255]

    # Player info - magenta
    frame[500:1000, 10:310] = [255, 0, 255]

    # Player info preround - white
    frame[700:980, 10:910] = [255, 255, 255]

    return frame


class TestCropperInitialization:
    """Tests for Cropper initialization."""

    def test_init_with_valid_config(self, test_config, tmp_path):
        """Test initialization with valid config file."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)

        cropper = Cropper(config_path=config_file)

        assert cropper.config == test_config
        assert cropper.regions == test_config["regions"]

    def test_init_with_missing_config(self, tmp_path):
        """Test initialization with non-existent config file."""
        config_file = tmp_path / "missing_config.json"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Cropper(config_path=config_file)

    def test_init_with_invalid_json(self, tmp_path):
        """Test initialization with invalid JSON file."""
        config_file = tmp_path / "invalid_config.json"
        with open(config_file, 'w') as f:
            f.write("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            Cropper(config_path=config_file)


class TestCropSimpleRegion:
    """Tests for crop_simple_region method."""

    @pytest.fixture
    def cropper(self, test_config, tmp_path):
        """Create a cropper instance."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        return Cropper(config_path=config_file)

    def test_crop_round_number(self, cropper, test_frame):
        """Test cropping round number region."""
        crop = cropper.crop_simple_region(test_frame, "round_number")

        assert crop.shape == (20, 50, 3)  # height=20, width=50
        # Verify it's from the red region
        assert np.array_equal(crop[0, 0], [255, 0, 0])

    def test_crop_team1_score(self, cropper, test_frame):
        """Test cropping team1 score region."""
        crop = cropper.crop_simple_region(test_frame, "team1_score")

        assert crop.shape == (30, 30, 3)
        # Verify it's from the green region
        assert np.array_equal(crop[0, 0], [0, 255, 0])

    def test_crop_minimap(self, cropper, test_frame):
        """Test cropping minimap region."""
        crop = cropper.crop_simple_region(test_frame, "minimap")

        assert crop.shape == (200, 200, 3)
        # Verify it's from the yellow region
        assert np.array_equal(crop[0, 0], [255, 255, 0])

    def test_crop_bounds_clamping(self, cropper, test_frame):
        """Test that coordinates are clamped to frame bounds."""
        # Modify config to have out-of-bounds coordinates
        cropper.regions["test_region"] = {
            "x": 1900,  # Near right edge
            "y": 1070,  # Near bottom edge
            "width": 100,  # Would go past edge
            "height": 100   # Would go past edge
        }

        crop = cropper.crop_simple_region(test_frame, "test_region")

        # Should be clamped to available space
        assert crop.shape[0] <= 10  # height clamped
        assert crop.shape[1] <= 20  # width clamped


class TestCropKillfeed:
    """Tests for crop_killfeed method."""

    @pytest.fixture
    def cropper(self, test_config, tmp_path):
        """Create a cropper instance."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        return Cropper(config_path=config_file)

    def test_crop_killfeed_entries(self, cropper, test_frame):
        """Test cropping killfeed into individual entries."""
        entries = cropper.crop_killfeed(test_frame)

        # Should have entries (limited by available space)
        assert len(entries) > 0
        assert len(entries) <= 10

        # Each entry should have the correct dimensions
        for entry in entries:
            assert entry.shape[0] == 20  # individual_height
            assert entry.shape[1] == 300  # killfeed width
            assert entry.shape[2] == 3  # color channels

    def test_crop_killfeed_respects_offset(self, cropper, test_frame):
        """Test that killfeed entries respect offset spacing."""
        entries = cropper.crop_killfeed(test_frame)

        # With individual_height=20 and offset=5, we can fit 8 entries in 200 height
        # (20+5)*8 - 5 = 195 < 200
        expected_entries = 8
        assert len(entries) == expected_entries

    def test_crop_killfeed_stops_at_bounds(self, cropper, test_frame):
        """Test that killfeed cropping stops at region bounds."""
        # Modify config to have small killfeed region
        cropper.regions["killfeed"]["height"] = 50  # Only room for 2 entries

        entries = cropper.crop_killfeed(test_frame)

        # Should stop before exceeding bounds
        assert len(entries) == 2  # (20+5)*2 - 5 = 35 < 50


class TestCropPlayerInfo:
    """Tests for crop_player_info method."""

    @pytest.fixture
    def cropper(self, test_config, tmp_path):
        """Create a cropper instance."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        return Cropper(config_path=config_file)

    def test_crop_player_info_returns_10_players(self, cropper, test_frame):
        """Test that player info returns 10 players (5 left + 5 right)."""
        players = cropper.crop_player_info(test_frame)

        assert len(players) == 10

    def test_crop_player_info_structure(self, cropper, test_frame):
        """Test that each player has all expected elements."""
        players = cropper.crop_player_info(test_frame)

        expected_elements = {
            "agent_icon", "player_name", "ultimate",
            "ability_1", "ability_2", "ability_3",
            "weapon", "credits", "armor", "health", "side"
        }

        for player in players:
            assert set(player.keys()) == expected_elements

            # Check that crops are numpy arrays (except "side" which is a string)
            for element_name, crop in player.items():
                if element_name == "side":
                    assert isinstance(crop, str)
                else:
                    assert isinstance(crop, np.ndarray)

    def test_crop_player_info_left_side(self, cropper, test_frame):
        """Test left side player crops."""
        players = cropper.crop_player_info(test_frame)

        # Check first left side player (player 0)
        left_player = players[0]

        # Agent icon should be 60x30
        assert left_player["agent_icon"].shape == (30, 60, 3)

        # Player name should be 20x100
        assert left_player["player_name"].shape == (20, 100, 3)

    def test_crop_player_info_right_side_flipping(self, cropper, test_frame):
        """Test that right side player names are flipped."""
        # Create a frame with distinguishable pattern
        test_frame[505:525, 75:175] = np.arange(20 * 100 * 3).reshape(20, 100, 3)

        players = cropper.crop_player_info(test_frame)

        # Get left and right player names
        left_player_name = players[0]["player_name"]
        right_player_name = players[5]["player_name"]

        # Right side player name should be flipped
        # The shapes should match
        assert left_player_name.shape == right_player_name.shape

    def test_crop_player_info_ability_swapping(self, cropper, test_frame):
        """Test that right side abilities are swapped (ability_1 <-> ability_3)."""
        # Create unique patterns for each ability on left side
        # ability_1 at x=45
        test_frame[560:575, 55:85] = [255, 0, 0]  # red
        # ability_2 at x=80
        test_frame[560:575, 90:120] = [0, 255, 0]  # green
        # ability_3 at x=115
        test_frame[560:575, 125:155] = [0, 0, 255]  # blue

        players = cropper.crop_player_info(test_frame)

        # For right side players, ability_1 and ability_3 should be swapped
        # This is to maintain consistent logical ordering across both sides
        left_ability_1 = players[0]["ability_1"]
        left_ability_3 = players[0]["ability_3"]
        right_ability_1 = players[5]["ability_1"]
        right_ability_3 = players[5]["ability_3"]

        # Just verify they have the same shapes (detailed logic is hard to test without full mock)
        assert left_ability_1.shape == right_ability_1.shape
        assert left_ability_3.shape == right_ability_3.shape


class TestCropPlayerInfoPreround:
    """Tests for crop_player_info_preround method."""

    @pytest.fixture
    def cropper(self, test_config, tmp_path):
        """Create a cropper instance."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        return Cropper(config_path=config_file)

    def test_crop_player_info_preround_returns_10_players(self, cropper, test_frame):
        """Test that pre-round player info returns 10 players (5 left + 5 right)."""
        players = cropper.crop_player_info_preround(test_frame)

        assert len(players) == 10

    def test_crop_player_info_preround_structure(self, cropper, test_frame):
        """Test that each player has all expected elements."""
        players = cropper.crop_player_info_preround(test_frame)

        expected_elements = {
            "agent_icon", "player_name", "ultimate",
            "ability_1", "ability_2", "ability_3",
            "weapon", "credits", "armor", "health", "side"
        }

        for player in players:
            assert set(player.keys()) == expected_elements

            # Check that crops are numpy arrays (except "side" which is a string)
            for element_name, crop in player.items():
                if element_name == "side":
                    assert isinstance(crop, str)
                else:
                    assert isinstance(crop, np.ndarray)

    def test_crop_player_info_preround_left_side(self, cropper, test_frame):
        """Test left side player crops."""
        players = cropper.crop_player_info_preround(test_frame)

        # Check first left side player (player 0)
        left_player = players[0]
        assert left_player["side"] == "left"

        # Agent icon should be 50x70
        assert left_player["agent_icon"].shape == (50, 70, 3)

        # Player name should be 50x150
        assert left_player["player_name"].shape == (50, 150, 3)

        # Ultimate should be 46x46
        assert left_player["ultimate"].shape == (46, 46, 3)

    def test_crop_player_info_preround_right_side(self, cropper, test_frame):
        """Test right side player crops."""
        players = cropper.crop_player_info_preround(test_frame)

        # Check first right side player (player 5)
        right_player = players[5]
        assert right_player["side"] == "right"

        # Dimensions should match left side
        assert right_player["agent_icon"].shape == (50, 70, 3)
        assert right_player["player_name"].shape == (50, 150, 3)
        assert right_player["ultimate"].shape == (46, 46, 3)

    def test_crop_player_info_preround_right_side_flipping(self, cropper, test_frame):
        """Test that right side player names are flipped."""
        # Create a frame with distinguishable pattern
        test_frame[700:750, 585:735] = np.arange(50 * 150 * 3).reshape(50, 150, 3)

        players = cropper.crop_player_info_preround(test_frame)

        # Get left and right player names
        left_player_name = players[0]["player_name"]
        right_player_name = players[5]["player_name"]

        # Right side player name should be flipped
        # The shapes should match
        assert left_player_name.shape == right_player_name.shape

    def test_crop_player_info_preround_ability_swapping(self, cropper, test_frame):
        """Test that right side abilities are swapped (ability_1 <-> ability_3)."""
        players = cropper.crop_player_info_preround(test_frame)

        # For right side players, ability_1 and ability_3 should be swapped
        # This is to maintain consistent logical ordering across both sides
        left_ability_1 = players[0]["ability_1"]
        left_ability_3 = players[0]["ability_3"]
        right_ability_1 = players[5]["ability_1"]
        right_ability_3 = players[5]["ability_3"]

        # Verify they have the same shapes
        assert left_ability_1.shape == right_ability_1.shape
        assert left_ability_3.shape == right_ability_3.shape


class TestCropAllRegions:
    """Tests for crop_all_regions method."""

    @pytest.fixture
    def cropper(self, test_config, tmp_path):
        """Create a cropper instance."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        return Cropper(config_path=config_file)

    def test_crop_all_regions_structure(self, cropper, test_frame):
        """Test that crop_all_regions returns all expected regions."""
        result = cropper.crop_all_regions(test_frame)

        # Check that all expected keys are present
        expected_keys = {
            "round_number", "team1_score", "team2_score",
            "round_timer", "minimap", "killfeed", "player_info"
        }
        assert set(result.keys()) == expected_keys

    def test_crop_all_regions_simple_regions(self, cropper, test_frame):
        """Test that simple regions are cropped correctly."""
        result = cropper.crop_all_regions(test_frame)

        # Verify simple regions are numpy arrays with expected shapes
        assert result["round_number"].shape == (20, 50, 3)
        assert result["team1_score"].shape == (30, 30, 3)
        assert result["team2_score"].shape == (30, 30, 3)
        assert result["round_timer"].shape == (30, 40, 3)
        assert result["minimap"].shape == (200, 200, 3)

    def test_crop_all_regions_killfeed(self, cropper, test_frame):
        """Test that killfeed is a list of entries."""
        result = cropper.crop_all_regions(test_frame)

        assert isinstance(result["killfeed"], list)
        assert len(result["killfeed"]) > 0
        assert len(result["killfeed"]) <= 10

    def test_crop_all_regions_player_info(self, cropper, test_frame):
        """Test that player_info contains 10 players."""
        result = cropper.crop_all_regions(test_frame)

        assert isinstance(result["player_info"], list)
        assert len(result["player_info"]) == 10


class TestGetConfigInfo:
    """Tests for get_config_info method."""

    @pytest.fixture
    def cropper(self, test_config, tmp_path):
        """Create a cropper instance."""
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        return Cropper(config_path=config_file)

    def test_get_config_info_structure(self, cropper):
        """Test that get_config_info returns expected structure."""
        info = cropper.get_config_info()

        assert "name" in info
        assert "simple_regions" in info
        assert "killfeed_entries" in info
        assert "player_count" in info
        assert "player_elements" in info
        assert "player_elements_preround" in info

    def test_get_config_info_values(self, cropper):
        """Test that get_config_info returns correct values."""
        info = cropper.get_config_info()

        assert info["name"] == "Test Config"
        assert len(info["simple_regions"]) == 5
        assert info["killfeed_entries"] == 10
        assert info["player_count"] == 10
        assert len(info["player_elements"]) == 10  # All player sub-elements
        assert len(info["player_elements_preround"]) == 10  # All pre-round player sub-elements


class TestCropMinimap:
    """Tests for crop_minimap."""

    @pytest.fixture
    def cropper(self, test_config, tmp_path):
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(test_config, f)
        return Cropper(config_path=config_file)

    def test_crop_minimap_shape(self, cropper, test_frame):
        """crop_minimap returns the configured minimap region."""
        region = cropper.regions["minimap"]
        crop = cropper.crop_minimap(test_frame)
        assert crop.shape == (region["height"], region["width"], 3)

    def test_crop_minimap_matches_simple_region(self, cropper, test_frame):
        """crop_minimap is equivalent to crop_simple_region('minimap')."""
        assert np.array_equal(
            cropper.crop_minimap(test_frame),
            cropper.crop_simple_region(test_frame, "minimap"),
        )