"""Extraction commands for saving crops and preprocessed templates."""

import typer
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from valoscribe.video.reader import VideoReader
from valoscribe.detectors.cropper import Cropper
from valoscribe.utils.logger import setup_logging

app = typer.Typer(help="Extraction commands for saving crops and templates")
@app.command(name="score-crops")
def extract_score_crops(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    output_dir: Path = typer.Option(
        Path("./score_crops"),
        "--output",
        "-o",
        help="Output directory for crop images",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 10s)",
    ),
) -> None:
    """Extract preprocessed score crops from a VOD for template creation."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Define preprocessing function for template extraction
        def preprocess_crop(crop: np.ndarray) -> np.ndarray:
            """Preprocess score crop for template extraction (white text on black)."""
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Upscale 3x
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

            # Use fixed threshold - results in white text on black background
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # No inversion - keep white text on black background for template matching

            return binary

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")
            typer.echo(f"Output directory: {output_dir}")

            typer.echo("\n" + "=" * 60)
            typer.echo("EXTRACTING SCORE CROPS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            saved_count = 0

            for frame_info in reader:
                # Get crops
                team1_crop = cropper.crop_simple_region(frame_info.frame, "team1_score")
                team2_crop = cropper.crop_simple_region(frame_info.frame, "team2_score")

                # Save crops if they exist
                if team1_crop.size > 0 and team2_crop.size > 0:
                    # Preprocess crops (same as score detector)
                    team1_preprocessed = preprocess_crop(team1_crop)
                    team2_preprocessed = preprocess_crop(team2_crop)

                    # Create filenames with timestamp
                    timestamp_str = f"t{int(frame_info.timestamp_sec):05d}"
                    frame_str = f"f{frame_info.frame_number:07d}"
                    base_name = f"{timestamp_str}_{frame_str}"

                    # Save original crops
                    cv2.imwrite(str(output_dir / f"{base_name}_team1_original.png"), team1_crop)
                    cv2.imwrite(str(output_dir / f"{base_name}_team2_original.png"), team2_crop)

                    # Save preprocessed crops
                    cv2.imwrite(str(output_dir / f"{base_name}_team1_preprocessed.png"), team1_preprocessed)
                    cv2.imwrite(str(output_dir / f"{base_name}_team2_preprocessed.png"), team2_preprocessed)

                    saved_count += 1

                    # Display progress
                    typer.secho(
                        f"[{frame_info.timestamp_sec:7.2f}s] Saved crops #{saved_count}",
                        fg=typer.colors.GREEN,
                    )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Crop sets saved: {saved_count}")
            typer.echo(f"Output directory: {output_dir}")
            typer.echo(f"\nPreprocessed images can be used to create digit templates (0.png - 9.png)")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="credits-crops")
def extract_credits_crops(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    output_dir: Path = typer.Option(
        Path("./credits_crops"),
        "--output",
        "-o",
        help="Output directory for crop images",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to extract (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 10s)",
    ),
) -> None:
    """Extract preprocessed credits crops from a VOD for template creation."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Define preprocessing function for template extraction
        def preprocess_crop(crop: np.ndarray) -> np.ndarray:
            """Preprocess credits crop for template extraction (white icon on black)."""
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Upscale 3x
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

            # Use Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Determine which players to extract
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Extracting credits for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Extracting credits for player {player_index}")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")
            typer.echo(f"Output directory: {output_dir}")

            typer.echo("\n" + "=" * 60)
            typer.echo("EXTRACTING CREDITS CROPS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            saved_count = 0

            for frame_info in reader:
                # Get player info crops
                player_crops = cropper.crop_player_info(frame_info.frame)

                # Extract credits for each player
                for p_idx in player_indices:
                    if p_idx >= len(player_crops):
                        continue

                    player_crop_data = player_crops[p_idx]
                    side = player_crop_data.get("side", "unknown")

                    # Get credits crop
                    if "credits" not in player_crop_data:
                        continue

                    credits_crop = player_crop_data["credits"]

                    # Save crop if it exists
                    if credits_crop.size > 0:
                        # Preprocess crop
                        credits_preprocessed = preprocess_crop(credits_crop)

                        # Create filename with timestamp and player info
                        timestamp_str = f"t{int(frame_info.timestamp_sec):05d}"
                        frame_str = f"f{frame_info.frame_number:07d}"
                        player_str = f"p{p_idx}_{side}"
                        base_name = f"{timestamp_str}_{frame_str}_{player_str}"

                        # Save original crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_credits_original.png"),
                            credits_crop
                        )

                        # Save preprocessed crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_credits_preprocessed.png"),
                            credits_preprocessed
                        )

                        saved_count += 1

                        # Display progress
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): Saved crop #{saved_count}",
                            fg=typer.colors.GREEN,
                        )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Crops saved: {saved_count}")
            typer.echo(f"Output directory: {output_dir}")
            typer.echo(f"\nPreprocessed images can be used to create credits icon template")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="preround-credits-crops")
def extract_preround_credits_crops(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    output_dir: Path = typer.Option(
        Path("./preround_credits_crops"),
        "--output",
        "-o",
        help="Output directory for crop images",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to extract (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 10s)",
    ),
) -> None:
    """Extract preprocessed pre-round credits crops from a VOD for template creation."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Define preprocessing function for template extraction
        def preprocess_crop(crop: np.ndarray) -> np.ndarray:
            """Preprocess pre-round credits crop for template extraction (white icon on black)."""
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Upscale 3x
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

            # Use Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Determine which players to extract
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Extracting pre-round credits for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Extracting pre-round credits for player {player_index}")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")
            typer.echo(f"Output directory: {output_dir}")

            typer.echo("\n" + "=" * 60)
            typer.echo("EXTRACTING PRE-ROUND CREDITS CROPS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            saved_count = 0

            for frame_info in reader:
                # Get pre-round player info crops
                player_crops = cropper.crop_player_info_preround(frame_info.frame)

                # Extract credits for each player
                for p_idx in player_indices:
                    if p_idx >= len(player_crops):
                        continue

                    player_crop_data = player_crops[p_idx]
                    side = player_crop_data.get("side", "unknown")

                    # Get credits crop
                    if "credits" not in player_crop_data:
                        continue

                    credits_crop = player_crop_data["credits"]

                    # Save crop if it exists
                    if credits_crop.size > 0:
                        # Preprocess crop
                        credits_preprocessed = preprocess_crop(credits_crop)

                        # Create filename with timestamp and player info
                        timestamp_str = f"t{int(frame_info.timestamp_sec):05d}"
                        frame_str = f"f{frame_info.frame_number:07d}"
                        player_str = f"p{p_idx}_{side}"
                        base_name = f"{timestamp_str}_{frame_str}_{player_str}"

                        # Save original crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_preround_credits_original.png"),
                            credits_crop
                        )

                        # Save preprocessed crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_preround_credits_preprocessed.png"),
                            credits_preprocessed
                        )

                        saved_count += 1

                        # Display progress
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): Saved crop #{saved_count}",
                            fg=typer.colors.GREEN,
                        )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Crops saved: {saved_count}")
            typer.echo(f"Output directory: {output_dir}")
            typer.echo(f"\nPreprocessed images can be used to create pre-round credits icon template")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="timer-crops")
def extract_timer_crops(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    output_dir: Path = typer.Option(
        Path("./timer_crops"),
        "--output",
        "-o",
        help="Output directory for crop images",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 10s)",
    ),
) -> None:
    """Extract preprocessed timer crops from a VOD for template creation."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Define preprocessing function for template extraction
        def preprocess_crop(crop: np.ndarray) -> np.ndarray:
            """Preprocess timer crop for template extraction (white text on black).

            Handles both white text (normal timer) and red text (low timer).
            """
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Upscale 3x
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

            # Use Otsu's thresholding - automatically finds optimal threshold
            # This handles both white text and red text (which is darker in grayscale)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")
            typer.echo(f"Output directory: {output_dir}")

            typer.echo("\n" + "=" * 60)
            typer.echo("EXTRACTING TIMER CROPS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            saved_count = 0

            for frame_info in reader:
                # Get timer crop
                timer_crop = cropper.crop_simple_region(frame_info.frame, "round_timer")

                # Save crop if it exists
                if timer_crop.size > 0:
                    # Preprocess crop (same as score detector)
                    timer_preprocessed = preprocess_crop(timer_crop)

                    # Create filename with timestamp
                    timestamp_str = f"t{int(frame_info.timestamp_sec):05d}"
                    frame_str = f"f{frame_info.frame_number:07d}"
                    base_name = f"{timestamp_str}_{frame_str}"

                    # Save original crop
                    cv2.imwrite(str(output_dir / f"{base_name}_timer_original.png"), timer_crop)

                    # Save preprocessed crop
                    cv2.imwrite(str(output_dir / f"{base_name}_timer_preprocessed.png"), timer_preprocessed)

                    saved_count += 1

                    # Display progress
                    typer.secho(
                        f"[{frame_info.timestamp_sec:7.2f}s] Saved crop #{saved_count}",
                        fg=typer.colors.GREEN,
                    )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Crops saved: {saved_count}")
            typer.echo(f"Output directory: {output_dir}")
            typer.echo(f"\nPreprocessed images can be used to create digit templates (0.png - 9.png, colon.png)")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)




@app.command(name="dead-credits-crops")
def extract_dead_credits_crops(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    output_dir: Path = typer.Option(
        Path("./dead_credits_crops"),
        "--output",
        "-o",
        help="Output directory for crop images",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to extract (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 10s)",
    ),
) -> None:
    """Extract preprocessed dead_credits crops from a VOD for template creation."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Define preprocessing function for template extraction
        def preprocess_crop(crop: np.ndarray) -> np.ndarray:
            """Preprocess dead_credits crop for template extraction (white icon on black)."""
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Upscale 3x
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

            # Use Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Determine which players to extract
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Extracting dead_credits for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Extracting dead_credits for player {player_index}")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")
            typer.echo(f"Output directory: {output_dir}")

            typer.echo("\n" + "=" * 60)
            typer.echo("EXTRACTING DEAD_CREDITS CROPS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            saved_count = 0

            for frame_info in reader:
                # Get player info crops
                player_crops = cropper.crop_player_info(frame_info.frame)

                # Extract dead_credits for each player
                for p_idx in player_indices:
                    if p_idx >= len(player_crops):
                        continue

                    player_crop_data = player_crops[p_idx]
                    side = player_crop_data.get("side", "unknown")

                    # Get dead_credits crop
                    if "dead_credits" not in player_crop_data:
                        continue

                    dead_credits_crop = player_crop_data["dead_credits"]

                    # Save crop if it exists
                    if dead_credits_crop.size > 0:
                        # Preprocess crop
                        dead_credits_preprocessed = preprocess_crop(dead_credits_crop)

                        # Create filename with timestamp and player info
                        timestamp_str = f"t{int(frame_info.timestamp_sec):05d}"
                        frame_str = f"f{frame_info.frame_number:07d}"
                        player_str = f"p{p_idx}_{side}"
                        base_name = f"{timestamp_str}_{frame_str}_{player_str}"

                        # Save original crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_dead_credits_original.png"),
                            dead_credits_crop
                        )

                        # Save preprocessed crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_dead_credits_preprocessed.png"),
                            dead_credits_preprocessed
                        )

                        saved_count += 1

                        # Display progress
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): Saved crop #{saved_count}",
                            fg=typer.colors.GREEN,
                        )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Crops saved: {saved_count}")
            typer.echo(f"Output directory: {output_dir}")
            typer.echo(f"\nPreprocessed images can be used to create dead_credits icon template")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="health-crops")
def extract_health_crops(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    output_dir: Path = typer.Option(
        Path("./health_crops"),
        "--output",
        "-o",
        help="Output directory for crop images",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to extract (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 10s)",
    ),
) -> None:
    """Extract preprocessed health crops from a VOD for template creation."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Define preprocessing function for template extraction
        def preprocess_crop(crop: np.ndarray) -> np.ndarray:
            """Preprocess health crop for template extraction (white text on black)."""
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Upscale 3x
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

            # Use Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Determine which players to extract
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Extracting health for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Extracting health for player {player_index}")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")
            typer.echo(f"Output directory: {output_dir}")

            typer.echo("\n" + "=" * 60)
            typer.echo("EXTRACTING HEALTH CROPS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            saved_count = 0

            for frame_info in reader:
                # Get player info crops
                player_crops = cropper.crop_player_info(frame_info.frame)

                # Extract health for each player
                for p_idx in player_indices:
                    if p_idx >= len(player_crops):
                        continue

                    player_crop_data = player_crops[p_idx]
                    side = player_crop_data.get("side", "unknown")

                    # Get health crop
                    if "health" not in player_crop_data:
                        continue

                    health_crop = player_crop_data["health"]

                    # Save crop if it exists
                    if health_crop.size > 0:
                        # Preprocess crop
                        health_preprocessed = preprocess_crop(health_crop)

                        # Create filename with timestamp and player info
                        timestamp_str = f"t{int(frame_info.timestamp_sec):05d}"
                        frame_str = f"f{frame_info.frame_number:07d}"
                        player_str = f"p{p_idx}_{side}"
                        base_name = f"{timestamp_str}_{frame_str}_{player_str}"

                        # Save original crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_health_original.png"),
                            health_crop
                        )

                        # Save preprocessed crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_health_preprocessed.png"),
                            health_preprocessed
                        )

                        saved_count += 1

                        # Display progress
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): Saved crop #{saved_count}",
                            fg=typer.colors.GREEN,
                        )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Crops saved: {saved_count}")
            typer.echo(f"Output directory: {output_dir}")
            typer.echo(f"\nPreprocessed images can be used to create health digit templates")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="armor-crops")
def extract_armor_crops(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    output_dir: Path = typer.Option(
        Path("./armor_crops"),
        "--output",
        "-o",
        help="Output directory for crop images",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    player_index: int = typer.Option(
        0,
        "--player",
        "-p",
        help="Player index to extract (0-9, or -1 for all players)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 10s)",
    ),
) -> None:
    """Extract preprocessed armor crops from a VOD for template creation."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Define preprocessing function for template extraction
        def preprocess_crop(crop: np.ndarray) -> np.ndarray:
            """Preprocess armor crop for template extraction."""
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Upscale 2x for better detail
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

            # Use Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Determine which players to extract
        if player_index == -1:
            player_indices = range(10)
            typer.echo("Extracting armor for all players")
        else:
            player_indices = [player_index]
            typer.echo(f"Extracting armor for player {player_index}")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")
            typer.echo(f"Output directory: {output_dir}")

            typer.echo("\n" + "=" * 60)
            typer.echo("EXTRACTING ARMOR CROPS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            saved_count = 0

            for frame_info in reader:
                # Get player info crops
                player_crops = cropper.crop_player_info(frame_info.frame)

                # Extract armor for each player
                for p_idx in player_indices:
                    if p_idx >= len(player_crops):
                        continue

                    player_crop_data = player_crops[p_idx]
                    side = player_crop_data.get("side", "unknown")

                    # Get armor crop
                    if "armor" not in player_crop_data:
                        continue

                    armor_crop = player_crop_data["armor"]

                    # Save crop if it exists
                    if armor_crop.size > 0:
                        # Preprocess crop
                        armor_preprocessed = preprocess_crop(armor_crop)

                        # Create filename with timestamp and player info
                        timestamp_str = f"t{int(frame_info.timestamp_sec):05d}"
                        frame_str = f"f{frame_info.frame_number:07d}"
                        player_str = f"p{p_idx}_{side}"
                        base_name = f"{timestamp_str}_{frame_str}_{player_str}"

                        # Save original crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_armor_original.png"),
                            armor_crop
                        )

                        # Save preprocessed crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_armor_preprocessed.png"),
                            armor_preprocessed
                        )

                        saved_count += 1

                        # Display progress
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Player {p_idx} ({side}): Saved crop #{saved_count}",
                            fg=typer.colors.GREEN,
                        )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Crops saved: {saved_count}")
            typer.echo(f"Output directory: {output_dir}")
            typer.echo(f"\nPreprocessed images can be used to create armor templates")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="killfeed-crops")
def extract_killfeed_crops(
    video_path: Path = typer.Argument(..., help="Path to the video file to process"),
    output_dir: Path = typer.Option(
        Path("./killfeed_crops"),
        "--output",
        "-o",
        help="Output directory for crop images",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to HUD config file (default: champs2025.json)",
    ),
    entry_index: int = typer.Option(
        0,
        "--entry",
        "-e",
        help="Killfeed entry index to extract (0-9, or -1 for all entries)",
    ),
    start_time: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start time in seconds",
    ),
    end_time: Optional[float] = typer.Option(
        None,
        "--end",
        help="End time in seconds",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        "-i",
        help="Time interval between samples in seconds (default: 10s)",
    ),
) -> None:
    """Extract preprocessed killfeed crops from a VOD for template creation."""
    setup_logging()

    if not video_path.exists():
        typer.secho(f"Error: Video file not found: {video_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cropper
        if config_path:
            cropper = Cropper(config_path=config_path)
        else:
            cropper = Cropper()

        typer.echo(f"Using HUD config: {cropper.config['name']}")

        # Define preprocessing function for template extraction
        def preprocess_crop(crop: np.ndarray) -> np.ndarray:
            """Preprocess killfeed crop for OCR (white text on black background)."""
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Upscale 2x for better OCR
            h, w = gray.shape
            gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

            # Use Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        # Calculate FPS from interval
        fps = 1.0 / interval

        # Determine which entries to extract
        if entry_index == -1:
            entry_indices = range(10)
            typer.echo("Extracting all killfeed entries (0-9)")
        else:
            entry_indices = [entry_index]
            typer.echo(f"Extracting killfeed entry {entry_index}")

        # Initialize video reader
        with VideoReader(
            video_path,
            fps_filter=fps,
            start_time_sec=start_time,
            end_time_sec=end_time,
        ) as reader:
            typer.echo(f"Reading video: {video_path}")
            typer.echo(f"Resolution: {reader.width}x{reader.height}")
            typer.echo(f"FPS: {reader.fps:.2f}")
            typer.echo(f"Duration: {reader.duration_sec:.2f}s")
            typer.echo(f"Sampling interval: {interval}s")
            typer.echo(f"Output directory: {output_dir}")

            typer.echo("\n" + "=" * 60)
            typer.echo("EXTRACTING KILLFEED CROPS")
            typer.echo("=" * 60 + "\n")

            frame_count = 0
            saved_count = 0

            for frame_info in reader:
                # Get killfeed entry crops
                killfeed_entries = cropper.crop_killfeed(frame_info.frame)

                # Extract each killfeed entry
                for e_idx in entry_indices:
                    if e_idx >= len(killfeed_entries):
                        continue

                    killfeed_crop = killfeed_entries[e_idx]

                    # Save crop if it exists
                    if killfeed_crop.size > 0:
                        # Preprocess crop
                        killfeed_preprocessed = preprocess_crop(killfeed_crop)

                        # Create filename with timestamp and entry info
                        timestamp_str = f"t{int(frame_info.timestamp_sec):05d}"
                        frame_str = f"f{frame_info.frame_number:07d}"
                        entry_str = f"entry{e_idx}"
                        base_name = f"{timestamp_str}_{frame_str}_{entry_str}"

                        # Save original crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_killfeed_original.png"),
                            killfeed_crop
                        )

                        # Save preprocessed crop
                        cv2.imwrite(
                            str(output_dir / f"{base_name}_killfeed_preprocessed.png"),
                            killfeed_preprocessed
                        )

                        saved_count += 1

                        # Display progress
                        typer.secho(
                            f"[{frame_info.timestamp_sec:7.2f}s] Entry {e_idx}: Saved crop #{saved_count}",
                            fg=typer.colors.GREEN,
                        )

                frame_count += 1

            # Summary
            typer.echo("\n" + "=" * 60)
            typer.echo("SUMMARY")
            typer.echo("=" * 60)
            typer.echo(f"Frames processed: {frame_count}")
            typer.echo(f"Crops saved: {saved_count}")
            typer.echo(f"Output directory: {output_dir}")
            typer.echo(f"\nPreprocessed images can be used for killfeed OCR analysis")

    except Exception as e:
        typer.secho(f"\nError: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

# @app.command(name="minimap-crops")

# def extract_minimap(
#     video_path: Path = typer.Argument(..., help="Path to the video file to process"),
#     output_dir: Path = typer.Option(Path("./minimap_crops"),
#         "--output",
#         "-o",
#         help="Output directory for crop images",
#     ),
#     config_path: Optional[Path] = typer.Option(
#         None,
#         "--config",
#         "-c",
#         help="Path to HUD config file (default: champs2025.json)",
#     ),
#     entry_index: int = typer.Option(
#         0,
#         "--entry",
#         "-e",
#         help="Killfeed entry index to extract (0-9, or -1 for all entries)",
#     ),
#     start_time: Optional[float] = typer.Option(
#         None,
#         "--start",
#         help="Start time in seconds",
#     ),
#     end_time: Optional[float] = typer.Option(
#         None,
#         "--end",
#         help="End time in seconds",
#     ),
#     interval: float = typer.Option(
#         10.0,
#         "--interval",
#         "-i",
#         help="Time interval between samples in seconds (default: 10s)",
#     ),
# ) -> None:
