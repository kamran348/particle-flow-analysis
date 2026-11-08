"""Configuration parsing and validation (Python 3.11+ TOML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class VideoConfig:
    path: Path = Path("data/raw/particles.avi")
    start_frame: int = 0
    stop_frame: int = 0             # 0 = no stop
    step: int = 1
    fps_override: float = 0.0       # 0 = use video metadata


@dataclass(frozen=True)
class PreprocessingConfig:
    gaussian_sigma: float = 0.7
    clahe_clip_limit: float = 0.0


@dataclass(frozen=True)
class ExclusionConfig:
    # Pixel rectangles [x0, y0, x1, y1]; right/bottom bounds are exclusive.
    rectangles: tuple[tuple[int, int, int, int], ...] = ()
    overlay_colors: tuple[str, ...] = ()
    # Expand detected colored overlays to include labels and compression fringes.
    padding: int = 60


@dataclass(frozen=True)
class PIVConfig:
    window_size: int = 32
    search_area_size: int = 48
    overlap: int = 24
    signal_to_noise_threshold: float = 1.05


@dataclass(frozen=True)
class CalibrationConfig:
    mm_per_pixel: float = 0.0       # 0 = uncalibrated; px/s


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    diameter: int = 7
    min_mass: float = 50.0
    search_range: float = 7.0
    memory: int = 1


@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    piv: PIVConfig = field(default_factory=PIVConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    exclusion: ExclusionConfig = field(default_factory=ExclusionConfig)

    def validate(self) -> None:
        v, p, a, t = self.video, self.piv, self.calibration, self.tracking
        if v.start_frame < 0 or v.stop_frame < 0 or v.step < 1:
            raise ValueError("Invalid video frame range or step")
        if v.stop_frame and v.stop_frame <= v.start_frame:
            raise ValueError("stop_frame must exceed start_frame")
        if v.fps_override < 0 or a.mm_per_pixel < 0:
            raise ValueError("fps_override and mm_per_pixel must be >= 0")
        if not (0 < p.window_size <= p.search_area_size):
            raise ValueError("Require 0 < window_size <= search_area_size")
        if not (0 <= p.overlap < p.search_area_size):
            raise ValueError("Require 0 <= overlap < search_area_size")
        if p.signal_to_noise_threshold <= 0:
            raise ValueError("Signal-to-noise threshold must be positive")
        if self.preprocessing.gaussian_sigma < 0 or self.preprocessing.clahe_clip_limit < 0:
            raise ValueError("Preprocessing settings must be nonnegative")
        if type(self.exclusion.padding) is not int or self.exclusion.padding < 0:
            raise ValueError("exclusion.padding must be a nonnegative integer")
        if any(color not in ("red", "green") for color in self.exclusion.overlay_colors):
            raise ValueError("exclusion.overlay_colors supports 'red' and 'green'")
        for rectangle in self.exclusion.rectangles:
            if (len(rectangle) != 4 or any(type(value) is not int for value in rectangle)
                    or not (0 <= rectangle[0] < rectangle[2])
                    or not (0 <= rectangle[1] < rectangle[3])):
                raise ValueError("Exclusion rectangles require integer [x0, y0, x1, y1] bounds")
        if t.diameter < 3 or t.diameter % 2 != 1:
            raise ValueError("tracking.diameter must be an odd integer >= 3")
        if t.min_mass < 0 or t.search_range <= 0 or t.memory < 0:
            raise ValueError("Invalid tracking settings")


def load_config(path: str | Path) -> AppConfig:
    """Read TOML config. Paths in TOML are relative to the working directory."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    known = {"video", "preprocessing", "piv", "calibration", "tracking", "exclusion"}
    extras = set(raw) - known
    if extras:
        raise ValueError(f"Unknown config sections: {sorted(extras)}")
    video_raw = raw.get("video", {}).copy()
    if "path" in video_raw:
        video_raw["path"] = Path(video_raw["path"])
    config = AppConfig(
        video=VideoConfig(**video_raw),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        piv=PIVConfig(**raw.get("piv", {})),
        calibration=CalibrationConfig(**raw.get("calibration", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        exclusion=ExclusionConfig(**raw.get("exclusion", {})),
    )
    config.validate()
    return config
