# ParticleFlow — Python PIV/PTV video analysis

A **regular `.py` project** (no Jupyter required) to analyze seeded-particle
videos. PIV uses OpenPIV to estimate a *2D projected velocity field*; optional
PTV uses Trackpy to follow distinct particle centroids.

## Requirements

- Python **3.11+** and VS Code
- On Windows, run the commands below in the VS Code PowerShell terminal.

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,tracking]"
```

For **macOS/Linux** instead run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,tracking]'
```

In VS Code select the interpreter from `.venv` via **Python: Select Interpreter**.

## Quick start (from the project root)

```bash
python main.py demo
python main.py analyze --config configs/default.toml --output results/run1
python -m pytest
```

The `demo` command generates `data/raw/particles.avi` with approximately
uniform rightward/upward tracer motion. The `analyze` command saves results
under `results/run1/`:

- `velocity_vectors.png`: time-averaged PIV vector field
- `speed_map.png`: average-field speed heatmap
- `streamlines.png`: streamlines of the **time-averaged** field
- `velocity_field.csv`: a row per interrogation cell
- `velocity_field.npz`: complete NumPy arrays, including valid counts
- `summary.json`: metadata and basic velocity statistics
- `trajectories.csv`: only if PTV tracking is enabled

To try particle tracking as well:

```bash
python main.py analyze --config configs/default.toml --output results/tracked --track
```

### Watch individual particle tracks

```powershell
python main.py track --config configs/default.toml --output results/tracking
```

For the experimental WMV currently referenced in `configs/default.toml`, use
`--config configs/tracking.toml` as a starting preset. Its `diameter = 15` and
`min_mass = 1000` reject much of the noise picked up by the generic settings.
These values still need visual checking; faint particles may be missed.

Open `results/tracking/tracking.avi` in your video player. Colored circles mark
detected particle centers, numbers are particle IDs, and short lines show their
recent paths. The same ID keeps its color. The header shows the source frame,
source time and number of detections. No particles are drawn at predicted positions
when they are missing; trails break across missed detections.

This command runs **PTV only**, independently of PIV and `[tracking].enabled`.
It saves `trajectories.csv` and `tracking_summary.json` alongside the video.
The replay is rendered **after** detection and linking finish; it is not a live
camera viewer. It uses your configured preprocessing, exclusion and calibration.

By default `track` uses **every frame**, overriding `[video].step`, since large
gaps make individual identities harder to maintain. It retains the configured
start/stop range. To change the range or trail duration:

```powershell
python main.py track --stop-frame 100 --step 1 --trail-length 30
```

`--stop-frame` is exclusive; use `0` for the whole remaining video.
`--trail-length` counts selected frames. Playback uses `fps / step`, preserving
the elapsed video time when frames are skipped.

The tracker first converts each frame to grayscale, applies Gaussian blur and
optional CLAHE, then uses Trackpy to find bright particle centers. Detections
touching excluded areas are removed before linking. Trackpy assigns particle
IDs by linking nearby centers across selected frames. The CSV then derives
velocity from each particle's displacement and actual elapsed source time.

Tune these settings in `[tracking]` while checking the replay:

- `diameter`: odd pixel width of the particle image, including its bright spot.
- `min_mass`: minimum integrated brightness after Trackpy processing; increase
  to reject faint noise, decrease if real particles are missing.
- `search_range`: maximum linking displacement in pixels per selected frame.
  Too small breaks tracks; too large permits incorrect matches.
- `memory`: number of missed selected frames allowed before losing an ID.

An ID represents the tracker's proposed match. Crossings, dense/overlapping
spots, particles leaving the image, and low contrast can cause ID switches or
broken tracks. Inspect continuity of circles and IDs against the actual spots.

## Use your own experimental video

1. Put your `.mp4` or `.avi` file in `data/raw/`.
2. Change `[video].path` in `configs/default.toml`.
3. Check video frame rate. If metadata is wrong, set `fps_override`.
4. If you know your camera calibration, enter **mm per pixel** in
   `[calibration].mm_per_pixel`. Example: a ruler's 10 mm span covers 400 px,
   so the scale is 10/400 = **0.025 mm/pixel**.
5. Tune `[piv]` settings to particle size, displacement and image density.
6. Run `python main.py analyze`.

### Exclude annotations or unwanted regions

Use the `[exclusion]` section in the TOML config. For TACSI's moving red cross:

```toml
[exclusion]
overlay_colors = ["red"]
padding = 60
rectangles = []
```

The mask is detected from each original color frame, before grayscale conversion.
The pixel margin covers the nearby gray "Stimulate" label. Use
`overlay_colors = ["red", "green"]` to also exclude the green circle/cross,
or `[]` to turn off color detection. This detector is intended for colored
annotations on grayscale footage; it also excludes real objects of those colors.

For fixed regions, add rectangles such as `rectangles = [[430, 1010, 530, 1070]]`.
Coordinates are `[left, top, right, bottom]` in original image pixels, measured
from the top left, with right/bottom bounds exclusive. These coordinates do not
use calibrated physical units. Rectangles remain fixed, so they will not follow
a moving annotation.

Any PIV search window overlapping the mask in **either** frame is invalidated.
Those vectors become NaN and contribute neither to statistics nor to the temporal
average. Other frame pairs may still contribute valid observations at that
location after a moving annotation leaves. Optional particle tracking also rejects
detections whose neighborhoods touch the mask before linking trajectories.
Images are not filled with black, which could introduce artificial edges.
Masking removes contaminated measurements; it does not skip OpenPIV's FFT work
or significantly shorten processing time.

Each run with exclusion enabled saves `exclusion_preview.png` (magenta overlay)
and `exclusion_mask.png` (white means excluded) for the first selected frame.
Inspect the preview and adjust padding if needed. Subsequent masks are recomputed
for each frame. The chosen settings are recorded in `summary.json`.

### How the physical units work

- If `mm_per_pixel = 0`, positions are **px**, velocities are **px/s**.
- If `mm_per_pixel > 0`, positions are **mm**, velocities are **mm/s**.
- Vorticity, divergence and strain-rate magnitude are **1/s** in either case.
- `dt = (frame_b_index - frame_a_index) / fps`, including skipped frames.
- Coordinate axes are **+x right, +y up**, unlike original video pixels.

**Limitations:** The example assumes steady, approximately planar motion,
adequate tracer seeding, and an approximately stationary camera. PIV measures
apparent motion of tracers, not guaranteed local fluid velocity if particles
have slip or out-of-plane motion. For strong unsteadiness, the streamlines of
a time-average may obscure physically important transient structures.

`streamlines.png` shows integral curves of the **mean velocity field**; it is
not a recording of particle paths. Use `tracking.py` for discrete trajectories.

**Force**, **pressure**, and **volumetric flow rate** cannot generally be
recovered from a generic 2D video alone. Those require additional geometry,
fluid properties, particle properties, and/or a physical model. This starter
deliberately does not invent such estimates.

## Code organization

- `main.py`: CLI — parse commands and launch the pipeline
- `config.py`: typed settings, TOML input, validation
- `video.py`: frame streaming and metadata
- `preprocessing.py`: grayscale conversion, Gaussian blur, optional CLAHE
- `piv.py`: OpenPIV measurement + validation + Cartesian units
- `tracking.py`: optional Trackpy centroid linking and particle speed
- `tracking_video.py`: particle IDs and recent trails over original frames
- `analysis.py`: speed, vorticity, divergence, strain rate
- `plotting.py`: vector arrows, speed maps, streamlines
- `pipeline.py`: coordinate processing, temporal average, file exports
- `synthetic.py`: generate a test video
- `tests/`: isolated and integration unit tests

### Notes on algorithm selection

PIV is preferable for dense seeding. For individually discernible particles,
PTV (Trackpy) can preserve individual paths. Trackpy parameters such as
`diameter`, `min_mass`, and `search_range` require tuning. This starter uses
**single-pass** PIV; multi-pass/window deformation, camera stabilization,
confidence intervals, ROI masking and 3D calibration are natural upgrades.

## Sources

- OpenPIV Python tutorial: https://github.com/OpenPIV/openpiv-python/blob/master/openpiv/tutorials/tutorial1.py
- Trackpy docs: https://soft-matter.github.io/trackpy/
