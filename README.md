# Broiler Early-Warning System

This system analyses the 5-minute clips recorded by overhead cameras in a broiler house. It extracts four flock-health signals and compares each one with what is normal for the same flock age and the same hour of day. When something is off, it raises an alert by itself. A web panel shows the current state of the four signals to the farmer and the veterinarian whenever they open it.

This repository is built on the code of the **MVBroTrack** paper:
Cardoen et al., *Multi-camera detection and tracking for individual broiler monitoring*, Computers and Electronics in Agriculture, 2025.

- The paper's code is kept unchanged, with its full history, at the root of this repository: `sv_main.py`, `mv_main.py`, `mvt_main.py` and the `single_camera_*` and `ground_plane_*` folders.
- Its original README is [`README_MVBroTrack.md`](README_MVBroTrack.md).
- The early-warning system lives next to it in `src/poultry_ews/`.

> This tool does not replace a veterinary examination. Its job is to tell you earlier where to look.
> Alert thresholds must be tuned with the project veterinarian on this farm's own data.

## The four signals

These are the four signals defined in the project document.

| # | Signal | What is measured | Main metric |
|---|---|---|---|
| 1 | **Flock activity index** | Optical flow on the rectified floor map (m/s). Share of bird pixels that are moving | `activity_index` |
| 2 | **Spatial distribution** | Share of birds at feeders and drinkers, and at walls and corners. Spread of the flock | `resource_frac`, `periphery_frac` |
| 3 | **Mortality** | A bird that stays motionless in one place for hours while the flock is active. Plus daily mortality entered by hand | still-bird candidates, `daily_mortality_pct` |
| 4 | **Circadian pattern** | Similarity and strength of the last 24 h rhythm compared with previous days | `rhythm_corr`, `rhythm_amplitude` |

Methods, references and limitations of each signal: [`docs/signals.md`](docs/signals.md).

**What these cameras cannot see** (last section of the project document): faecal colour and consistency, comb colour, and open-mouth breathing. These need close-up cameras or another method. The system does not alert on them.

## Data flow

```
DVR / cameras -> farm_data/raw/cam1/2025-12-09/05.45.00-05.50.00RO000.h264
                      |   (one folder per camera; recording time read from the file name)
                      v
  analyzer   -- each clip, sampled at 2 fps:
     |          bird segmentation -> mapped to the floor (homography, metres)
     |          +- optical flow -> activity                      (signal 1)
     |          +- occupancy map -> distribution                 (signal 2)
     |          +- always-present, never-moving pixels -> still blobs  (signal 3)
     v
  pipeline   -- cameras of one pen fused per 5-minute slot
     |          baseline: median and MAD at this hour, previous days and previous flocks
     |          24 h rhythm                                      (signal 4)
     |          still-bird candidates tracked across clips
     v
  alerts     -- rules in configs/rules.yaml (z-score / threshold, persistence, cooldown)
     v
  SQLite -> web panel (FastAPI) + webhook notifications
```

## Setup

```bash
# --recurse-submodules: the MVBroTrack code needs the sort and mot3d submodules
git clone --recurse-submodules <this-repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # early-warning system only (requirements-ews.txt)
# to run the MVBroTrack code itself: README_MVBroTrack.md (conda + ./install.sh + requirements.txt)

cp configs/farm.example.yaml configs/farm.yaml
cp configs/cameras/cam1.example.yaml configs/cameras/cam1.yaml
# in farm.yaml: pen dimensions, placement date, bird count, cameras and pens
```

### 1) Calibrate each camera

```bash
# on a computer with a display (full opencv-python, not headless):
pews calibrate --camera cam1 --image frame.png --valid-from 2025-12-01

# after every change, look at the preview:
pews preview --camera cam1 --image frame.png --out preview.jpg
```

Open the preview and check that:
- the green 50 cm grid sits on the floor,
- the feeder and drinker circles are around the equipment,
- birds are red and the litter is not.

Each time a camera is moved, add a new version with `--valid-from`. Full guide: [`docs/calibration.md`](docs/calibration.md)

### 2) Process past clips

Process the previous flock's clips first so the baseline exists:

```bash
pews process --camera cam1 farm_data/raw/cam1/
```

### 3) Run continuously

```bash
pews watch                    # watches each camera folder and processes new clips
pews serve --port 8000        # panel: http://<server>:8000/
```

To password-protect the panel, set `PEWS_USER` and `PEWS_PASSWORD`. Expose it only on the farm LAN or a VPN.

### Demo data

```bash
python scripts/make_demo.py
pews serve --config demo/configs/farm.yaml
```

The demo data is simulated, not real.

## Repository layout

```
-- MVBroTrack paper code (unchanged) --
sv_main.py, mv_main.py, mvt_main.py   single-view detection, ground-plane detection, multi-view tracking
single_camera_detection/, single_camera_tracking/
ground_plane_detection/, ground_plane_tracking/
utils/, scripts/visualize_*.py, install.sh, requirements.txt
sort/, mot3d/                         submodules
README_MVBroTrack.md                  original README of that repository
LICENSE                               AGPL-3.0 (whole repository)

-- early-warning system --
configs/
  farm.example.yaml        pens, cameras, dated equipment layouts, analysis parameters
  cameras/cam1.example.yaml  calibration (dated versions)
  rules.yaml               alert rules for all four signals
src/poultry_ews/
  timeparse.py             recording time from file names
  video.py                 clip reading (mp4 / raw h264) with fast sampling
  geometry.py              homography, floor map and zones (wall / corner / feeder / drinker)
  segmentation.py          colour (stage 1) | YOLO (stage 2) | external detections (stage 3)
  signals/activity.py      signal 1
  signals/distribution.py  signal 2
  signals/mortality.py     signal 3
  signals/circadian.py     signal 4
  analyzer.py              one clip from one camera
  pipeline.py              fusion, baseline, mortality tracking, folder watching
  baseline.py              "normal" by age and hour of day
  alerts.py                alert engine and messages
  api/app.py, static/      web panel
  calibrate.py, preview.py calibration tools
docs/                      methods, calibration, labelling guide, roadmap
tests/                     tests on synthetic video (pytest)
scripts/make_demo.py       demo data for the panel
requirements-ews.txt       early-warning system dependencies
farm_data/samples/         seven sample frames from cam1 (Nov and Dec 2025)
```

## Development stages

| Stage | Bird detection | Status |
|---|---|---|
| 1 | Colour segmentation, no training needed | **Implemented.** Good enough for activity and distribution. Weak for mortality in dense huddles |
| 2 | YOLO-seg trained on this farm's frames | Adapter ready. Needs labelled data: [`docs/labeling_guide.md`](docs/labeling_guide.md) |
| 3 | MVBroTrack multi-view tracking (code in this repository) | Its output reaches the system through `ExternalDetections`; see [`docs/roadmap.md`](docs/roadmap.md) |

## License and citation

The MVBroTrack code is released under **AGPL-3.0** and this repository is built on it, so the whole repository, including the early-warning system, is under the same license ([`LICENSE`](LICENSE)).

If you offer this system to others as a web service, you must also make its source code, including your changes, available to them.

If you use this work in a paper or report, cite Cardoen et al. (2025) and the MVBroTrack dataset. The citation is in `README_MVBroTrack.md`.

## Tests

```bash
pytest -q
```

There are 27 tests. They generate synthetic video with moving and still birds and check: activity, distribution, a mortality alert after 15 minutes of stillness, an activity-drop alert after 3 consecutive slots, idempotent re-processing, and the API.
