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

Each signal is computed in two modes. The colour mode works without any
training; the per-bird mode needs a detector fine-tuned on this farm
([`docs/finetune.md`](docs/finetune.md)) and is more accurate.

| # | Signal | Colour mode (stage 1) | Per-bird mode (stage 2/3) |
|---|---|---|---|
| 1 | **Flock activity** | optical flow: share of bird pixels moving (`activity_index`) | share of inactive birds and speed of active ones from tracks (`inactive_frac`, `active_speed`) |
| 2 | **Spatial distribution** | share of bird pixels at feed/water and walls/corners (`resource_frac`, `periphery_frac`) | bird counts per zone and each bird's time at the feeders (`resource_count_frac`, `periphery_count_frac`, `feeder_time_frac`) |
| 3 | **Mortality** | bird-sized blob that never moves for hours while the flock is active | track that never moves, or a bird classified `dead`; plus `bird_count` |
| 4 | **Daily rhythm** | last 24 h vs usual hourly profile (`rhythm_corr`, `rhythm_amplitude`), dark hours of the lighting programme excluded | same, from per-bird activity |

Daily mortality entered by hand (`daily_mortality_pct`) feeds signal 3 in both modes.

Methods, the paper behind each metric, and limitations: [`docs/signals.md`](docs/signals.md).

**What these cameras cannot see** (last section of the project document): faecal colour and consistency, comb colour, and open-mouth breathing. These need close-up cameras or another method. The system does not alert on them.

## Data flow

```
DVR / cameras -> farm_data/raw/cam1/2025-12-09/05.45.00-05.50.00RO000.h264
                      |   (one folder per camera; recording time read from the file name)
                      v
  analyzer   -- each clip, sampled at 2 fps:
     |          birds: colour segmentation, or detector boxes (YOLO / MVBroTrack output)
     |          -> mapped to the floor (homography, metres)
     |          +- optical flow -> activity                      (signal 1)
     |          +- occupancy map -> distribution                 (signal 2)
     |          +- per-bird tracks (detector modes) -> inactivity, speed,
     |             zone counts, feeder time, still / dead birds  (signals 1-3)
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

### 2) Record reference data from day one

In the panel: **Vet log** (clinical signs, diagnoses, deaths found, treatments,
lighting changes) and **Daily mortality log**. Put the lighting programme in
`farm.yaml` (`lighting`). Send temperature and humidity to
`POST /api/environment` if a sensor is available. Without these records the
system's accuracy cannot be measured.

```bash
pews observe --pen pen1 --kind clinical_sign --note "tremors, 5 birds near drinker 2" --by vet
```

### 3) Fine-tune the detector (recommended)

Starting from the MVBroTrack broiler detector: sample frames, draft boxes,
correct them, train, evaluate per growth phase. Step by step: [`docs/finetune.md`](docs/finetune.md).

```bash
pews sample-frames --n 300          # or: pews import-frames <folders> --placement-date ...
pews prelabel
# correct the boxes in CVAT / Label Studio, export YOLO to farm_data/labeling/labels_corrected
pews build-dataset --labels farm_data/labeling/labels_corrected
pews train
pews evaluate --weights farm_data/runs/<run>/weights/best.pt
```

Then set `analysis.segmenter.kind: yolo` and the weights path in `farm.yaml`.

### 4) Process past clips

Process the previous flock's clips first so the baseline exists:

```bash
pews process --camera cam1 farm_data/raw/cam1/
```

### 5) Run continuously

```bash
pews watch                    # watches each camera folder and processes new clips
pews serve --port 8000        # panel: http://<server>:8000/
```

To password-protect the panel, set `PEWS_USER` and `PEWS_PASSWORD`. Expose it only on the farm LAN or a VPN.

### 6) Measure accuracy

```bash
pews validate --pen pen1 --since 2025-11-17 --out report.md
```

Compares every alert with the vet log: recall, precision, false alerts per
week, warning time, and how many mortality candidates were really dead.

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
  segmentation.py          colour segmentation (stage 1)
  detection.py             per-bird detectors: YOLO, MVBroTrack per-frame output, JSONL
  tracking.py              floor-plane tracker (Hungarian, metres)
  signals/activity.py      signal 1
  signals/distribution.py  signal 2
  signals/mortality.py     signal 3
  signals/circadian.py     signal 4
  signals/tracks.py        per-bird metrics for signals 1-3
  analyzer.py              one clip from one camera
  pipeline.py              fusion, baseline, mortality tracking, folder watching
  baseline.py              "normal" by age and hour of day
  alerts.py                alert engine and messages
  api/app.py, static/      web panel
  calibrate.py, preview.py calibration tools
  finetune.py              frame sampling / import, pre-labelling, CVAT export, dataset split, training, evaluation
  osd.py                   recording time from the timestamp burned into the frame
  validation.py            alerts vs vet log
docs/                      signals, calibration, fine-tuning, labelling guide, roadmap
notebooks/prelabel_colab.ipynb  import and pre-label frames on a free Colab GPU
tests/                     tests on synthetic video (pytest)
scripts/make_demo.py       demo data for the panel
requirements-ews.txt       early-warning system dependencies
farm_data/samples/         seven sample frames from cam1 (Nov and Dec 2025)
```

## Development stages

| Stage | Bird detection | Status |
|---|---|---|
| 1 | Colour segmentation, no training needed | **Implemented.** Flock-level activity and distribution. Weak for mortality in dense huddles |
| 2 | YOLO detector fine-tuned on this farm's frames, run inline, with floor-plane tracking | **Implemented.** Needs labelled frames: [`docs/finetune.md`](docs/finetune.md) |
| 3 | MVBroTrack inference output (`segmenter.kind: mvbrotrack`) | **Implemented** for its per-frame detections. Multi-camera ground-plane fusion: [`docs/roadmap.md`](docs/roadmap.md) |

## License and citation

The MVBroTrack code is released under **AGPL-3.0** and this repository is built on it, so the whole repository, including the early-warning system, is under the same license ([`LICENSE`](LICENSE)).

If you offer this system to others as a web service, you must also make its source code, including your changes, available to them.

If you use this work in a paper or report, cite Cardoen et al. (2025) and the MVBroTrack dataset. The citation is in `README_MVBroTrack.md`.

## Tests

```bash
pytest -q
```

There are 43 tests. They generate synthetic video (and detector output) with moving, still and dead birds and check: both activity measures, distribution and feeder time, mortality alerts from stillness and from the dead class, an activity-drop alert after 3 consecutive slots, the lighting programme, the day-wise dataset split, alert validation, idempotent re-processing, and the API.
