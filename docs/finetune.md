# Fine-tuning the bird detector

All four signals get more accurate once every bird is found individually.
This guide takes you from raw DVR clips to a detector trained on this farm,
with its accuracy measured per growth phase.

The workflow reuses the MVBroTrack conventions: YOLO box labels, a
single "broiler" class (plus an optional "dead" class here), and test results
reported separately for the starter (< day 11), grower (day 11-23) and finisher
(>= day 24) phases.

## 0. Get the MVBroTrack detector (starting point)

The MVBroTrack dataset contains detectors already trained on top-down broiler
images from day 5 to day 38 (`yolo11x_1920.pt`, `yolov8x_1920.pt`). Starting
from them needs far fewer labels than starting from a general model.

```bash
mkdir -p DATA
wget https://datasets.ilabt.imec.be/broilertracking/MVBroTrack_v2.zip -O DATA/MVBroTrack_v2.zip
unzip DATA/MVBroTrack_v2.zip -d DATA/
ls DATA/MVBroTrack/Single-View-Detection/trained_models/
```

The download URL is the one used by `sv_main.py`. `pews prelabel` and `pews train`
use `DATA/MVBroTrack/Single-View-Detection/trained_models/yolo11x_1920.pt`
automatically when it exists.

```bash
pip install ultralytics          # PyTorch comes with it
```

## 1. Pick frames to label

```bash
pews sample-frames --n 300 --out farm_data/labeling
```

Frames are spread evenly over cameras, 3-day age buckets and lights on/off,
one frame per clip, at full resolution. `farm_data/labeling/manifest.csv`
records camera, time, age and phase of each frame.

Also copy in frames of every bird that was marked "dead" or "alive" in the
panel. They are the hardest and most useful examples.

### Frames you already exported

If you already have frames (for example in `train`, `val` and `test` folders),
import them instead of sampling new ones:

```bash
pews import-frames path/to/train path/to/val path/to/test --placement-date 2025-11-15
```

The recording date and time are read from the timestamp burned into each frame
(`osd.py`, needs `tesseract-ocr` and `pytesseract`), because file names like
`05.45.00-05.50.00[RO][0@0][0].h264_<export date>.png` carry only the time of
day. Frames are renamed `cam1_YYYYmmdd_HHMMSS.png`. An existing split is
ignored on purpose: it is redone by day in step 4.

### All of steps 1b and 2 on a free GPU

[`notebooks/prelabel_colab.ipynb`](../notebooks/prelabel_colab.ipynb) does the
import and the pre-labelling in Google Colab, reading the frames and the
MVBroTrack model from Google Drive and writing the results back:
[open in Colab](https://colab.research.google.com/github/hastitajik/broiler-health-early-warning/blob/main/notebooks/prelabel_colab.ipynb).

## 2. Draft the boxes automatically

```bash
pews prelabel --dir farm_data/labeling
```

This writes YOLO labels (`labels/<image>.txt`, plus `classes.txt`) predicted by
the MVBroTrack model, a preview JPG per frame with the boxes drawn
(`previews/`), and `cvat_prelabels_yolo1.1.zip` to import the drafts into CVAT.
Annotators then only correct the boxes instead of drawing 150 per frame.

Look at a few drafts first. How many boxes are missing or wrong tells you how
far the MVBroTrack model is from your cameras before any training.

## 3. Correct the boxes

Use CVAT (app.cvat.ai, or installed locally):

1. Create a task with labels `broiler` and `dead`, and upload the files from `images/`.
2. Actions > Upload annotations > YOLO 1.1 > `cvat_prelabels_yolo1.1.zip`.
3. Correct the boxes.
4. Export dataset > YOLO 1.1, and put the `.txt` files into
   `farm_data/labeling/labels_corrected/`.

Label Studio works as well (YOLO import and export).

Rules:
- One tight box around the body of every bird whose body is at least half visible.
- Birds behind feeders or drinkers: box only the visible part.
- Class `dead` (id 1) only for birds confirmed dead on site by the farmer or vet.
  Everything else is `broiler` (id 0).
- Do not skip crowded areas along the walls. These are where the colour
  segmenter fails and where the detector is most needed.

With drafts, correcting one frame takes a few minutes. Start with about 150
frames, train, evaluate, then label more frames of the phase that scores lowest.

## 4. Build the dataset

```bash
pews build-dataset --dir farm_data/labeling --labels farm_data/labeling/labels_corrected \
                   --out farm_data/yolo_dataset
```

The split is made **by day** within each growth phase: frames from the same
day never end up in both training and test. Otherwise near-identical frames
would inflate the test score. Defaults: 20% test, 15% validation.

Outputs: `data.yaml`, `test_starter.yaml`, `test_grower.yaml`,
`test_finisher.yaml` and `split_summary.json`.

## 5. Train

```bash
pews train --data farm_data/yolo_dataset/data.yaml --imgsz 1280 --epochs 100
```

- A GPU is strongly recommended. A free Colab or Kaggle GPU is enough for a
  few hundred images. On a CPU, training works but takes many hours.
- `--imgsz 1280`: birds are small in 2304-pixel frames, especially chicks in
  week one. Lower sizes lose them.
- `--base`: the default is the MVBroTrack `yolo11x_1920.pt` (most accurate, but
  slow at inference). If the farm computer has no GPU, also fine-tune a small
  model on the same labels (`--base yolo11s.pt`) and compare both in step 6.

The best weights are saved in `farm_data/runs/<run>/weights/best.pt`.

## 6. Evaluate

```bash
pews evaluate --weights farm_data/runs/<run>/weights/best.pt --out farm_data/detector_report.md
```

You get precision, recall and mAP on the held-out test frames, overall and
for each growth phase. For reference, Hernandez et al. (Poultry Science 2026)
reached precision 0.95 and recall 0.92 with a fine-tuned YOLOv8 on commercial
broiler houses. Aim for at least about 0.9 in every phase before relying on the
per-bird signals. If one phase is clearly lower, label more frames of that phase.

## 7. Use it

In `configs/farm.yaml`:

```yaml
analysis:
  segmenter:
    kind: yolo
    weights: farm_data/runs/<run>/weights/best.pt
    imgsz: 1280
    conf: 0.35
```

From now on the pipeline also tracks every bird and fills the per-bird metrics
(`inactive_frac`, `active_speed`, `feeder_time_frac`, `periphery_count_frac`,
`bird_count`). The panel shows them instead of the colour-based ones as soon
as they have data.

Speed: the detector runs on every sampled frame (2 per second, 600 per 5-minute
clip and camera). On a CPU this can be slower than real time for three cameras.
In that case use a GPU, a smaller model, or `analysis.analysis_fps: 1`.

Alternatively run the MVBroTrack inference code offline and point the
pipeline at its output (`segmenter.kind: mvbrotrack`, one `<clip>.dets/`
folder of `000001.txt` files per clip).
