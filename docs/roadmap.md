# Roadmap

## Stage 1: built

- colour segmentation, homography and floor map
- the four signals
- baseline by age and hour of day
- alert engine
- web panel
- tests on synthetic video

**Next steps on real data:**

1. Real house dimensions and calibration of all three cameras. Current values are estimates.
2. Process all clips from the November and December 2025 flock so the baseline exists.
3. Review that flock's alerts with the vet: which were right and which were wrong? Tune `rules.yaml` from that review.
4. Enter that flock's daily mortality from the farm records with `pews log-mortality`.

## Stage 2: individual bird detection

- Train YOLO-seg following [`labeling_guide.md`](labeling_guide.md)
- Individual detection enables:
  - bird counting
  - mortality detection in huddles
  - (if the class is trained) lethargic birds

## Stage 3: multi-view tracking (MVBroTrack)

The MVBroTrack code (Cardoen et al., Computers and Electronics in Agriculture, 2025) is at the root of this repository. It tracks birds from several overhead cameras on a shared ground plane. It runs in its own environment (`README_MVBroTrack.md`).

**Integration with the early-warning system:**

1. Run its single-view detector (`sv_main.py`, YOLO) on the farm's videos. Fine-tune it with labelled frames from this farm ([`labeling_guide.md`](labeling_guide.md)).
2. Save the detections of each clip next to the clip in this format:
   ```
   05.45.00-05.50.00RO000.h264.dets.jsonl
   {"frame": 0, "boxes": [[x1, y1, x2, y2], ...]}     # full-resolution pixels
   ```
3. Set `segmenter.kind: external` in `farm.yaml`.

Tracking individual birds adds:
- net displacement of each bird (the activity measure in Soster et al.)
- gait distribution
- direct detection of a bird that does not leave one spot

Two pieces are still missing for this:
- a script that converts the output of `sv_main.py` or `mvt_main.py` to `.dets.jsonl`
- a `signals/tracks.py` module for per-bird trajectory metrics

The exact output format of that code must be checked on a real run.

## Out of reach for these cameras (project document)

- faecal colour and consistency
- comb colour
- open-mouth breathing and panting

These need close-up cameras or sound. A low-cost option is **audio**: detecting sneezes and respiratory sounds with a microphone, as studied in PoultryFI (arXiv 2510.15757) and by Carpentier et al.
