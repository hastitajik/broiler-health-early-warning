# Roadmap

## Built

- Colour segmentation (stage 1), homography and floor map
- Per-bird detection modes: inline YOLO, MVBroTrack per-frame output, generic JSONL
- Floor-plane tracker and per-bird metrics (`signals/tracks.py`)
- The four signals in both modes, baseline by age and hour of day, lighting-aware rhythm
- Alert engine, web panel, vet log and daily mortality log
- Fine-tuning workflow: frame sampling, pre-labelling, day-wise split, training,
  per-phase evaluation ([`finetune.md`](finetune.md))
- Validation of alerts against the vet log (`pews validate`)
- Tests on synthetic video

## Next steps on real data (in this order)

1. **Calibration.** Measure the house and calibrate all three cameras. Current values are estimates.
2. **Reference data from today on.** Vet log in the panel (clinical signs,
   diagnoses, deaths, treatments), daily mortality, lighting programme in
   `farm.yaml`, and a temperature/humidity sensor (`POST /api/environment`).
3. **Fine-tune the detector** ([`finetune.md`](finetune.md)) and switch
   `segmenter.kind` to `yolo`.
4. **Replay a full flock** (`pews process`) to build the baseline, then
   `pews validate` against the vet log. Tune `rules.yaml` from the report.

## Later

- Multi-camera fusion of tracks on the shared ground plane, using the
  MVBroTrack ground-plane code (`ground_plane_detection/`, `ground_plane_tracking/`)
  instead of the simple per-camera tracker, if the cameras overlap.
- A `dead` class trained from the birds confirmed in the panel.
- A small converter from `mvt_main.py` tracks to per-bird metrics, once its
  output on this farm's clips has been inspected.

## Out of reach for these cameras (project document)

- faecal colour and consistency
- comb colour
- open-mouth breathing and panting

These need close-up cameras or sound. A low-cost option is **audio**: detecting
sneezes and respiratory sounds with a microphone, as studied in PoultryFI
(arXiv 2510.15757) and by Carpentier et al.
