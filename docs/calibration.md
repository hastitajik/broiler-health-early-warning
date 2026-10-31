# Camera calibration

Calibration tells the system four things:

1. **Image-to-floor mapping** (homography): at least 4 points whose pixel coordinates and floor position in metres you know.
2. **ROI**: the part of the image that is analysed. It includes the floor and the bottom of the walls, because birds press against them.
3. **Exclusion areas**: things to ignore:
   - the on-screen clock (OSD)
   - the thermometer
   - **white feeders and drinkers**, which the colour segmenter mistakes for birds
4. **Equipment positions in metres**: defined in `farm.yaml` under `layouts`.

## On-site measurements

- Measure the inner width and length of the house and write them in `farm.yaml` (`width_m` and `length_m`). The current values are **estimates**.
- The coordinate origin is the **far-left** corner as seen from the camera:
  - x to the right
  - y towards the camera
- If the near floor corner is outside the frame (in the samples the bottom edge of the image cuts the floor), pick another point whose position you know, for example the base of a drinker whose distance to the walls you have measured.

## Versions

Each time a camera is bumped or re-aimed, add a new version with `--valid-from`.

In `farm_data/samples` the camera framing changed between November and December, so `cam1.example.yaml` has two versions.

## Steps

```bash
# 1) take a frame from that day's video
pews extract-frames farm_data/raw/cam1/2025-12-09/05.45.00-05.50.00RO000.h264 --every 300 --out /tmp/f

# 2) calibrate (on a computer with a display and full opencv-python)
pews calibrate --camera cam1 --image /tmp/f/....jpg --valid-from 2025-12-01

# 3) convert equipment positions to metres (if you did not click them in step 2)
pews to-metres --camera cam1 --date 2025-12-09 1060,150 1070,370

# 4) always check the preview
pews preview --camera cam1 --image /tmp/f/....jpg --out preview.jpg
```

## What to check in the preview

- The green grid (every 50 cm) sits on the floor and its lines run parallel to the floor and wall joints.
- The orange (feeder) and blue (drinker) circles surround the base of the equipment.
- Birds are red. Litter, white equipment and walls are **not**.
  - If bright litter (e.g. in week one) turns red, raise `contrast` in `analysis.segmenter`.
  - If birds are not red, lower `v_min`.
- In the right-hand floor map, birds appear white and zones are coloured.
