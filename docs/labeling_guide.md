# Labelling guide for the stage 2 model (for the vet and the team)

The stage 1 colour segmenter does not separate individual birds. To make mortality detection in huddles and bird counting reliable, a YOLO-seg model must be trained on frames from **this farm**.

## Choosing frames

- `pews extract-frames <clips> --every 120 --out farm_data/labeling`
- Take frames from **all ages**. A yellow chick looks very different from a heavy white broiler.
- Take frames from **different hours** of day and night, and from **all three cameras**.
- Initial target: about 150 frames, roughly 25 per week of the flock.
- **Always** include frames that the farmer or vet labelled "dead" or "alive" in the panel. These are the hardest and most valuable examples.

## Classes

| Class | Description |
|---|---|
| `bird` | any healthy or unknown bird, with a polygon mask around the body |
| `dead` | only when confirmed on site by the vet or farmer |
| `lethargic` | optional. Only if the vet was in the house at that time and saw the bird |

For a bird partly hidden behind equipment, label only the visible part.

## Tools and output

- Tool: CVAT or Label Studio. Both are free and can be installed locally.
- Export in **YOLO segmentation** format.
- Train with `ultralytics`. The MVBroTrack code in this repository also contains YOLO training and inference functions (`single_camera_detection/`).
- Put the weights in `models/broiler-seg.pt`.
- Set `segmenter.kind: yolo` in `farm.yaml`.

## Evaluation

Hold out 20% of the frames. Report results **per age**, because a single overall number hides weak performance in week one.

This matters: in a study on laying hens, dead-bird detection accuracy dropped noticeably at 2 m camera height compared with 0.5 m. The numbers that count are the ones measured on your own cameras, not those in papers.
