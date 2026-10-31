# Labelling guide (for the vet and the team)

The full step-by-step workflow, with commands, is in [`finetune.md`](finetune.md).
This page is the short version for the people who draw the boxes.

## What to label

| Class | Id | When |
|---|---|---|
| `broiler` | 0 | every bird whose body is at least half visible |
| `dead` | 1 | only birds confirmed dead on site by the farmer or vet |

- One tight box around the body. Include the tail, exclude the legs if they stick out.
- A bird partly behind a feeder or drinker: box only the visible part.
- Birds pressed against the walls count too. Do not skip crowded areas.
- Unsure whether something is a bird? Leave it out rather than guess.

## Which frames

- Use the frames picked by `pews sample-frames`. They are spread over all
  cameras, all ages and day and night.
- A yellow chick looks very different from a heavy white broiler, so every
  week of the flock must be covered.
- **Always** add frames of birds that were marked "The bird was dead" or
  "It was alive" in the panel. These are the hardest and most valuable examples.

## Tools

CVAT or Label Studio, both free and installable locally. Import the images
together with the draft labels from `pews prelabel` and correct them; export
in YOLO format.

## Why the effort matters

In a study on laying hens, dead-bird detection dropped noticeably at 2 m
camera height compared with 0.5 m. The accuracy that counts is the one
measured on your own cameras (`pews evaluate`), not the one in papers.
