# How the four signals are computed

All measurements are made on a **rectified floor map** built with a homography, in metres (default: 100 pixels per metre).

The cameras are mounted at an angle. In the raw image a bird at the far end of the house looks smaller, and 10 cm of movement there covers fewer pixels than in front of the camera. The floor map removes that error.

Each 5-minute clip is sampled at 2 frames per second and produces one row of metrics for that 5-minute slot. When several cameras see the same pen, their results are fused on the same floor map; where views overlap, values are averaged.

## Baseline: what is "normal"?

Each metric is compared with a reference from the same pen made of:

- **this flock:** 1 to 4 days earlier, at the same hour of day (±1 hour)
- **previous flocks:** the same age (±2 days) and the same hour

The median and MAD of these samples give a robust z-score:

`z = (value − median) / (1.4826 × MAD)`

The denominator is floored at 5% of the median so that a very stable baseline does not turn noise into alerts.

With fewer than 6 samples the signal is in the **learning** state and raises no alerts. This is why processing the previous flock's clips is worthwhile.

This is the "normalised baseline per age group and time of day" described in the project document.

---

## Signal 1: flock activity index

**Method:** dense optical flow (Farneback) between two consecutive floor-map frames. Each pixel's speed is converted to m/s. Only pixels that are birds are counted.

| Metric | Definition |
|---|---|
| `activity_index` | bird pixels faster than 4 cm/s ÷ all bird pixels |
| `mean_speed` | mean speed of bird pixels (m/s) |

Dividing by bird pixels makes the index independent of how many birds are in view and of their size, so a day-old chick and a 5-week broiler are comparable.

**Rules:**
- `activity_drop`: z ≤ −2 for 3 consecutive slots
- `activity_spike`: a sudden increase, which can indicate fear, disturbance or a predator

**References:** Dawkins et al. (optical flow in commercial flocks); relation of activity with hock burn in de Jong et al. 2012 (R²=0.70).

**Limitations:**
- Lighting changes (lamps switching on or off) can raise the index for one slot. The persistence rule filters this.
- A change in the lighting programme changes the activity pattern.

## Signal 2: spatial distribution

**Method:** a floor occupancy map, i.e. for each pixel the mean "is bird" value over the clip. Zones are defined on the floor in metres:

| Zone | Definition |
|---|---|
| feeder and drinker | a circle centred on the equipment with the radius of its feeding area |
| wall | a 30 cm band along the long walls |
| corner | a 60 cm square in each corner |
| open | the rest of the floor |

The equipment layout changes during the flock (trays and small drinkers in week one, tube feeders and bell drinkers later), so layouts are defined in `farm.yaml` with `valid_from` dates.

| Metric | Definition |
|---|---|
| `resource_frac` | share of birds at feed + water. A drop means birds eat and drink less |
| `periphery_frac` | share at walls + corners. A rise means crowding at the edges |
| `dispersion_entropy` | 0 to 1. Higher means birds are spread more evenly |
| `bird_coverage` | share of the visible floor covered by birds. Rises with age |

**Interpretation for Newcastle disease and influenza** (project document): fewer birds at feed and water and more in corners, even before clear neurological signs.

**Important differential:** crowding at the walls also happens with **high temperature**. In the sample frames the thermometer showed more than 30 °C. Record temperature and humidity with `POST /api/environment` (from a sensor or by hand). Activity and distribution alerts include the latest recorded temperature.

## Signal 3: mortality

**Method** (idea from Hao et al., Agriculture 2025: a dead bird stays still in the video):

1. In each clip, find pixels that:
   - are "bird" in at least 90% of frames, and
   - move in at most 5% of frames.
2. Group these pixels into blobs. Keep only blobs between 0.4 and 2.5 times the area of one bird of that age (`bird_area_m2_by_age` in `farm.yaml`).
3. Follow blobs across clips by distance. Two blobs less than 20 cm apart are the same candidate.
4. A candidate that stays still for **120 minutes**, **and** during which the flock was active at least once, raises a **critical** alert. The alert carries an image with the candidate's location marked.
5. The farmer or vet records in the panel whether the bird was dead or alive. These labels are later used to evaluate and train the stage 2 model.

In addition, daily mortality is entered by hand (`daily_mortality_pct`). The default thresholds are 0.2% and 0.5% per day and **must be set with the vet.** The project document stresses that in HPAI and NDV a sudden rise in deaths is often the first sign.

**Limitations:**
- **A sleeping bird is also still.** The "flock was active" condition and the 120-minute duration reduce this false alarm, but heavy birds in the last weeks may still trigger it. The thresholds can be tuned with the "alive" labels.
- **A dead bird lying against resting birds** forms one large blob with them and fails the size filter. If another bird walks over it, it can also be missed for one clip. Detecting individual birds in stage 2 solves both.

## Signal 4: circadian pattern

**Method:** at each slot, the last 24 hours of `activity_index` are turned into hourly means. This profile is compared with the usual profile, the median of hourly means from 2 to 7 days earlier.

| Metric | Definition |
|---|---|
| `rhythm_corr` | Pearson correlation of the shapes of the two profiles. 1 means the same pattern |
| `rhythm_amplitude` | (90th − 10th percentile) today ÷ the same in the usual profile. Below 1 means the rhythm has flattened |

At least 18 hours of data in the last 24 hours are needed.

**Reference:** Roberts, Cain and Dawkins (2012): welfare problems predicted 1 to 2 days early from changes in optical-flow patterns.

**Limitation:** any change to the lighting programme (dark hours) changes the rhythm. Record these changes and tell the vet.

---

## Detection mode: the same signals per bird

When a per-bird detector is configured (`analysis.segmenter.kind: yolo`,
`mvbrotrack` or `external`), every sampled frame yields bird boxes. Box centres
are projected to floor metres and linked into tracks with the Hungarian
algorithm on distance (`tracking.py`). The colour-based metrics above are still
computed (from the boxes), and these per-bird metrics are added
(`signals/tracks.py`):

| Signal | Metric | Definition | Source of the method |
|---|---|---|---|
| 1 | `inactive_frac` | share of 10-s track segments with net speed below 2 cm/s | proportion of inactivity, Hernandez et al., Poultry Science 2026 |
| 1 | `active_speed` | mean net speed of the active segments (m/s) | mean speed of active segments, same paper |
| 2 | `feeder_time_frac` | mean share of time each bird (track >= 20 s) spends in a feeder zone | time near the food source, Scheidwasser et al., R. Soc. Open Sci. 2025 (41% lower in E. coli-infected birds) |
| 2 | `resource_count_frac`, `periphery_count_frac` | share of detected birds at feed + water, and at walls + corners | zone counts, Guo et al., Sensors 2020 |
| 3 | still / dead points | a track that covers 90% of the clip without leaving an 8 cm radius, or a track mostly classified `dead` | stationary persistence, Hao et al., Agriculture 2025 |
| 3 | `bird_count` | mean number of birds detected per frame | |
| 4 | `rhythm_corr`, `rhythm_amplitude` | computed from `1 - inactive_frac` instead of the optical-flow index | |

Rules for these metrics are in `configs/rules.yaml` (`inactivity_rise`,
`active_speed_drop`, `feeder_time_drop`, `periphery_count_rise`). The panel
shows the per-bird metric of a signal as soon as it has data.

A dead-class detection in `mortality.dead_class_min_clips` (default 2)
consecutive clips at the same spot raises a mortality alert without waiting
for the stillness timer.

Track identities are kept only within a clip and per camera. With overlapping
cameras, counts are summed across cameras and can double-count birds in the
overlap; the MVBroTrack ground-plane fusion is the way to remove that.

## Lighting programme

`lighting` in `farm.yaml` lists the programme with the date it started.
Dark hours are left out of the rhythm comparison, and the rhythm is not
judged for 2 days after a change, so a new programme does not look like disease.

## Measuring accuracy on this farm

Published numbers come from other houses, cameras and flocks. Accuracy on
this farm is measured against the vet log:

```bash
pews validate --pen pen1 --since 2025-11-17 --until 2025-12-31 --out report.md
```

An alert counts as correct when a vet-recorded problem (clinical sign,
diagnosis or deaths) follows it within 72 h or preceded it by at most 24 h.
The report gives recall (problems that were preceded by an alert), precision,
false alerts per week, median warning time and the share of mortality
candidates that were really dead. The panel shows the same summary under
Vet log.
