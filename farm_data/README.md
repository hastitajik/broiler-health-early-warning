# farm_data/

- `samples/`: seven sample frames from cam1. Each file name is the date and time shown on that image.
  - November: day-old chicks
  - December: 3 to 5 week old broilers
- `raw/<camera>/<YYYY-MM-DD>/`: put the DVR exports here. The date folder is required because names like `05.45.00-05.50.00RO000.h264` contain only the time.
- Everything else in this folder is created by the system (`pews.sqlite`, `grids/`, `media/`) and is not tracked by git.

This folder is named `farm_data/` rather than `data/` so it does not clash with MVBroTrack's `DATA/` folder on case-insensitive file systems (Windows, macOS).
