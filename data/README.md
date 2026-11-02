# Data

**Dataset:** football-players-detection, version 12, from Roboflow Universe (workspace `roboflow-jvuqo`)
**Licence:** CC BY 4.0
**Link:** https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc

There are 372 frames at 1280×1280 from Bundesliga broadcast clips, labelled with four classes: ball, goalkeeper, player and referee.

| Split | Frames | Objects |
|---|---|---|
| train | 298 | 7,133 |
| valid | 49 | 1,174 |
| test | 25 | 598 |

## Setup
1. On the dataset page, download version 12 in **YOLOv5 PyTorch** format (or YOLOv8).
2. Unzip it into this folder so it contains `train/`, `valid/` and `test/` folders, each with `images/` and `labels/` subfolders.
3. Keep this folder's `data.yaml`. It uses relative paths, so the notebook runs as is.

The images are not included in this repository to keep it small.
