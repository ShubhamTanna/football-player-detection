# Facts: Football Player Detection

Every number quoted for this project, with the file it comes from.

| Fact | Value | Source |
|---|---|---|
| Headline | Test mAP@0.5 0.826 (all classes, 960 px) | `outputs/test_metrics.csv` |
| Player / referee / goalkeeper / ball mAP@0.5 | 0.992 / 0.954 / 0.931 / 0.426 | `outputs/test_metrics.csv` |
| Ball precision / recall | 0.946 / 0.375 | `outputs/test_metrics.csv` |
| v1 (Roboflow 3.0, course project) mAP@0.5 | 0.82 overall; player 0.98, referee 0.91, goalkeeper 0.91, ball 0.47 | notebook cell 16 (from the v1 report) |
| Test mAP@0.5 at 640 px vs 960 px | 0.787 vs 0.826 | notebook cell 15 |
| Ball mAP@0.5 at 640 px vs 960 px | 0.287 vs 0.432, on the validation set | notebook cell 12 |
| Inference sizes compared | 640, 960, 1280 px; 960 best on validation | notebook cell 12 |
| Model size | 3,133,436 parameters (YOLOv8-nano) | notebook cell 12 |
| Data | 372 labelled frames (298 / 49 / 25), 8,905 objects, 56 clips | notebook cells 4 and 23 |
| Class imbalance | Players outnumber balls 23 to 1; median ball 8 px wide | notebook cell 5 |
| Training | 43 epochs, 1.75 h on CPU | notebook cell 9 |
| Speed | 244 ms per frame at 960 px on a 2-core CPU, about 4 fps | notebook cell 19 |
| Possession computable | 47% of frames (ball detected in 47.3%) | notebook cell 25 |
| Context | v1: Cognitive Sciences and Analytics course, B.Tech CSBS Semester 7, 2025. v2 rebuilt solo in 2026 | README |

## Caveats

- v2 matches v1 overall (0.826 vs 0.82) and beats it on 3 of 4 classes.
- The ball improvement from 0.29 to 0.43 is a validation-set figure. On the test set the overall score rose from 0.79 to 0.83.
- Test frames share clips with training frames, so the scores are an upper bound for unseen matches.
