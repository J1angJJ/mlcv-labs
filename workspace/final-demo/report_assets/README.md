# Report Assets

This directory contains lightweight, curated assets for the final report and local review.

## Browser Screenshots

`browser_screenshots/` contains screenshots captured from the local FastAPI frontend:

```text
01_success_complex_23_targets.png
  Complex scene with 23 targets, all correctly detected.

02_failure_case_001_drone_far_both_under_count.png
  Failure case 1: drone ultra-long-range scene; both models under-count, YOLO11n is closer.

03_failure_case_002_drone_far_under_to_over.png
  Failure case 2: drone ultra-long-range scene; YOLO26n under-counts, YOLO11n over-counts.

04_failure_case_003_near_overlap_both_under_count.png
  Failure case 3: close-range overlap; both models miss highly overlapping targets.

05_failure_case_004_single_closeup_yolo11n_over_count.png
  Failure case 4: single close-up target; YOLO11n over-counts.

06_case_005_near_overlap_yolo11n_success.png
  Case 5: close-range overlap; YOLO11n correctly detects all targets and fixes YOLO26n's miss.
```

The generated datasets, model weights, prediction outputs, and raw failure-case folders remain ignored under `data/`, `runs/`, `outputs/`, and `weights/`.
