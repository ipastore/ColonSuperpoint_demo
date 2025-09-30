# SuperPoint Demo Reporting Cheat Sheet

This document explains the metrics and figures produced when you run:

```bash
./demo_superpoint.py <input> --config configs/config.yaml --no_display --report [--report_dir <label>]
```

Each run creates a timestamped folder inside `./reports` containing CSV summaries, PNG plots, and a `report_config.yaml` snapshot of the run settings.

## Key Concepts

- **Keypoint** – A salient pixel detected by SuperPoint. Each keypoint has a confidence score (`0–1`). Lowering `--conf_thresh` admits more low-confidence detections, so the total count of points rises and the confidence histogram shifts left.

- **Match** – A pairing between descriptors of two consecutive frames. SuperPoint descriptors are unit-normalised; we match using L2 distance.

- **Track** – The sequence of successful matches for the same feature across frames. We maintain up to `max_length` observations per track (90 by default). Older entries drop off once the window is full.

- **Match Score** – The average L2 distance of all matches inside a track. Because descriptors have unit norm, lower distances mean better matches.

## Files & How to Read Them

- `metrics.csv` – Per-frame stats: keypoint counts/confidence summary, track counts & lengths, track score stats, untracked ratios, processing times, and counts of tracks of exact length `L`.

- `summary.csv` – Aggregate stats (mean, std, min, max, median) for key metrics plus the correlation between tracked keypoint confidence and match score.

- `track_length_distribution.csv` & `track_scores.csv` – Totals for each final track length and the average score per track ID.

- `keypoints_per_frame.png` – Line chart of keypoints per frame.

- `keypoint_confidence_hist.png` – Fractional histogram with a horizontal box plot; `n=<count>` tells how many keypoints were included.

- `track_length_hist.png` / `track_score_hist.png` – Normalised histograms with box plots for track lengths and match scores.

- `untracked_ratio_per_frame.png` – Share of keypoints each frame that never joined a track.

- `report_config.yaml` – Snapshot of thresholds, model weights, run label, frame counts, etc.

## Practical Effects of Tweaks

- Lower `conf_thresh` ⇒ more low-confidence keypoints, higher untracked ratios, shorter tracks, worse match scores. Raise it to focus on high-confidence features.

- Lower `nms_dist` ⇒ more densely clustered keypoints (potentially redundant). Raise it for sparser, more distinct detections.

- Lower `nn_thresh` ⇒ only very similar descriptors match, so you get fewer but stronger tracks. Raise it to accept noisier matches (watch the score histogram grow toward higher distances).

## Link to Visualisation

When you run without `--no_display`, track colours in the UI reflect match quality: greener lines mean lower L2 distances (better matches); cooler colours or redder tones indicate weaker matches. Use the report to confirm that colour intuition across longer runs.

Happy analysing! Adjust thresholds, rerun the command, and compare the normalised plots and CSVs side by side to understand how your settings impact tracking quality.

