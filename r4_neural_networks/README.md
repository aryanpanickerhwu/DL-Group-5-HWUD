# R4 – Neural networks (MLP and CNN)

All models are in Keras. Protocol: shared song-level split, 10-fold StratifiedGroupKFold on train (groups = song_id), 6 random settings per feature set, early stopping on validation, test set used once per model.

## Test results
Regression MLP (baseline: training mean, MAE 13.41, RMSE 16.70, R² 0.00):

| feature set | columns | R² | RMSE | MAE |
|---|---|---|---|---|
| all | 49 | 0.668 | 9.62 | 7.46 |
| metadata (R3) | 23 | 0.658 | 9.76 | 7.59 |
| selected 3 | 5 | 0.567 | 10.99 | 8.61 |
| no artist, genre kept | 47 | 0.400 | 12.94 | 10.27 |
| no artist (R3) | 34 | 0.309 | 13.88 | 11.04 |
| audio (R3) | 26 | 0.234 | 14.61 | 11.68 |

MLP classifier (Low 0–40, Medium 41–60, High 61–100; baseline: majority class, macro-F1 0.222):

| feature set | macro-F1 | accuracy | High recall |
|---|---|---|---|
| all | 0.640 | 0.675 | 0.79 |
| metadata | 0.640 | 0.678 | 0.78 |
| no artist (R3) | 0.449 | 0.493 | 0.52 |
| audio | 0.408 | 0.448 | 0.48 |

Compared with the R3 models on the same test set (all features), the Random Forest is best and the MLP is second on regression and on macro-F1.

## Scripts (run from the repo root, where spotgentrack_clean.csv is)
- `mlp_keras.py <versions> [test]` – regression MLP. Versions: `with_artist`, `without_artist`, `selected_3`, `r3_no_artist`, `r3_audio`, `r3_metadata`. Example: `py r4_neural_networks\mlp_keras.py r3_audio r3_metadata`. Adding the word `test` runs the final test (once). `selected_3` needs `selected_features.txt` in the repo root.
- `mlp_classifier_keras.py <versions> [test]` – MLP classifier, same protocol, scored on macro-F1. Versions: `all`, `r3_no_artist`, `r3_audio`, `r3_metadata`.
- `compare_r3.py`, `compare_by_feature_set.py`, `compare_r3_test.py`, `make_report_figures.py`, `plot_mlp.py` – tables and figures; run from inside this folder. They need R3's result files (`results/`).
- `build_matched_features.py`, `download_covers.py` – match MSD-I to SpotGenTrack (442 matches, 394 used) and download the covers.
- `cover_features_keras.py <with_artist|without_artist>` – MLP with the MSD-I precomputed visual features.
- `cnn_covers_keras.py <with_artist|without_artist>` – MobileNetV2 fine-tuned on the raw covers (single run).

## Cover experiments
On the 394 matched tracks, covers gave no reliable improvement, with or without artist features. The CNN looked better on the team split (58 test tracks, shared artists) but not in the artist-grouped CV.

## Requirements
Python with tensorflow/keras, scikit-learn, pandas, numpy, pillow, matplotlib.