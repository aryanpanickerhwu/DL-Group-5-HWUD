# R4 – Neural networks (MLP and CNN)

## MLP results (Keras)
Test set, used once per version. Mean baseline: MAE 13.41, RMSE 16.70, R² 0.00.

| version | features | MAE | RMSE | R² |
|---|---|---|---|---|
| all features | 49 columns | 7.46 | 9.62 | 0.668 |
| selected 3 | artist_popularity, log_followers, album_type (5 columns) | 8.61 | 10.99 | 0.567 |
| without artist features | 47 columns | 10.27 | 12.94 | 0.400 |

"Artist features" = `artist_popularity` and `log_followers`.
Plot: `mlp_pred_vs_actual.png`.
Result files: `mlp_test_results.json` (all features and without artist), `mlp_test_results_selected_3.json`.

## Scripts (run from the repo root, where spotgentrack_clean.csv is)
1. `mlp_keras.py` – Keras MLP. 10-fold StratifiedGroupKFold on train (groups = song_id), 6 random settings per version, early stopping on validation. `RUN_VERSIONS` picks which version to run. Keep `RUN_TEST = False` for tuning and validation; set it to `True` once for the final test. The selected-3 version needs `selected_features.txt` from the repo root.
2. `plot_mlp.py` – predicted vs actual plot (run from inside this folder).
3. `build_matched_features.py` – links MSD-I tracks to SpotGenTrack ids (442 matches after cleaning; 394 kept, 48 dropped because they matched several versions of the same song). Needs the raw SpotGenTrack and MSD-I files.
4. `download_covers.py` – downloads the 394 cover images.
5. `cover_features_keras.py <with_artist|without_artist>` – MLP with and without MSD-I precomputed visual features (3 seeds).
6. `cnn_covers_keras.py <with_artist|without_artist>` – MobileNetV2 fine-tuned on the raw covers (single seed).

## Cover experiments: summary
On the 394 matched tracks, covers gave no reliable improvement, with or without artist features. The CNN looked slightly better on the team split (test n = 58, artist overlap between splits) but not in the artist-grouped 5-fold CV.

## Requirements
tensorflow/keras, scikit-learn, pandas, numpy, pillow, matplotlib.