# Song Popularity Prediction from Audio Features and Metadata

**F20DL Data Mining and Machine Learning, Group 5, Heriot-Watt University Dubai**

| Requirement | Members |
|---|---|
| R1 Topic and direction | All |
| R2 Data analysis | Sahil, Aryan|
| R3 Baseline models | Abhinav, Raees |
| R4 Neural networks | Suha |

---

## 1. Project topic

We predict a track's Spotify **popularity** from its audio features and its track, artist and album metadata, in two ways:

- **Regression:** predict the exact popularity score (0–100).
- **Classification:** predict a popularity class: **Low (0–40)**, **Medium (41–60)** or **High (61–100)**. This lets us use Naïve Bayes and compare it with Decision Trees and k-NN on the same problem.

We chose this topic because the dataset is large, openly licensed and mostly tabular, and popularity works as both a regression and a classification target. It also lets us test whether popularity depends more on how a song sounds or on who released it.

The core project uses tabular data only. Album cover features are an optional extra. We do not use lyrics, other text or generative models.

## 2. Datasets

### 2.1 SpotGenTrack (main dataset)

| | |
|---|---|
| Source | Mendeley Data, https://doi.org/10.17632/4m2x4zngny.1 |
| Licence | CC BY 4.0 |
| Collected from | Spotify Web API (audio features, metadata, popularity) and Genius (lyrics, not used) |
| Size | 101,939 tracks, 56,129 artists, 75,511 albums (101,572 tracks after cleaning) |
| Files | `spotify_tracks.csv` (32 columns), `spotify_artists.csv` (9), `spotify_albums.csv` (16) |
| Task | Regression on `popularity` (0–100), plus classification into 3 popularity classes |

**Features (21 candidates out of 57 columns):**

| Group | Features |
|---|---|
| Audio (12) | acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, key, mode, time_signature |
| Track (3) | duration_ms, disc_number, track_number |
| Artist (3) | artist_popularity, followers, genres |
| Album (3) | album_type, release_date, total_tracks |

Dropped: IDs, URLs, names, the index column and `lyrics` (text data is out of scope).

**Previous use in ML:** the dataset was released with Martín-Gutiérrez et al. (2020) [1], who used it to predict popularity with deep learning. We use their results as a point of comparison.

### 2.2 MSD-I (optional dataset)

| | |
|---|---|
| Source | Zenodo, https://doi.org/10.5281/zenodo.1240485 |
| Licence | CC BY 4.0 |
| Size | 30,713 tracks, 16,753 albums, 15 genre labels |
| Contents | MSD track and artist IDs, genre, album cover URL, pre-computed visual feature vectors (2048-dim) |
| Previous use | Multimodal genre classification, Oramas et al. (2018) [2] |

There is no shared ID, so we matched on artist name + song title (names added from MSD's `unique_tracks.txt`, both sides lowercased, bracketed text, "feat." and punctuation removed). Only **442 tracks** match (303 exact), so MSD-I is only used for an optional experiment. Code: `match_msdi_spotgentrack.py`, output: `msdi_spotgentrack_matches.csv`.

## 3. Data preparation (done)

Full details and reasons are in `data_preparation_log.txt`. Code: `prepare_data.py`.

- Joined tracks, artists (first listed artist) and albums: 101,939 tracks, no missing values.
- Removed 367 tracks with tempo or time signature = 0 (failed audio analysis). Kept real but unusual values.
- log(1 + x) on followers (skew 9.5 → −0.20) and duration (skew 9.8 → 0.45).
- release_date → release_year.
- 2,371 genre labels grouped into 11 broad genres + "other" + "unknown".
- One-hot encoded key, time_signature, album_type and broad genre. mode kept as 0/1.
- **Split:** 70/15/15, stratified by popularity class, seed 42, done at **song level** so versions of the same song (e.g. single and album) stay in the same split. Train 71,065, validation 15,237, test 15,270.
- Features are **not** scaled in the file. Scalers are fitted on the training data only, inside each model notebook.

## 4. Exploratory analysis so far (train set)

Notebook: `02_r2_eda_sahil.ipynb`. Table: `r2_summary_stats.csv`. Figures: `fig1_popularity_hist.png`, `fig2_correlation.png`.

- Popularity: mean 39.9, median 41, SD 16.7, roughly symmetric (skew −0.14).
- Classes: Low 49.9%, Medium 39.4%, High 10.7%. High is the minority class.
- No missing values in any feature.
- Strongest links to popularity: artist_popularity (r = 0.70) and log_followers (0.56). Best audio feature: speechiness (−0.31), then loudness (0.24). The rest are weak (|r| < 0.15).
- Redundant pairs: artist_popularity and log_followers (0.88), energy and loudness (0.79), acousticness and energy (−0.72).
- Still skewed: track_number, total_tracks and disc_number (large compilations).

**Next:** scatter plots of the top features vs popularity, followers before/after log, and feature selection.

## 5. Approach (MoSCoW)

| Priority | Task |
|---|---|
| **Must (MVP)** | **Classical models (R3):** Decision Tree, k-NN, Naïve Bayes, linear models (linear regression with L2 regularisation for regression, logistic regression for classification) and Random Forest. Compare with baselines that predict the training mean (regression) or the majority class (classification). **Neural network (R4):** an MLP (Keras, as in tutorial P6.1), compared with the best classical model. |
| **Should** | Feature selection: a filter method (correlation ranking) and a wrapper method (forward selection). Compare audio only, metadata only and both. Train with and without the artist features. Optional k-means clustering to find groups of similar tracks. |
| **Could** | Add MSD-I cover features for the 442 matched tracks. |
| **Won't** | Lyrics or other text data, generative models. |

**Evaluation**
- Hyperparameters (e.g. tree depth, k, MLP layers, learning rate) tuned with **stratified 10-fold cross-validation** on the training set (GridSearchCV), then checked on validation.
- Regression: MSE, RMSE, MAE, R².
- Classification: accuracy, precision, recall and F-measure per class, confusion matrix.
- Every model is tested once on the same test tracks.

## 6. Work plan

| Task | Owner | Deadline |
|---|---|---|
| Data preparation + shared split | Sahil | ✅ 30 Oct |
| Summary stats, histogram, correlation matrix | Sahil | ✅ 5 Nov |
| Scatter plots, log plot, feature selection | Sahil | 6 Nov |
| Shared evaluation code + baselines | Raees | 29 Oct |
| Classical models tuned (DT, k-NN, NB, linear, RF) | Abhinav, Raees | 6 Nov |
| MLP built and tuned | Suha | 11 Nov |
| All report sections drafted | All | 13 Nov |
| Full report draft | All | 18 Nov |
| Final checks (figures, references, repo runs) | All | 20 Nov |
| Submit | All | 23 Nov |

Final deadline: **25 November 2026, 15:30**.

### Pitch feedback
We did not get specific feedback at the pitch. After the marking rubric and later lectures were released, we made these changes

---

## Repository structure

```
README.md
prepare_data.py                 # join, clean, encode and split the data
spotgentrack_clean.csv          # cleaned data with the shared split column (use this)
split_ids.csv                   # track id -> train / val / test
data_preparation_log.txt        # every cleaning decision, with reasons
02_r2_eda_sahil.ipynb           # summary statistics and EDA plots
r2_summary_stats.csv            # summary statistics table (train set)
fig1_popularity_hist.png        # popularity distribution
fig2_correlation.png            # correlation matrix
match_msdi_spotgentrack.py      # MSD-I to SpotGenTrack matching
msdi_spotgentrack_matches.csv   # the 442 matched tracks
```

## How to run

1. Install Python 3.10+ and the packages: `pip install pandas numpy scikit-learn matplotlib notebook`
2. **Use the prepared data:** load `spotgentrack_clean.csv` and filter by the `split` column, e.g. `train = df[df["split"] == "train"]`. Fit any scaler on the train rows only. Do not re-split.
3. **Rebuild from scratch:** download the SpotGenTrack CSVs (see below) into the same folder and run `python prepare_data.py`.
4. **EDA:** open `02_r2_eda_sahil.ipynb` with Jupyter (`python -m notebook`) and run all cells.
5. **MSD-I matching:** put `MSD-I_dataset.tsv` in the same folder and `unique_tracks.txt` in a `msdi-data/` subfolder, then run `python match_msdi_spotgentrack.py`.

## Getting the data

1. SpotGenTrack: https://doi.org/10.17632/4m2x4zngny.1
2. MSD-I: `MSD-I_dataset.tsv` and `msdi-data.tar.gz` from https://doi.org/10.5281/zenodo.1240485
3. MSD names: `unique_tracks.txt` from the Million Song Dataset site (Getting the dataset → Additional files)

The raw data files are too large to store in this repository.

## AI use

Claude (Anthropic) was used to help with: pitch slide design and text editing, drafting and editing this README. All decisions were reviewed by the group.Dataset choice, project direction and final analysis are the group's own.

## References

[1] Martín-Gutiérrez, D., Hernández-Peñaloza, G., Belmonte-Hernández, A. and Álvarez García, F. (2020) 'A multimodal end-to-end deep learning architecture for music popularity prediction', *IEEE Access*, 8, pp. 39361–39374. https://doi.org/10.1109/ACCESS.2020.2976033

[2] Oramas, S., Barbieri, F., Nieto, O. and Serra, X. (2018) 'Multimodal deep learning for music genre classification', *Transactions of the International Society for Music Information Retrieval*, 1(1), pp. 4–21. https://doi.org/10.5334/tismir.10

[3] Martín-Gutiérrez, D. et al. (2019) 'SpotGenTrack Popularity Dataset', Mendeley Data, V1. https://doi.org/10.17632/4m2x4zngny.1

[4] Oramas, S. (2018) 'MSD-I: Million Song Dataset with Images for Multimodal Genre Classification', Zenodo. https://doi.org/10.5281/zenodo.1240485
