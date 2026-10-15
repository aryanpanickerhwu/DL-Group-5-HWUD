# Song Popularity Prediction from Audio Features and Metadata

**F20/21DL Data Mining and Machine Learning, Group 5, Heriot-Watt University Dubai**

| Member | Main responsibility |
|---|---|
| Aryan | Pitch, project scope |
| Sahil | Dataset validation, MSD-I matching |
| Suha | Cleaning, preprocessing, EDA, clustering |
| Raees | Baseline models |
| Abhinav | Model tuning, MLP, optional cover features |

---

## Initial Project Proposal

> Revised after the Week 4 pitch. Feedback received and the changes we made are listed in [Pitch feedback](#pitch-feedback).

### 1. Project topic

We predict a track's Spotify **popularity** from its audio features and its track, artist and album metadata, in two ways:

- **Regression:** predict the exact popularity score (0–100).
- **Classification:** predict a popularity class: **Low (0–40)**, **Medium (41–60)** or **High (61–100)**. This lets us use classifiers such as Naïve Bayes and compare them with Decision Trees and k-NN on the same problem.

We chose this topic because it is a clear regression problem on a large, open, mostly tabular dataset. It also lets us test which kind of information explains popularity best: how a song sounds, or who released it and how.

The core project uses tabular data only. Album cover features are an optional extension (see the MoSCoW plan below). We do not use lyrics, other text, or generative models.

### 2. Datasets

#### 2.1 SpotGenTrack (main dataset)

| | |
|---|---|
| Source | Mendeley Data, https://doi.org/10.17632/4m2x4zngny.1 |
| Licence | CC BY 4.0 |
| Collected from | Spotify Web API (audio features, metadata, popularity) and Genius (lyrics, not used) |
| Size | 101,939 tracks, 56,129 artists, 75,511 albums |
| Files | `spotify_tracks.csv` (32 columns), `spotify_artists.csv` (9), `spotify_albums.csv` (16) |
| Task | Regression on `popularity` (0–100), plus classification into 3 popularity classes derived from it |

**Features (21 candidates out of 57 columns):**

| Group | Features | Type |
|---|---|---|
| Audio (12) | acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, key, mode, time_signature | Numeric (key, mode and time_signature are categorical and will be one-hot encoded) |
| Track (3) | duration_ms, disc_number, track_number | Numeric |
| Artist (3) | artist_popularity, followers, genres | Numeric, numeric (log-transformed), list of categories |
| Album (3) | album_type, release_date, total_tracks | Categorical, date (converted to year), numeric |

Dropped: IDs, URLs, names, the index column, and `lyrics` (text data is out of scope).

**Previous use in ML:** the dataset was released with Martín-Gutiérrez et al. (2020) [1], who used it for deep-learning popularity prediction. We use their work as a reference point.

#### 2.2 MSD-I (optional dataset)

| | |
|---|---|
| Source | Zenodo, https://doi.org/10.5281/zenodo.1240485 |
| Licence | CC BY 4.0 |
| Size | 30,713 tracks, 16,753 albums, 15 genre labels |
| Contents | MSD track and artist IDs, genre, album cover URL, and pre-computed visual feature vectors (2048-dim) |
| Previous use | Multimodal genre classification, Oramas et al. (2018) [2] |

**Matching to SpotGenTrack.** There is no shared ID, so we matched on artist name + song title. MSD-I only stores IDs, so names were added from MSD's `unique_tracks.txt`. Both sides were lowercased, and bracketed text ("(Remastered)"), "feat." and punctuation were removed.

- Exact matches: 303
- After cleaning: **442 tracks** (1.4% of MSD-I, 0.4% of SpotGenTrack)
- Artists in both datasets: 3,136, covering 14,448 SpotGenTrack tracks

Because so few tracks match, MSD-I is only used for an optional experiment. Code: `match_msdi_spotgentrack.py`.

### 3. Exploratory analysis so far

| Statistic | Popularity |
|---|---|
| Mean | 39.78 |
| Median | 41 |
| Standard deviation | 16.79 |
| Q1 / Q3 | 29 / 52 |
| Range | 0–97 |

| Popularity band | Tracks | % |
|---|---|---|
| 0–20 | 13,929 | 13.7% |
| 21–40 | 36,998 | 36.3% |
| 41–60 | 40,102 | 39.3% |
| 61–80 | 10,552 | 10.4% |
| 81–100 | 358 | 0.35% |

**Classification target:** Low (0–40) = 50,927 tracks (50.0%), Medium (41–60) = 40,102 (39.3%), High (61–100) = 10,910 (10.7%). The High class is the minority, so we use a stratified split and report macro-F1 as well as accuracy.

75.6% of tracks score between 21 and 60, and only 358 tracks score above 80. Models will probably under-predict hits, so we will report error on high-popularity tracks separately and stratify the split by popularity band.

**Checks done:** popularity is numeric with no missing values. There are no duplicate track IDs, and 1 track has no artist name after the join.

**Still to check:** missing values in the other columns, outliers (duration, tempo), and the same song appearing under more than one ID.

### 4. Approach (MoSCoW)

| Priority | Task |
|---|---|
| **Must (MVP)** | **Classical models (R3):** Decision Tree, k-Nearest Neighbours, Naïve Bayes, a linear model (Ridge / Logistic Regression) and Random Forest, on audio + metadata. Regression models are compared with a baseline that always predicts the training mean, and classifiers with one that always predicts the majority class. **Neural network (R4):** an MLP on the same features, compared against the best classical model. |
| **Should** | Compare metadata only vs audio only vs combined features. Train with and without artist features (artist_popularity, followers). Gaussian Mixture Model clustering (EM) to find groups of similar tracks and compare their popularity. |
| **Could** | Add MSD-I album cover features for the 442 matched tracks and test whether they improve predictions. |
| **Won't** | Lyrics or other text data, generative models. |

**Evaluation:** 70% train / 15% validation / 15% test, stratified by popularity class. Scalers and encoders are fitted on the training set only. Hyperparameters (e.g. tree depth, k in k-NN, MLP layers and learning rate) are tuned with 5-fold cross-validation on the training set, and choices are reported with their validation results.

- Regression: MAE, RMSE, R²
- Classification: accuracy, macro-F1, confusion matrix

Every model is scored on the same test tracks.

### 5. How we meet the requirements

| Requirement (marks) | How we meet it | Weeks |
|---|---|---|
| **R1** Project topic and direction (4) | Clear objective (what predicts Spotify popularity?), compared with prior work [1] | W4–5 |
| **R2** Data analysis (6) | Source, licence and collection method of both datasets; feature descriptions; summary statistics; skew, imbalance and missing values handled; cleaning and matching steps justified; labelled visualisations | W5–7 |
| **R3** Baseline training and evaluation (18) | Decision Tree, k-NN, Naïve Bayes, linear model and Random Forest; tuned with cross-validation; stratified split; results tables, confusion matrices and feature importance; discussion of what DT, k-NN and NB each show about popularity | W7–8 |
| **R4** Neural networks (10) | MLP with hyperparameters tuned by experiment; comparison with the classical models and with [1]; optional cover features | W8–9 |
| **D3** Mini-viva (10) | Every member can explain the data, the models and the limitations | W10–11 |

### 6. Work plan

| Task | Lead | Weeks |
|---|---|---|
| Pitch and confirm scope | Aryan | W4 |
| Validate datasets and proposal | All | W4–5 |
| Check MSD-I matches | Sahil | W5–6 |
| Clean, join, preprocess | Suha | W5–6 |
| EDA, features, clustering | Suha | W6–7 |
| Train classical models (DT, k-NN, NB, linear, RF) | Raees | W7–8 |
| Tune models, test MLP | Abhinav | W8–9 |
| Cover features (optional) | Abhinav | W8–9 |
| Evaluate and write report | All | W9–10 |
| Final testing and fixes | All | W11 |

Final submission: **25 November 2026, 15:30**.

### Pitch feedback

No specific feedback was given at the pitch, so the scope stays as presented. After reading the published marking rubric we made these changes:

| Change | Reason |
|---|---|
| Added a 3-class classification version of the target | R3 expects Naïve Bayes, Decision Trees and k-NN, and Naïve Bayes needs classes |
| Added k-NN and Naïve Bayes to the classical models | R3 lists them and asks for their practical use on our topic |
| Hyperparameters tuned with cross-validation | R3 and R4 reward justified hyperparameters |
| Mapped the plan to R1–R4 | Rubric published after the pitch |

---

## Repository structure

```
README.md
match_msdi_spotgentrack.py      # MSD-I ↔ SpotGenTrack matching
data/                           # not committed (see "Getting the data")
notebooks/                      # EDA and modelling
```

## Getting the data

1. SpotGenTrack: download from https://doi.org/10.17632/4m2x4zngny.1
2. MSD-I: download `MSD-I_dataset.tsv` and `msdi-data.tar.gz` from https://doi.org/10.5281/zenodo.1240485
3. MSD names: `unique_tracks.txt` from the Million Song Dataset site (Getting the dataset → Additional files)

The data files are not stored in this repository because of their size.

## AI use

Slide design, formatting and some text editing for the pitch and this README were assisted by Claude (Anthropic). The project idea, dataset choice, analysis, statistics and work plan are the group's own.

## References

[1] Martín-Gutiérrez, D., Hernández-Peñaloza, G., Belmonte-Hernández, A. and Álvarez García, F. (2020) 'A multimodal end-to-end deep learning architecture for music popularity prediction', *IEEE Access*, 8, pp. 39361–39374. https://doi.org/10.1109/ACCESS.2020.2976033

[2] Oramas, S., Barbieri, F., Nieto, O. and Serra, X. (2018) 'Multimodal deep learning for music genre classification', *Transactions of the International Society for Music Information Retrieval*, 1(1), pp. 4–21. https://doi.org/10.5334/tismir.10

[3] Martín-Gutiérrez, D. et al. (2019) 'SpotGenTrack Popularity Dataset', Mendeley Data, V1. https://doi.org/10.17632/4m2x4zngny.1

[4] Oramas, S. (2018) 'MSD-I: Million Song Dataset with Images for Multimodal Genre Classification', Zenodo. https://doi.org/10.5281/zenodo.1240485
