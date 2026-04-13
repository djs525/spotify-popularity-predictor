# Spotify Track Popularity Prediction

Predicting Spotify track popularity (0–100) from audio features using LightGBM, Random Forest, and Linear Regression. Includes SHAP explainability analysis and genre mean-target encoding.

## Project structure

```
spotify-popularity-predictor/
├── data
│   ├── raw
│   │   └── dataset.csv
│   ├── feature_names.pkl
│   ├── genre_means.csv
│   ├── scaler.pkl
│   ├── X_test.npy
│   ├── X_train.npy
│   ├── y_test.npy
│   └── y_train.npy
├── figures
│   ├── 01_popularity_distribution.png
│   ├── 02_audio_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_genre_popularity.png
│   ├── 05_pca_projection.png
│   ├── 06_model_comparison.png
│   ├── 07_pred_vs_actual_and_residuals.png
│   ├── 09_lgb_feature_importance.png
│   ├── 10_ablation_genre.png
│   ├── 11_shap_beeswarm.png
│   ├── 12_shap_bar.png
│   ├── 13_shap_dep_duration_min.png
│   ├── 13_shap_dep_energy.png
│   ├── 13_shap_dep_genre_encoded.png
│   └── 14_shap_waterfall.png
├── models
│   ├── lgb_best_params.json
│   ├── lightgbm.pkl
│   ├── linear_regression.pkl
│   └── random_forest.pkl
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_modeling.ipynb
├── 04_shap.ipynb
├── README.md
├── requirements.txt
```

## Setup

```bash
git clone https://github.com/yourusername/spotify-popularity-prediction.git
cd spotify-popularity-prediction
pip install -r requirements.txt
```

## Data

Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle and place `dataset.csv` in `data/raw/`.

## Running the project

Run the notebooks in order:

```bash
jupyter notebook 01_eda.ipynb
jupyter notebook 02_preprocessing.ipynb
jupyter notebook 03_modeling.ipynb
jupyter notebook 04_shap.ipynb
```
OR just hit Run All in any IDE of your choice.

Each notebook saves its outputs (processed data, models, figures) so the next notebook can load them directly.

## Results summary

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 16.97 | 12.11 | 0.32 |
| Random Forest | 15.47 | 10.78 | 0.44 |
| LightGBM | 15.37 | 10.69 | 0.44 |
