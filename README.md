# Ames Housing Price Prediction

A regression project to predict residential property sale prices in Ames, Iowa. The dataset comes from Dean De Cock and covers 2,930 home sales between 2006 and 2010 across 82 features: everything from square footage and garage finish to neighborhood and zoning classification.

---

## Results

**Final Model: Ridge Regression (alpha = 10, log-transformed target)**

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Ridge Baseline | $11,296 | $16,011 | 0.9250 |
| LightGBM Baseline | $11,917 | $16,899 | 0.9184 |
| Ridge Tuned (alpha=10) | **$11,004** | **$15,727** | **0.9315** |

The model explains **93.15% of the variance in sale prices**. Predictions are off by $11,004 on average. LightGBM actually lost to Ridge here without tuning. More on that below.

---

## Dataset

- **Source:** Ames Housing Dataset by Dean De Cock
- **Size:** 2,930 rows, 82 features
- **Target:** `SalePrice` (continuous, in USD)
- **Sale years:** 2006 to 2010

---

## Folder Structure

```
ames-housing-regression/
├── data/
│   └── ames_housing.csv
├── models/
│   ├── ridge_ames.pkl
│   ├── ames_preprocessor.pkl
│   └── ames_columns.pkl
├── notebook/
│   └── notebook.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Project Walkthrough

### 1. Data Cleaning

Six columns were dropped immediately for exceeding 1,400 missing values: `Alley`, `Fireplace Qu`, `Pool QC`, `Fence`, `Misc Feature`, and `Mas Vnr Type`. For columns with only 1 or 2 missing rows (`Electrical`, the basement square footage columns, `Garage Cars`, `Garage Area`), those rows were dropped entirely rather than imputed since losing a handful of rows from a 2,930-row dataset is not a real cost.

The remaining nulls split into two groups. Numerical columns (`Lot Frontage`, `Mas Vnr Area`, `Garage Yr Blt`) were filled with their medians. Categorical columns like `Bsmt Qual` and `Garage Type` were filled with `"None"`. These are not unknown values, they mean the house simply does not have that feature.

### 2. Outlier Handling

The dataset creator specifically flagged a few `Gr Liv Area` observations above 4,000 sqft as outliers that should be removed before modeling. Those four rows were dropped first.

After that, all numerical columns were checked using the IQR method and boxplotted to see what was actually there. Eight columns had extreme values that were capped at the 99th percentile: `Lot Area`, `BsmtFin SF 1`, `Total Bsmt SF`, `1st Flr SF`, `Low Qual Fin SF`, `Gr Liv Area`, `Misc Val`, and `SalePrice`. Year-based columns and count columns were left alone since a house built in 1880 or a property with 8 bedrooms is unusual but not wrong.

Capping was used instead of removal because Ridge is a linear model and is sensitive to extreme values pulling coefficients in ways tree models are not.

### 3. Feature Engineering

Four composite features were created before the train/test split:

- **`Total SF`**: basement area plus first and second floor area combined into one total size measure
- **`Total Bathrooms`**: full baths plus half baths weighted at 0.5, across both above-grade and basement
- **`House Age`**: year sold minus year built, measuring how old the property was at time of sale
- **`Remod Age`**: year sold minus year last remodeled, capturing how recently the home was updated

### 4. Encoding

The 39 object columns were split into two groups based on whether the categories have a meaningful order.

16 ordinal columns (quality ratings, condition grades, functional ratings) were encoded using `OrdinalEncoder` with explicit category sequences before the train/test split. The category order was defined manually for each column. For example, `Exter Qual` goes `['Po', 'Fa', 'TA', 'Gd', 'Ex']` and `Bsmt Exposure` goes `['None', 'No', 'Mn', 'Av', 'Gd']`. Using `OrdinalEncoder` without defining these sequences would assign arbitrary numbers and lose the ordering information entirely.

22 nominal columns with no inherent order (`Neighborhood`, `MS Zoning`, `Sale Type`, etc.) were one-hot encoded using `OneHotEncoder` inside a `ColumnTransformer` applied after the split. `StandardScaler` was also applied to all numerical columns inside the same `ColumnTransformer`. The final feature matrix came out at 167 columns.

The ordinal encoding was applied to the full dataframe before splitting, which is technically a mild form of leakage for those 16 columns. Since ordinal encoding just replaces string labels with their predefined rank and does not learn any statistics from the data, the practical impact is negligible. There is nothing learned from the test set that could inflate scores.

### 5. Target Transformation

`SalePrice` was log-transformed using `np.log1p` before training. Skewness dropped from 1.76 to 0.08. Predictions were converted back using `np.expm1` before computing MAE and RMSE so the error metrics are in actual dollar values.

### 6. Model Selection

Ridge and LightGBM were both trained as baselines. Train scores:

```
Ridge train score:    0.93
LightGBM train score: 0.98
```

LightGBM's train score of 0.98 versus its test score of 0.9184 is a 6.4-point gap. Clear overfitting with default parameters. Ridge generalized better out of the box at 0.9250 on the test set. Ridge moved forward as the primary model.

### 7. Cross Validation

5-fold cross-validation on the training set returned a mean R2 of **0.9158** with a standard deviation of **0.0131**. All five folds stayed above 0.90.

### 8. Hyperparameter Tuning

`GridSearchCV` tested 8 alpha values (`[0.1, 1, 10, 50, 100, 200, 500, 1000]`) across 5 folds, 40 fits total. Best alpha was **10**.

| | Before Tuning | After Tuning |
|---|---|---|
| CV R2 | 0.9158 | 0.9193 |
| Test R2 | 0.9250 | 0.9315 |
| MAE | $11,296 | $11,004 |
| RMSE | $16,011 | $15,727 |

Alpha=10 applies enough regularization to shrink noisy and redundant coefficients without underfitting. Going higher than 10 starts compressing coefficients that actually carry signal.

---

## Why Ridge Beat LightGBM Here

LightGBM is usually the stronger model on tabular data. It lost here without tuning because the post-encoding feature space is 167 columns wide, many of which are one-hot encoded dummies that are sparse and correlated. LightGBM's default tree depth and learning rate were not set up for this kind of space. Ridge with L2 regularization handles high-dimensional spaces naturally. It penalizes large coefficients and deals well with collinear features.

This is not a statement that Ridge is better than LightGBM in general. Tuned LightGBM would likely come close or beat Ridge here. But for this dataset and this setup, Ridge is the cleaner and more interpretable choice.

---

## Feature Importance (Top 5)

Ridge does not have built-in feature importance, so absolute coefficient values were used as a proxy after scaling.

| Feature | Insight |
|---|---|
| `MS Zoning_C (all)` | Commercial zoning sharply reduces residential value |
| `Exterior 1st_BrkFace` | Brick face exterior signals premium construction |
| `Overall Qual` | Overall material and finish quality is the single strongest price driver |
| `Gr Liv Area` | Above-ground living area scales directly with price |
| `Neighborhood_MeadowV` | Location effects are strong and well captured through neighborhood encoding |

---

## Saved Files

| File | Description |
|---|---|
| `models/ridge_ames.pkl` | Trained Ridge model (alpha=10) |
| `models/ames_preprocessor.pkl` | Fitted ColumnTransformer (OHE + StandardScaler) |
| `models/ames_columns.pkl` | Ordered feature columns expected by the model |
| `notebook/notebook.ipynb` | Full notebook with code and outputs |

---

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
joblib
```

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn joblib
```

---

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/ammartech-sol/ames-housing-regression
cd ames-housing-regression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The dataset is already in the `data/` folder. Open and run the notebook from top to bottom:
```bash
jupyter notebook notebook/notebook.ipynb
```

---

## Key Takeaways

- R2 is computed on log-scale predictions but MAE and RMSE should always be reported in actual dollar values after reversing the transformation. Mixing the two makes error metrics uninterpretable
- Ordinal encoding without defining explicit category order is worse than no encoding at all. The model would learn that `Ex` and `Po` are just arbitrary numbers with no relationship to each other
- LightGBM overfitting on the baseline is not a reason to rule it out. It is a reason to tune it before drawing conclusions
- Ridge handles high-dimensional sparse feature spaces well because L2 regularization naturally shrinks correlated and redundant coefficients
- Capping outliers at the 99th percentile is a practical choice for linear models; for tree-based models it would not matter and could safely be skipped
- The dataset creator's own documentation flagged the large `Gr Liv Area` outliers. Reading the dataset documentation before touching the data is worth the 10 minutes
