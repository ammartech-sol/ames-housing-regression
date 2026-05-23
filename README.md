# Ames Housing Price Prediction

**Goal:** Predict the sale price of residential properties in Ames, Iowa using physical attributes, location, quality ratings, and sale conditions.

**Why it matters:** Accurate price prediction benefits buyers, sellers, and real estate platforms alike. A reliable model removes guesswork from property valuation and surfaces which features actually drive residential value.

&nbsp;

| Step | Details |
|---|---|
| Dataset | Ames Housing by Dean De Cock, 2,930 rows, 82 features |
| Model | Ridge Regression |
| Target transformation | log1p applied to SalePrice |
| Best alpha | 10 |
| Final R2 | 0.9315 |
| Final MAE | $11,004 |
| Final RMSE | $15,727 |

&nbsp;

## Dataset

- **Source:** Ames Housing Dataset by Dean De Cock
- **Size:** 2,930 rows, 82 features
- **Target:** SalePrice (continuous, in USD)
- **Sale years:** 2006 to 2010

## Project Structure

```
ames-housing-regression/
├── data/
│   └── ames_housing.csv
├── models/
│   ├── ridge_ames.pkl
│   ├── preprocessor.pkl
│   └── ames_columns.pkl
├── notebook/
│   └── notebook.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Pipeline

**1. Data Cleaning**

Removed 6 columns exceeding 1,400 missing values: Alley, Fireplace Qu, Pool QC, Fence, Misc Feature, and Mas Vnr Type. Dropped rows with only 1 or 2 missing values in low-null columns. Filled remaining numerical nulls with median and categorical nulls with "None" to represent absent features rather than unknown values.

**2. Outlier Handling**

Removed 4 extreme Gr Liv Area observations above 4,000 sqft as recommended by the dataset creator. Capped 8 numerical columns at the 99th percentile including Lot Area, Total Bsmt SF, and 1st Flr SF. Year-based and count columns were left untouched since their extreme values represent valid rare cases. Capping was prioritized here because Ridge as a linear model is sensitive to extreme values, unlike tree-based models.

**3. Feature Engineering**

Four composite features were added before the train/test split. Total SF combines basement and above-ground floor area into one size measure. Total Bathrooms weights full and half baths into a single amenity score. House Age and Remod Age measure how old the house was and how recently it was renovated at the time of sale.

**4. Encoding**

The 39 object columns were split into two groups. 16 ordinal columns with a meaningful quality or grade order were encoded using OrdinalEncoder with explicit category sequences, applied to the full dataframe before splitting. 22 nominal columns with no inherent order were one-hot encoded using OneHotEncoder inside a ColumnTransformer applied after the split to prevent data leakage. StandardScaler was applied to all numerical columns inside the same ColumnTransformer. Final feature matrix contained 167 columns after encoding.

**5. Model Selection**

Compared Ridge Regression and LightGBM as baselines. Ridge achieved a test R2 of 0.9250 while LightGBM reached 0.9183. LightGBM showed a 6.4 point gap between its train score (0.9826) and test score, indicating significant overfitting without tuning. Ridge was selected as the primary model due to better generalization out of the box.

**6. Cross Validation**

5-fold cross validation on the training set returned a mean R2 of 0.9158 with a standard deviation of 0.0131. Scores across all five folds stayed above 0.90, confirming the model is stable and consistent across different data splits.

**7. Hyperparameter Tuning**

GridSearchCV tested 8 values of alpha across 5 folds for a total of 40 fits. Best alpha was 10, improving CV R2 to 0.9193 and final test R2 to 0.9315. MAE improved from $11,296 to $11,004 and RMSE dropped from $16,011 to $15,727.

**8. Feature Importance**

Ridge coefficients were taken as absolute values to measure feature importance. Top drivers confirmed that location (MS Zoning, Neighborhood), material quality (Exterior BrkFace), and size (Overall Qual, Gr Liv Area) are the primary determinants of residential property value in this dataset.

## Why Ridge Over LightGBM

LightGBM is generally the stronger model on tabular data but overfitted significantly here without tuning. Ridge with its built-in L2 regularization handled the 167-feature post-encoding space better out of the box. Alpha=10 applied enough regularization to prevent over-reliance on noisy or redundant encoded columns without underfitting. For a dataset of this size, a well-regularized linear model is a strong and interpretable choice.

## Feature Importance (Top 5)

| Feature | Insight |
|---|---|
| MS Zoning C (all) | Commercial zoning strongly reduces residential value |
| Exterior 1st BrkFace | Brick face exterior signals premium construction quality |
| Overall Qual | Overall material and finish quality is a core price driver |
| Gr Liv Area | Above ground living area directly scales with price |
| Neighborhood MeadowV | Location effects are well captured through neighborhood encoding |

## Model Performance

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Ridge Baseline | $11,296 | $16,011 | 0.9250 |
| LightGBM Baseline | $11,917 | $16,899 | 0.9184 |
| Ridge Tuned (alpha=10) | $11,004 | $15,727 | 0.9315 |

R2 of 0.9315 means the model explains 93.15% of the variance in house prices. On average, predictions are within $11,004 of the actual sale price.

## Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- LightGBM
- seaborn, matplotlib
- joblib

## How to Run

```bash
git clone https://github.com/ammartech-sol/ames-housing-regression
cd ames-housing-regression
pip install -r requirements.txt
jupyter notebook notebook/notebook.ipynb
```

The dataset is already included in the data folder.
