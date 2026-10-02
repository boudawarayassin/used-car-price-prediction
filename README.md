# 🚗 Used Car Price Prediction

Machine Learning project for predicting the price of used vehicles from vehicle characteristics such as manufacturer, model, year, mileage, condition and location.

The project covers the complete Machine Learning workflow:

**Data Cleaning → Exploratory Data Analysis → Preprocessing → Feature Engineering → Model Comparison → Error Analysis → Model Persistence → FastAPI Deployment**

---

## 📌 Project Overview

The objective is to build a regression model capable of estimating the price of a used vehicle based on its available characteristics.

The project uses a large used-car dataset containing approximately **426,880 observations and 26 original columns**.

The final solution uses a **Random Forest Regressor** combined with:

* Missing-value imputation
* One-Hot Encoding for categorical variables
* A leakage-safe `model_frequency` feature
* FastAPI for deployment
* HTML/CSS/JavaScript for the user interface

---

## 🎯 Objectives

The main objectives are:

* Understand and clean a large real-world dataset.
* Identify missing values and problematic observations.
* Prevent data leakage during preprocessing.
* Compare several regression algorithms.
* Handle high-cardinality categorical variables.
* Perform feature engineering.
* Analyze prediction errors.
* Save the final model.
* Deploy the model through a web API and interface.

---

# 📊 Dataset

The original dataset contains **426,880 rows and 26 columns**.

Original features include:

```text
id
url
region
region_url
price
year
manufacturer
model
condition
cylinders
fuel
odometer
title_status
transmission
VIN
drive
size
type
paint_color
image_url
description
county
state
lat
long
posting_date
```

The target variable is:

```text
price
```

---

# 🧹 Data Cleaning

Several invalid or problematic observations were removed or corrected.

### Price

Only realistic positive prices were retained:

```python
df = df[
    (df["price"] > 0) &
    (df["price"] <= 1_000_000)
].copy()
```

### Year

Years below 1950 were considered invalid and replaced with missing values.

### Odometer

Values above 1,000,000 miles were replaced with missing values.

---

# 🗑️ Removed Columns

The following columns were removed for the first version of the project:

| Column         | Reason                                     |
| -------------- | ------------------------------------------ |
| `county`       | 100% missing                               |
| `size`         | Very high percentage of missing values     |
| `VIN`          | High-cardinality identifier                |
| `id`           | Identifier                                 |
| `url`          | Raw URL                                    |
| `region_url`   | Raw URL                                    |
| `image_url`    | Raw URL                                    |
| `description`  | Text feature postponed to a future version |
| `posting_date` | Temporal information not used in V1        |

The `posting_date` column was also examined before removal. The available dates were concentrated in 2021, mainly April and May.

---

# 🔀 Train/Test Split

The data was divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

Result:

```text
Training set: 315,145 observations
Testing set:   78,787 observations
```

The test set was kept separate until final evaluation.

---

# ⚠️ Data Leakage Prevention

Data leakage was an important consideration throughout the project.

Preprocessing operations such as:

* Median imputation
* One-Hot Encoding
* Feature construction

were implemented through training-based transformations and pipelines whenever applicable.

For example, numerical missing values were handled using:

```python
SimpleImputer(strategy="median")
```

The median is learned from the training data rather than from the complete dataset.

Categorical variables were processed using:

```python
SimpleImputer(
    strategy="constant",
    fill_value="unknown"
)
```

and:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

---

# 🔎 Exploratory Data Analysis

The numerical variables used in the baseline models included:

```text
year
odometer
lat
long
```

A correlation matrix was also analyzed.

Correlation was used only as an exploratory tool because it mainly measures linear relationships and does not properly represent relationships involving categorical variables.

---

# 📈 Price Distribution

The target variable is strongly right-skewed.

Important training-set quantiles:

| Percentile |      Price |
| ---------: | ---------: |
|        25% |     $7,000 |
|        50% |    $15,295 |
|        75% |    $27,590 |
|        90% |    $38,000 |
|        95% |    $44,999 |
|        99% |    $68,660 |
|      99.9% |   $119,900 |
|    Maximum | $1,000,000 |

There were relatively few extremely expensive vehicles.

Therefore, extreme prices were not arbitrarily removed from the dataset.

---

# 🤖 Models Compared

Several regression approaches were evaluated:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest
* HistGradientBoosting
* XGBoost
* CatBoost

---

# 📊 Model Comparison

The main comparison was:

| Model                   |    MAE |    RMSE |     R² |
| ----------------------- | -----: | ------: | -----: |
| Linear Regression       | $4,780 |  $9,713 | 0.6408 |
| Ridge                   | $4,894 |  $9,666 | 0.6443 |
| Lasso                   | $6,091 | $10,720 | 0.5625 |
| Random Forest baseline  | $3,802 |  $7,551 | 0.7829 |
| Random Forest optimized | $3,688 |  $8,199 | 0.7441 |
| HistGradientBoosting    | $4,073 |  $8,529 | 0.7231 |
| XGBoost                 | $4,352 |  $8,129 | 0.7484 |
| CatBoost                | $3,938 |  $8,110 | 0.7496 |

The baseline Random Forest performed particularly well on the held-out test set.

---

# 🧠 High-Cardinality Problem

The `model` variable contained a very large number of distinct categories.

In the training data:

```text
Distinct models: 25,164
```

Distribution:

```text
13,099 models → only 1 observation
19,975 models → ≤ 5 observations
21,817 models → ≤ 10 observations
968 models    → ≥ 50 observations
498 models    → ≥ 100 observations
```

This means that `model` is highly fragmented.

Using One-Hot Encoding alone creates a very large sparse feature space and does not provide much information for rare models.

---

# 🛠️ Feature Engineering: Model Frequency

Instead of using target encoding, a leakage-safe frequency feature was created.

The frequency was calculated **only from the training set**:

```python
model_frequency = X_train["model"].value_counts(
    dropna=False
)
```

Then:

```python
X_train_freq["model_frequency"] = (
    X_train_freq["model"]
    .map(model_frequency)
    .fillna(0)
)
```

For the test set:

```python
X_test_freq["model_frequency"] = (
    X_test_freq["model"]
    .map(model_frequency)
    .fillna(0)
)
```

An unseen model in the test set receives:

```text
0
```

This prevents the test set from influencing the feature.

---

# 🏆 Final Model

The final model is:

```text
RandomForestRegressor
n_estimators = 50
random_state = 42
n_jobs = -1
```

with:

* Numerical imputation
* One-Hot Encoding
* `model_frequency`

The final feature groups are:

### Numerical

```text
year
odometer
lat
long
model_frequency
```

### Categorical

```text
region
manufacturer
model
condition
cylinders
fuel
title_status
transmission
drive
type
paint_color
state
```

---

# 📊 Final Performance

Evaluation on the held-out test set:

| Metric                |        Result |
| --------------------- | ------------: |
| MAE                   | **$2,201.99** |
| RMSE                  | **$5,928.03** |
| R²                    |    **0.8662** |
| Median Absolute Error |   **$692.62** |

The model achieved an R² of **0.8662** on the held-out test set.

The median absolute error was substantially lower than the MAE, indicating that a relatively small number of very large errors have a significant effect on the mean absolute error.

---

# ⚠️ Target Encoding Experiment

Target encoding was also investigated for the high-cardinality `model` and `region` variables.

An initial experiment produced:

```text
MAE = $2,310.96
RMSE = $6,566.72
R² = 0.8358
```

However, this result was identified as **optimistic** because the encoder was initially fitted using all training targets before transforming the training data.

A subsequent leakage-safe validation experiment showed substantially weaker results.

Therefore, the target-encoding result was **not selected as the final model**.

The final approach uses the training-derived `model_frequency` feature instead.

---

# 🔍 Error Analysis

The final model was analyzed using absolute errors:

```python
absolute_error = abs(actual_price - predicted_price)
```

The final model obtained:

```text
Median Absolute Error: $692.62
Maximum Error:         $437,789.68
```

The largest errors were associated with rare and/or extreme vehicle listings.

Examples included very expensive vehicles and anomalous listings.

This shows that the model performs much better on the majority of normal-priced vehicles than on extremely rare high-value vehicles.

---

# 💰 Error by Price Range

The analysis showed that prediction difficulty increases considerably for expensive vehicles.

| Price Range |  Count |     MAE |
| ----------- | -----: | ------: |
| < $10k      | 28,880 |  $2,312 |
| $10k–$25k   | 27,271 |  $1,793 |
| $25k–$50k   | 20,126 |  $2,042 |
| $50k–$100k  |  2,391 |  $7,653 |
| > $100k     |    119 | $58,844 |

The highest-priced vehicles are much harder to predict because they are rare and can have characteristics that are poorly represented in the training data.

---

# 📉 Actual vs Predicted

An actual-vs-predicted scatter plot was created to evaluate how closely predictions follow the ideal diagonal relationship.

The majority of observations follow the expected trend, while extreme high-priced vehicles generate larger deviations.

---

# 💾 Model Persistence

The final model was saved using Joblib:

```python
joblib.dump(
    rf_freq_pipeline,
    "car_price_model_final.pkl"
)
```

The model was successfully reloaded and used for prediction.

The frequency dictionary was also saved:

```python
joblib.dump(
    model_frequency,
    "model_frequency.pkl"
)
```

Both files are required by the deployed application.

---

# 🌐 FastAPI Deployment

The final model is exposed through a FastAPI application.

The API provides:

### Home page

```text
GET /
```

### Health check

```text
GET /health
```

### Prediction

```text
POST /predict
```

The prediction endpoint receives vehicle information and returns the estimated price.

Example response:

```json
{
    "predicted_price": 35089.14
}
```

---

# 🖥️ Web Interface

The application includes a web interface built with:

* HTML
* CSS
* JavaScript

The user can enter:

* Region
* Manufacturer
* Model
* Year
* Odometer
* Condition
* Cylinders
* Fuel
* Title status
* Transmission
* Drive
* Vehicle type
* Paint color
* State
* Latitude
* Longitude

The JavaScript frontend sends the information to:

```text
POST /predict
```

and displays the predicted price.

---

# 📁 Project Structure

```text
car-price-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── car_price_model_final.pkl
│   └── model_frequency.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── app.js
│
├── notebooks/
│   └── car_price_prediction.ipynb
│
└── data/
    └── vehicles.csv
```

---

# ⚙️ Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start FastAPI with:

```bash
uvicorn app:app --reload --port 8000
```

Then open:

```text
http://127.0.0.1:
```
