# Analytics Pipeline

This module uses one Titanic dataset load for profiling, visual analysis, classification, imbalance handling, tuning, and fare regression. The implementation is split into two ordered scripts so the work is easy to follow.

## Setup and run

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r analytics/requirements.txt
python analytics/01_eda.py
python analytics/02_modeling.py
python analytics/predict_saved_model.py
```

`01_eda.py` contains the module's only `sns.load_dataset("titanic")` call. On the first run it immediately saves `analytics/titanic.csv`. Because that CSV is committed, later and offline runs use `pd.read_csv` instead. `02_modeling.py` always reads the same committed CSV and never downloads the dataset again.

## Profiling and missing values

The raw data has 891 rows and 15 columns. Four columns contain missing values:

| Column | Missing | Decision |
|---|---:|---|
| age | 19.87% | Median imputation because it is inside the 5%-30% range |
| embarked | 0.22% | Drop affected rows because it is below 5% |
| deck | 77.22% | Drop the column because filling such a large missing share would be unreliable |
| embark_town | 0.22% | Drop affected rows because it is below 5% |

The EDA copy has 889 rows after applying these rules. The complete `df.info()`, `df.describe()`, missing-value output, and decisions are recorded in [reports/eda_results.md](reports/eda_results.md).

## Univariate and bivariate findings

- The age IQR limits are 2.50 to 54.50, with 65 outliers.
- The fare IQR limits are -26.76 to 65.66, with 114 outliers.
- Fare has mean 32.097, median 14.454, and mode 8.050. Since mean > median > mode, the distribution is right-skewed; high fares pull the mean upward.

Survival rates calculated with boolean masks are:

| Group | Survival rate |
|---|---:|
| Female | 0.740 |
| Male | 0.189 |
| First class | 0.626 |
| Second class | 0.473 |
| Third class | 0.242 |

| Sex and class | Survival rate |
|---|---:|
| Female, first | 0.967 |
| Female, second | 0.921 |
| Female, third | 0.500 |
| Male, first | 0.369 |
| Male, second | 0.157 |
| Male, third | 0.135 |

The correlation matrix contains exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`. The strongest absolute pair is `pclass`/`fare` at -0.548, showing that lower-numbered classes paid higher fares. The second is `sibsp`/`parch` at 0.415, suggesting that passengers travelling with siblings or a spouse often also travelled with parents or children.

## Four-chart data story

1. **Survival by sex and class:** First-class female survival was 96.7%, compared with 13.5% for third-class males. Sex creates the largest separation, while passenger class adds another strong difference.
2. **Age by outcome:** Both survivors and non-survivors had a median age of 28.0 after EDA imputation. The box plots overlap greatly, so age alone is not a clear separator.
3. **Age, fare, and outcome:** Survivors paid a median fare of 26.00, versus 10.50 for non-survivors. Higher-fare observations include more survivors, although both outcomes appear across the age range.
4. **Family size and outcome:** Among family sizes with at least ten passengers, size four had the highest survival rate at 72.4%. Travelling alone or in a very large group was less favorable than travelling with a small family group.

The full set of chart files is in `charts/`. The EDA-only z-score check gave age and fare means of approximately 0 and standard deviations of 1. Those standardized columns are not used by the modeling script.

## Classification

The target contains 549 non-survivors (61.6%) and 342 survivors (38.4%). Stratification keeps that balance similar in train and test data. The split occurs before preprocessing; median imputation, most-frequent imputation, one-hot encoding, and scaling are fitted only on training data through a `ColumnTransformer` pipeline.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 0.765 | 0.755 | 0.580 | 0.656 | 0.797 |
| Random Forest | 0.821 | 0.814 | 0.696 | 0.750 | 0.830 |

The decision tree, all confusion matrices, and all ROC curves are saved in `charts/`.

## Imbalance, tuning, and regression

Training-only SMOTE gave the strongest imbalance-comparison F1 at 0.761, compared with 0.724 for the baseline and 0.755 for balanced class weights. Its precision was 0.740 and recall was 0.783, giving the best measured precision/recall balance on this split.

Grid search selected `max_depth=5`, `max_features="sqrt"`, and `n_estimators=100`. The best cross-validation F1 was 0.744, test F1 was 0.718, and the required OOB score was 0.826.

The multivariate fare regression produced MAE 20.898, RMSE 30.533, R2 0.398, and adjusted R2 0.362. The residual spread changes as predicted fare increases, and the correlation between predicted fare and absolute residual size is 0.546, so the plot shows heteroscedasticity.

Classification and regression numbers are kept in separate metric groups in [reports/model_results.md](reports/model_results.md) because they are not directly comparable.

## Recommendation and saved model

I would deploy the Random Forest from the three-model comparison. It produced the highest F1 (0.750) and accuracy (0.821), while its AUC was 0.830. Recall of 0.696 still leaves room to improve positive-case detection. Because this is a small historical dataset, I would continue checking the model on new data before using it for an important decision.

The selected complete preprocessing-plus-classifier pipeline is saved at `models/best_classifier_pipeline.joblib`. The reload script passes a raw row containing unscaled numbers and unencoded text directly to that saved object and successfully returns a prediction.
