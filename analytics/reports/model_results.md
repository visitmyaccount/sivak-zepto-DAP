# Titanic Modeling Results

## Split and preprocessing

The target contains 549 non-survivors (61.6%) and 342 survivors (38.4%). A stratified 80/20 split keeps this class ratio similar in both train and test data, which makes the three model comparisons fairer.

The split happens before preprocessing. Numeric columns use a median imputer followed by `StandardScaler`; `sex` and `embarked` use most-frequent imputation and one-hot encoding. Each complete pipeline is fitted on the training split only, and the test split is passed only to `predict` and `predict_proba`.

## Classifier comparison

| model | accuracy | precision | recall | f1 | auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 0.765 | 0.755 | 0.580 | 0.656 | 0.797 |
| Random Forest | 0.821 | 0.814 | 0.696 | 0.750 | 0.830 |

The confusion matrices and ROC curves are saved in `charts/`. All three classifiers use the identical train/test split. The decision tree image labels the transformed feature names and both target classes.

## Imbalance handling

SMOTE is inside an imbalanced-learn pipeline after preprocessing, so it resamples only during training and never sees the test fold.

| method | precision | recall | f1 |
|---|---|---|---|
| Baseline | 0.793 | 0.667 | 0.724 |
| Balanced weights | 0.730 | 0.783 | 0.755 |
| Training-only SMOTE | 0.740 | 0.783 | 0.761 |

`Training-only SMOTE` produced the strongest F1 score (0.761) in this comparison. Its precision was 0.740 and recall was 0.783, so it gave the best measured balance between missed survivors and false survivor predictions on this split.

## Random Forest tuning

- Best parameters: `{'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100}`
- Best cross-validation F1: 0.744
- Test F1 for the tuned model: 0.718
- OOB score from `RandomForestClassifier(oob_score=True)`: 0.826

## Fare regression

| Metric | Value |
|---|---:|
| MAE | 20.898 |
| RMSE | 30.533 |
| R2 | 0.398 |
| Adjusted R2 | 0.362 |

The residual plot shows heteroscedasticity because the residual spread changes with predicted fare (correlation with absolute residual size: 0.546).

## Combined model table

Classification and regression results are kept in separate metric columns because they measure different kinds of predictions.

| model | class_accuracy | class_precision | class_recall | class_f1 | class_auc | reg_mae | reg_rmse | reg_r2 | reg_adjusted_r2 |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 | - | - | - | - |
| Decision Tree | 0.765 | 0.755 | 0.580 | 0.656 | 0.797 | - | - | - | - |
| Random Forest | 0.821 | 0.814 | 0.696 | 0.750 | 0.830 | - | - | - | - |
| Linear Regression (fare) | - | - | - | - | - | 20.898 | 30.533 | 0.398 | 0.362 |

## Recommendation

For this project, I selected the **Random Forest** classifier as the best of the three models compared. It achieved F1 0.750, accuracy 0.821, and AUC 0.830 on the held-out test set. This does not make it ready for real-world deployment because the results come from one split of a small historical dataset. Its recall of 0.696 also shows that it still misses some positive cases, so more validation with relevant data would be needed before considering practical use.

## Saved pipeline check

The highest test-F1 candidate, **Random Forest** (F1 0.750), was saved as one complete preprocessing-plus-classifier object at `models/best_classifier_pipeline.joblib`. Reloading it and predicting one raw test row succeeded with prediction `0`.
