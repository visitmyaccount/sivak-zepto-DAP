"""Train, compare, tune, and save models using the committed Titanic CSV."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

MODULE_DIR = Path(__file__).parent
RAW_CSV = MODULE_DIR / "titanic.csv"
CHART_DIR = MODULE_DIR / "charts"
REPORT_DIR = MODULE_DIR / "reports"
MODEL_DIR = MODULE_DIR / "models"
MODEL_REPORT = REPORT_DIR / "model_results.md"
MODEL_PATH = MODEL_DIR / "best_classifier_pipeline.joblib"
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def make_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Create fresh preprocessing steps for numeric and categorical columns."""
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_steps = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_steps, numeric_features),
            ("categorical", categorical_steps, categorical_features),
        ]
    )


def make_classifier_pipeline(model: object) -> Pipeline:
    """Put preprocessing and a classifier into one fitted object."""
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])


def classification_metrics(
    model: Pipeline | ImbalancedPipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Return the assignment's classification metrics and predictions."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "auc": auc(false_positive_rate, true_positive_rate),
    }
    return metrics, predictions, probabilities


def markdown_table(dataframe: pd.DataFrame, digits: int = 3) -> str:
    """Create a small Markdown table without an extra formatting dependency."""
    display = dataframe.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_classifier_charts(
    models: dict[str, Pipeline],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
) -> None:
    """Save confusion matrices, ROC curves, and the decision tree."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (name, model) in zip(axes, models.items()):
        predictions = model.predict(x_test)
        ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, predictions),
            display_labels=["Not survived", "Survived"],
        ).plot(ax=axis, colorbar=False)
        axis.set_title(name)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "classifier_confusion_matrices.png", dpi=150)
    plt.close(figure)

    plt.figure(figsize=(7, 6))
    for name, values in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, values)
        area = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={area:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Classifier ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "classifier_roc_curves.png", dpi=150)
    plt.close()

    decision_tree = models["Decision Tree"]
    feature_names = decision_tree.named_steps["preprocess"].get_feature_names_out()
    plt.figure(figsize=(22, 10))
    plot_tree(
        decision_tree.named_steps["model"],
        feature_names=feature_names,
        class_names=["Not survived", "Survived"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Decision Tree (first four levels shown)")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "decision_tree.png", dpi=150)
    plt.close()


def run_regression(dataframe: pd.DataFrame) -> tuple[dict[str, float], str]:
    """Predict fare and save the residual plot."""
    numeric_features = ["pclass", "age", "sibsp", "parch", "survived"]
    categorical_features = ["sex", "embarked"]
    regression_features = numeric_features + categorical_features
    x_data = dataframe[regression_features]
    y_data = dataframe["fare"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.20, random_state=RANDOM_STATE
    )

    regression_pipeline = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(numeric_features, categorical_features),
            ),
            ("model", LinearRegression()),
        ]
    )
    regression_pipeline.fit(x_train, y_train)
    predictions = regression_pipeline.predict(x_test)
    residuals = y_test.to_numpy() - predictions

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r_squared = r2_score(y_test, predictions)
    transformed_count = len(
        regression_pipeline.named_steps["preprocess"].get_feature_names_out()
    )
    sample_count = len(y_test)
    adjusted_r_squared = 1 - (1 - r_squared) * (sample_count - 1) / (
        sample_count - transformed_count - 1
    )
    spread_correlation = float(np.corrcoef(predictions, np.abs(residuals))[0, 1])
    conclusion = (
        "The residual plot shows heteroscedasticity because the residual spread changes "
        f"with predicted fare (correlation with absolute residual size: {spread_correlation:.3f})."
        if abs(spread_correlation) >= 0.20
        else "The residual plot does not show strong heteroscedasticity because the residual "
        f"spread is fairly random (correlation with absolute residual size: {spread_correlation:.3f})."
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(predictions, residuals, alpha=0.65)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Linear Regression Residual Plot")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "fare_regression_residuals.png", dpi=150)
    plt.close()

    return (
        {
            "mae": mae,
            "rmse": rmse,
            "r2": r_squared,
            "adjusted_r2": adjusted_r_squared,
        },
        conclusion,
    )


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError("Run 01_eda.py first to create titanic.csv")

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(RAW_CSV)
    x_data = dataframe[FEATURES]
    y_data = dataframe["survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_data,
    )

    class_balance = y_data.value_counts().sort_index()
    class_percent = y_data.value_counts(normalize=True).sort_index().mul(100)

    classifiers = {
        "Logistic Regression": make_classifier_pipeline(
            LogisticRegression(
                max_iter=1000, solver="liblinear", random_state=RANDOM_STATE
            )
        ),
        "Decision Tree": make_classifier_pipeline(
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
        ),
        "Random Forest": make_classifier_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        ),
    }

    metric_rows = []
    probability_values = {}
    fitted_models = {}
    for name, pipeline in classifiers.items():
        pipeline.fit(x_train, y_train)
        metrics, _, probabilities = classification_metrics(pipeline, x_test, y_test)
        metric_rows.append({"model": name, **metrics})
        probability_values[name] = probabilities
        fitted_models[name] = pipeline
    classifier_results = pd.DataFrame(metric_rows)
    save_classifier_charts(fitted_models, x_test, y_test, probability_values)

    imbalance_models = {
        "Baseline": make_classifier_pipeline(
            LogisticRegression(
                max_iter=1000, solver="liblinear", random_state=RANDOM_STATE
            )
        ),
        "Balanced weights": make_classifier_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
        ),
        "Training-only SMOTE": ImbalancedPipeline(
            [
                ("preprocess", make_preprocessor()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    imbalance_rows = []
    for name, pipeline in imbalance_models.items():
        pipeline.fit(x_train, y_train)
        metrics, _, _ = classification_metrics(pipeline, x_test, y_test)
        imbalance_rows.append(
            {
                "method": name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    imbalance_results = pd.DataFrame(imbalance_rows)
    best_imbalance = imbalance_results.loc[imbalance_results["f1"].idxmax()]

    tuning_pipeline = make_classifier_pipeline(
        RandomForestClassifier(
            random_state=RANDOM_STATE, oob_score=True, bootstrap=True
        )
    )
    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(
        tuning_pipeline,
        parameter_grid,
        cv=5,
        scoring="f1",
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    tuned_metrics, _, _ = classification_metrics(search.best_estimator_, x_test, y_test)
    oob_score = search.best_estimator_.named_steps["model"].oob_score_

    model_candidates = {
        row["model"]: (fitted_models[row["model"]], row["f1"])
        for row in metric_rows
    }
    model_candidates["Tuned Random Forest"] = (
        search.best_estimator_,
        tuned_metrics["f1"],
    )
    selected_name, (selected_pipeline, selected_f1) = max(
        model_candidates.items(), key=lambda item: item[1][1]
    )
    joblib.dump(selected_pipeline, MODEL_PATH)
    reloaded_pipeline = joblib.load(MODEL_PATH)
    reload_prediction = int(reloaded_pipeline.predict(x_test.iloc[[0]])[0])

    regression_metrics, residual_conclusion = run_regression(dataframe)

    recommended_row = classifier_results.loc[classifier_results["f1"].idxmax()]
    comparison_rows = []
    for row in classifier_results.to_dict(orient="records"):
        comparison_rows.append(
            {
                "model": row["model"],
                "class_accuracy": f"{row['accuracy']:.3f}",
                "class_precision": f"{row['precision']:.3f}",
                "class_recall": f"{row['recall']:.3f}",
                "class_f1": f"{row['f1']:.3f}",
                "class_auc": f"{row['auc']:.3f}",
                "reg_mae": "-",
                "reg_rmse": "-",
                "reg_r2": "-",
                "reg_adjusted_r2": "-",
            }
        )
    comparison_rows.append(
        {
            "model": "Linear Regression (fare)",
            "class_accuracy": "-",
            "class_precision": "-",
            "class_recall": "-",
            "class_f1": "-",
            "class_auc": "-",
            "reg_mae": f"{regression_metrics['mae']:.3f}",
            "reg_rmse": f"{regression_metrics['rmse']:.3f}",
            "reg_r2": f"{regression_metrics['r2']:.3f}",
            "reg_adjusted_r2": f"{regression_metrics['adjusted_r2']:.3f}",
        }
    )
    comparison_table = pd.DataFrame(comparison_rows)

    best_parameters = {
        key.replace("model__", ""): value for key, value in search.best_params_.items()
    }
    report = f"""# Titanic Modeling Results

## Split and preprocessing

The target contains {class_balance[0]} non-survivors ({class_percent[0]:.1f}%) and {class_balance[1]} survivors ({class_percent[1]:.1f}%). A stratified 80/20 split keeps this class ratio similar in both train and test data, which makes the three model comparisons fairer.

The split happens before preprocessing. Numeric columns use a median imputer followed by `StandardScaler`; `sex` and `embarked` use most-frequent imputation and one-hot encoding. Each complete pipeline is fitted on the training split only, and the test split is passed only to `predict` and `predict_proba`.

## Classifier comparison

{markdown_table(classifier_results)}

The confusion matrices and ROC curves are saved in `charts/`. All three classifiers use the identical train/test split. The decision tree image labels the transformed feature names and both target classes.

## Imbalance handling

SMOTE is inside an imbalanced-learn pipeline after preprocessing, so it resamples only during training and never sees the test fold.

{markdown_table(imbalance_results)}

`{best_imbalance['method']}` produced the strongest F1 score ({best_imbalance['f1']:.3f}) in this comparison. Its precision was {best_imbalance['precision']:.3f} and recall was {best_imbalance['recall']:.3f}, so it gave the best measured balance between missed survivors and false survivor predictions on this split.

## Random Forest tuning

- Best parameters: `{best_parameters}`
- Best cross-validation F1: {search.best_score_:.3f}
- Test F1 for the tuned model: {tuned_metrics['f1']:.3f}
- OOB score from `RandomForestClassifier(oob_score=True)`: {oob_score:.3f}

## Fare regression

| Metric | Value |
|---|---:|
| MAE | {regression_metrics['mae']:.3f} |
| RMSE | {regression_metrics['rmse']:.3f} |
| R2 | {regression_metrics['r2']:.3f} |
| Adjusted R2 | {regression_metrics['adjusted_r2']:.3f} |

{residual_conclusion}

## Combined model table

Classification and regression results are kept in separate metric columns because they measure different kinds of predictions.

{markdown_table(comparison_table)}

## Recommendation

I would deploy the **{recommended_row['model']}** classifier from the three-model comparison. It achieved F1 {recommended_row['f1']:.3f}, accuracy {recommended_row['accuracy']:.3f}, and AUC {recommended_row['auc']:.3f} on the held-out test set. Its recall of {recommended_row['recall']:.3f} is important because a survival model should not overlook too many positive cases. These results are based on one small historical dataset, so I would continue checking the model on new data before using it for an important decision.

## Saved pipeline check

The highest test-F1 candidate, **{selected_name}** (F1 {selected_f1:.3f}), was saved as one complete preprocessing-plus-classifier object at `models/best_classifier_pipeline.joblib`. Reloading it and predicting one raw test row succeeded with prediction `{reload_prediction}`.
"""
    MODEL_REPORT.write_text(report, encoding="utf-8")

    print(classifier_results.to_string(index=False))
    print(f"Best grid parameters: {best_parameters}")
    print(f"OOB score: {oob_score:.3f}")
    print(f"Saved pipeline: {selected_name}")
    print(f"Reload prediction: {reload_prediction}")
    print(f"Report: {MODEL_REPORT}")


if __name__ == "__main__":
    main()
