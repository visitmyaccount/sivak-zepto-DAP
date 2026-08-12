"""Reload the saved complete pipeline and predict from one raw row."""

from pathlib import Path

import joblib
import pandas as pd

MODULE_DIR = Path(__file__).parent
FEATURES = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]


def main() -> None:
    pipeline = joblib.load(MODULE_DIR / "models" / "best_classifier_pipeline.joblib")
    raw_data = pd.read_csv(MODULE_DIR / "titanic.csv")[FEATURES].iloc[[0]]
    prediction = int(pipeline.predict(raw_data)[0])
    print(f"Raw-row prediction: {prediction}")


if __name__ == "__main__":
    main()
