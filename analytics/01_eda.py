"""Load Titanic once, clean an EDA copy, and create the data story."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MODULE_DIR = Path(__file__).parent
RAW_CSV = MODULE_DIR / "titanic.csv"
CLEANED_CSV = MODULE_DIR / "titanic_cleaned.csv"
CHART_DIR = MODULE_DIR / "charts"
REPORT_DIR = MODULE_DIR / "reports"
EDA_REPORT = REPORT_DIR / "eda_results.md"
CORRELATION_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]


def load_raw_data() -> tuple[pd.DataFrame, str]:
    """Use the committed fallback when present, otherwise load and save it once."""
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV), "committed CSV fallback"

    dataframe = sns.load_dataset("titanic")
    dataframe.to_csv(RAW_CSV, index=False)
    return dataframe, "Seaborn loader"


def clean_for_eda(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply the assignment's percentage thresholds to an EDA copy."""
    cleaned = dataframe.copy()
    missing_percent = dataframe.isna().mean().mul(100)
    decisions = []

    high_missing = missing_percent[(missing_percent >= 30) & (missing_percent > 0)]
    for column, percent in high_missing.items():
        cleaned = cleaned.drop(columns=column)
        decisions.append(
            f"- `{column}`: {percent:.2f}% missing, so the column was dropped because "
            "filling more than 30% would be unreliable."
        )

    middle_missing = missing_percent[
        (missing_percent >= 5) & (missing_percent < 30)
    ]
    for column, percent in middle_missing.items():
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            fill_value = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(fill_value)
            decisions.append(
                f"- `{column}`: {percent:.2f}% missing, so values were filled with "
                f"the median ({fill_value:.2f}) under the 5%-30% rule."
            )
        else:
            fill_value = cleaned[column].mode().iloc[0]
            cleaned[column] = cleaned[column].fillna(fill_value)
            decisions.append(
                f"- `{column}`: {percent:.2f}% missing, so values were filled with "
                f"the mode (`{fill_value}`) under the 5%-30% rule."
            )

    low_missing = [
        column
        for column, percent in missing_percent.items()
        if 0 < percent < 5 and column in cleaned.columns
    ]
    for column in low_missing:
        decisions.append(
            f"- `{column}`: {missing_percent[column]:.2f}% missing, so affected rows "
            "were dropped under the under-5% rule."
        )
    if low_missing:
        cleaned = cleaned.dropna(subset=low_missing)

    return cleaned, decisions


def outlier_count(series: pd.Series) -> tuple[int, float, float]:
    """Count values outside the 1.5 x IQR limits."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum()), lower, upper


def save_basic_plots(dataframe: pd.DataFrame) -> None:
    """Save the required univariate plots."""
    for column in ("age", "fare"):
        plt.figure(figsize=(7, 4))
        sns.histplot(data=dataframe, x=column, bins=30, kde=True)
        plt.title(f"Distribution of {column.title()}")
        plt.tight_layout()
        plt.savefig(CHART_DIR / f"{column}_histogram.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 3))
        sns.boxplot(data=dataframe, x=column)
        plt.title(f"Box Plot of {column.title()}")
        plt.tight_layout()
        plt.savefig(CHART_DIR / f"{column}_boxplot.png", dpi=150)
        plt.close()


def save_data_story_plots(dataframe: pd.DataFrame, correlation: pd.DataFrame) -> None:
    """Save the correlation heatmap and four multivariate charts."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Titanic Numeric Correlation Matrix")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=dataframe, x="pclass", y="survived", hue="sex")
    plt.ylabel("Survival rate")
    plt.title("Survival Rate by Sex and Passenger Class")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "story_1_survival_sex_class.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=dataframe, x="survived", y="age")
    plt.xticks([0, 1], ["Did not survive", "Survived"])
    plt.title("Age by Survival Outcome")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "story_2_age_survival.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=dataframe, x="age", y="fare", hue="survived", alpha=0.65)
    plt.title("Fare and Age by Survival Outcome")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "story_3_age_fare_survival.png", dpi=150)
    plt.close()

    story_frame = dataframe.assign(
        family_size=dataframe["sibsp"] + dataframe["parch"] + 1
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=story_frame, x="family_size", y="survived")
    plt.ylabel("Survival rate")
    plt.title("Survival Rate by Family Size")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "story_4_family_survival.png", dpi=150)
    plt.close()


def strongest_correlations(correlation: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Return the two largest absolute off-diagonal correlation pairs."""
    pairs = []
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1 :]:
            pairs.append((left, right, correlation.loc[left, right]))
    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)[:2]


def format_rates(dataframe: pd.DataFrame) -> tuple[str, str, str]:
    """Calculate survival rates using explicit boolean masks."""
    sex_lines = []
    for sex in sorted(dataframe["sex"].dropna().unique()):
        mask = (dataframe["sex"] == sex) & dataframe["survived"].notna()
        sex_lines.append(f"- {sex}: {dataframe.loc[mask, 'survived'].mean():.3f}")

    class_lines = []
    for passenger_class in sorted(dataframe["pclass"].dropna().unique()):
        mask = (dataframe["pclass"] == passenger_class) & dataframe["survived"].notna()
        class_lines.append(
            f"- Class {passenger_class}: {dataframe.loc[mask, 'survived'].mean():.3f}"
        )

    combined_lines = []
    for sex in sorted(dataframe["sex"].dropna().unique()):
        for passenger_class in sorted(dataframe["pclass"].dropna().unique()):
            mask = (
                (dataframe["sex"] == sex)
                & (dataframe["pclass"] == passenger_class)
                & dataframe["survived"].notna()
            )
            combined_lines.append(
                f"- {sex}, class {passenger_class}: "
                f"{dataframe.loc[mask, 'survived'].mean():.3f}"
            )
    return "\n".join(sex_lines), "\n".join(class_lines), "\n".join(combined_lines)


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw, source = load_raw_data()
    missing_percent = raw.isna().mean().mul(100)
    affected_missing = missing_percent[missing_percent > 0]
    cleaned, decisions = clean_for_eda(raw)
    cleaned.to_csv(CLEANED_CSV, index=False)

    info_buffer = io.StringIO()
    raw.info(buf=info_buffer)
    description = raw.describe(include="all").transpose()
    info_text = "\n".join(line.rstrip() for line in info_buffer.getvalue().splitlines())
    description_text = "\n".join(
        line.rstrip() for line in description.to_string().splitlines()
    )

    age_outliers, age_lower, age_upper = outlier_count(cleaned["age"])
    fare_outliers, fare_lower, fare_upper = outlier_count(cleaned["fare"])
    fare_mean = cleaned["fare"].mean()
    fare_median = cleaned["fare"].median()
    fare_mode = cleaned["fare"].mode().iloc[0]
    skew = "right-skewed" if fare_mean > fare_median > fare_mode else "not clearly ordered"

    correlation = cleaned[CORRELATION_COLUMNS].corr()
    top_pairs = strongest_correlations(correlation)
    sex_rates, class_rates, combined_rates = format_rates(cleaned)

    standardized = cleaned[["age", "fare"]].copy()
    standardized = (standardized - standardized.mean()) / standardized.std()
    standard_summary = pd.DataFrame(
        {
            "before_mean": cleaned[["age", "fare"]].mean(),
            "before_std": cleaned[["age", "fare"]].std(),
            "after_mean": standardized.mean(),
            "after_std": standardized.std(),
        }
    )

    save_basic_plots(cleaned)
    save_data_story_plots(cleaned, correlation)

    female_first = cleaned[
        (cleaned["sex"] == "female") & (cleaned["pclass"] == 1)
    ]["survived"].mean()
    male_third = cleaned[
        (cleaned["sex"] == "male") & (cleaned["pclass"] == 3)
    ]["survived"].mean()
    survived_age = cleaned.loc[cleaned["survived"] == 1, "age"].median()
    not_survived_age = cleaned.loc[cleaned["survived"] == 0, "age"].median()
    survived_fare = cleaned.loc[cleaned["survived"] == 1, "fare"].median()
    not_survived_fare = cleaned.loc[cleaned["survived"] == 0, "fare"].median()
    family_frame = cleaned.assign(family_size=cleaned["sibsp"] + cleaned["parch"] + 1)
    family_rates = family_frame.groupby("family_size")["survived"].agg(["mean", "count"])
    common_family_rates = family_rates[family_rates["count"] >= 10]
    best_family_size = int(common_family_rates["mean"].idxmax())
    best_family_rate = common_family_rates.loc[best_family_size, "mean"]

    missing_text = "\n".join(
        f"- `{column}`: {percent:.2f}%" for column, percent in affected_missing.items()
    )
    top_pair_text = "\n".join(
        f"- `{left}` and `{right}`: {value:.3f}"
        for left, right, value in top_pairs
    )

    report = f"""# Titanic EDA Results

## Load and profile

- Source used in this run: {source}
- Shape: {raw.shape[0]} rows x {raw.shape[1]} columns

### `df.info()`

```text
{info_text}
```

### `df.describe(include="all")`

```text
{description_text}
```

## Missing values before cleaning

{missing_text}

### Decisions

{chr(10).join(decisions)}

The cleaned EDA copy contains {len(cleaned)} rows and {len(cleaned.columns)} columns.

## Univariate findings

- Age IQR limits: {age_lower:.2f} to {age_upper:.2f}; outliers: **{age_outliers}**.
- Fare IQR limits: {fare_lower:.2f} to {fare_upper:.2f}; outliers: **{fare_outliers}**.
- Fare mean: {fare_mean:.3f}; median: {fare_median:.3f}; mode: {fare_mode:.3f}.
- Fare is **{skew}** because mean > median > mode. A small number of very high fares pull the mean upward.

## Survival rates

### By sex

{sex_rates}

### By passenger class

{class_rates}

### By sex and passenger class

{combined_rates}

## Correlation matrix

The matrix uses exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`. The derived boolean columns `adult_male` and `alone` are excluded.

```text
{correlation.round(3).to_string()}
```

The two strongest absolute off-diagonal pairs are:

{top_pair_text}

The class/fare relationship reflects the much higher fares paid in the lower-numbered passenger classes. The positive `sibsp`/`parch` relationship suggests that passengers travelling with siblings or a spouse often also travelled with parents or children as part of a family group.

## Four-chart data story

1. **Sex and class:** First-class female survival was {female_first:.1%}, compared with {male_third:.1%} for third-class males. Sex is the clearest separation, while passenger class adds another strong difference inside each group.
2. **Age and survival:** Median age was {survived_age:.1f} for survivors and {not_survived_age:.1f} for non-survivors. The boxes overlap greatly, so age alone does not separate the outcomes as clearly as sex and class.
3. **Age, fare, and survival:** Survivors paid a median fare of {survived_fare:.2f}, versus {not_survived_fare:.2f} for non-survivors. Higher-fare observations contain more survivors, which agrees with the class pattern, although both outcomes occur across the age range.
4. **Family size:** Among family sizes with at least 10 passengers, size {best_family_size} had the highest survival rate at {best_family_rate:.1%}. Travelling completely alone or in very large groups was less favorable than travelling with a small family group.

## Exploratory standardization check

This check uses the full cleaned EDA copy only and is not passed into the modeling pipeline.

```text
{standard_summary.to_string(float_format=lambda value: f'{value:.6f}')}
```

After z-score standardization, age and fare both have means close to 0 and standard deviations of 1.
"""
    EDA_REPORT.write_text(report, encoding="utf-8")

    print(f"Raw shape: {raw.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Age outliers: {age_outliers}")
    print(f"Fare outliers: {fare_outliers}")
    print(f"Charts saved: {len(list(CHART_DIR.glob('*.png')))}")
    print(f"Report: {EDA_REPORT}")


if __name__ == "__main__":
    main()
