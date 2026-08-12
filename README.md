# Zepto Data & AI Platform

This capstone repository contains three connected learning modules:

1. `data_pipeline` scrapes catalog data, cleans it, converts prices, and stores it in SQLite.
2. `analytics` explores the Titanic dataset and builds classification and regression models.
3. `support_assistant` answers questions using a small Zepto policy document collection.

The implementation uses simple Python scripts so each step can be followed from the command line. Each module has its own `requirements.txt` and README with setup and run instructions.

## Python setup

Python 3.11 is recommended. Create a separate virtual environment for each module so their dependencies remain easy to understand.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Then follow the module README files in this order:

1. [Data pipeline](data_pipeline/README.md)
2. [Analytics pipeline](analytics/README.md)
3. `support_assistant/README.md`

The data pipeline can be run from the repository root after installing its requirements:

```bash
python data_pipeline/run_pipeline.py
```

The analytics scripts continue from the single committed Titanic CSV:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
python analytics/predict_saved_model.py
```

## Design summary

The data pipeline uses a normalized two-table database so categories are stored once and books reference them by ID. The analytics module separates exploratory cleaning from train-only model preprocessing so evaluation data does not leak into training. The support assistant uses local embeddings and deterministic mock responses by default, while keeping the optional real-provider call behind one environment switch.

## Public repository safety

Do not commit real environment files or credentials. Read [SECURITY.md](SECURITY.md) before making changes.
