# Data Pipeline

This module collects practice catalog data, cleans it, converts prices, stores it in SQLite, and queries it with SQL and pandas.

## Setup

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r data_pipeline/requirements.txt
```

## Run

```bash
python data_pipeline/run_pipeline.py
```

The command recreates these artifacts:

- `data/books_cleaned.csv`
- `data/books.db`
- `outputs/sql_query_results.txt`

An internet connection is needed while scraping `books.toscrape.com`.

## Scraping result

The completed run collected 133 books from three categories:

| Category | Rows |
|---|---:|
| Historical Fiction | 26 |
| Mystery | 32 |
| Sequential Art | 75 |

All category pages are followed until their `next` link is no longer present.

## Cleaning decisions

- The price currency symbol is removed and the value is stored as a floating-point `price_gbp` column.
- Rating words from `One` through `Five` are mapped to integers from 1 through 5.
- Availability text is converted to a boolean `in_stock` column.
- INR prices use the fixed project baseline **1 GBP = 105.50 INR**. Decimal half-up rounding is used to keep two currency decimal places.
- A row is dropped if any required value cannot be parsed. This was chosen instead of filling only the numeric values because an unknown title, category, or availability cannot be repaired with a median. The recorded run dropped 0 rows, but the check keeps the pipeline from crashing if the page contains unexpected text later.

## Database design

The SQLite database uses two normalized tables:

```text
categories(category_id PK, category_name UNIQUE)
       1
       |
       *
books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)
```

Foreign-key checking is enabled whenever the database is opened. The database is rebuilt from the cleaned CSV during every full run.

## Queries and pandas comparison

Five saved queries demonstrate:

- `SELECT`, `WHERE`, `ORDER BY`, and `LIMIT`
- `DISTINCT`
- `BETWEEN`
- `IN`
- a `JOIN` between books and categories

The JOIN ranks the ten highest-rated books inside each category. Query results are loaded into pandas with `pd.read_sql`. The same result is recreated from the in-memory tables using `pd.merge`, sorting, grouping, and ranking. `pd.testing.assert_frame_equal` confirms that the SQL and pandas results match; the completed output records `Equality check: PASSED`.
