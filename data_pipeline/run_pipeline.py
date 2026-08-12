"""Run scraping, cleaning, database loading, and querying in one command."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from database import build_database, open_database
from queries import JOIN_QUERY_NAME, QUERIES
from scrape_books import clean_books, scrape_books

MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data"
OUTPUT_DIR = MODULE_DIR / "outputs"
CSV_PATH = DATA_DIR / "books_cleaned.csv"
DATABASE_PATH = DATA_DIR / "books.db"
QUERY_OUTPUT_PATH = OUTPUT_DIR / "sql_query_results.txt"


def pandas_join(
    categories: pd.DataFrame, books: pd.DataFrame
) -> pd.DataFrame:
    """Reproduce the ranked SQL JOIN using pandas merge operations."""
    merged = books.merge(categories, on="category_id")
    merged = merged.rename(columns={"category_name": "category"})
    merged = merged.sort_values(
        ["category", "rating", "price_gbp", "title"],
        ascending=[True, False, False, True],
    )
    merged["category_rank"] = merged.groupby("category").cumcount() + 1
    return merged.loc[
        merged["category_rank"] <= 10,
        ["title", "rating", "price_gbp", "category", "category_rank"],
    ].reset_index(drop=True)


def run_queries(
    categories: pd.DataFrame, books: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Execute SQL queries, save their output, and verify the pandas JOIN."""
    results = {}
    with open_database(DATABASE_PATH) as connection:
        for name, query in QUERIES.items():
            results[name] = pd.read_sql(query, connection)

    sql_join = results[JOIN_QUERY_NAME].reset_index(drop=True)
    merged_join = pandas_join(categories, books)
    pd.testing.assert_frame_equal(sql_join, merged_join, check_dtype=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["Zepto Data Pipeline - SQL and pandas results", "=" * 48]
    for name, frame in results.items():
        lines.extend(["", name, "-" * len(name), frame.to_string(index=False)])
    lines.extend(
        [
            "",
            "pd.read_sql JOIN result",
            "-----------------------",
            sql_join.to_string(index=False),
            "",
            "pd.merge JOIN result",
            "--------------------",
            merged_join.to_string(index=False),
            "",
            "Equality check: PASSED",
        ]
    )
    QUERY_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    raw_rows = scrape_books()
    cleaned_books, dropped_count = clean_books(raw_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_books.to_csv(CSV_PATH, index=False)

    categories, normalized_books = build_database(cleaned_books, DATABASE_PATH)
    results = run_queries(categories, normalized_books)

    print(f"Scraped rows: {len(raw_rows)}")
    print(f"Clean rows: {len(cleaned_books)}")
    print(f"Dropped malformed rows: {dropped_count}")
    print(f"Categories: {len(categories)}")
    print(f"SQL queries executed: {len(results)}")
    print("pd.read_sql and pd.merge JOIN comparison: PASSED")
    print(f"Database: {DATABASE_PATH}")
    print(f"Query output: {QUERY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
