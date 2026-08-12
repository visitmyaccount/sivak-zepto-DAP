"""Create and load the normalized SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def open_database(database_path: Path) -> sqlite3.Connection:
    """Open SQLite and make sure foreign keys are checked."""
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def build_database(
    books: pd.DataFrame, database_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the database and return the normalized in-memory tables."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    categories = pd.DataFrame(
        {
            "category_id": range(1, books["category"].nunique() + 1),
            "category_name": sorted(books["category"].unique()),
        }
    )
    normalized_books = (
        books.merge(categories, left_on="category", right_on="category_name")
        .drop(columns=["category", "category_name"])
        .reset_index(drop=True)
    )
    normalized_books.insert(0, "book_id", range(1, len(normalized_books) + 1))
    normalized_books["in_stock"] = normalized_books["in_stock"].astype(int)

    with open_database(database_path) as connection:
        connection.executescript(SCHEMA)
        categories.to_sql("categories", connection, if_exists="append", index=False)
        normalized_books.to_sql("books", connection, if_exists="append", index=False)

        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise RuntimeError(f"Foreign key errors found: {foreign_key_errors}")

    return categories, normalized_books
