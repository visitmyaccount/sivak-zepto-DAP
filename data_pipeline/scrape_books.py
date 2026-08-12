"""Scrape and clean book data from books.toscrape.com."""

from __future__ import annotations

import argparse
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATEGORY_NAMES = ("Mystery", "Historical Fiction", "Sequential Art")
GBP_TO_INR = Decimal("105.50")
RATING_VALUES = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def fetch_page(session: requests.Session, url: str) -> BeautifulSoup:
    """Download one page and return its parsed HTML."""
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def find_category_urls(session: requests.Session) -> dict[str, str]:
    """Find the URLs for the three selected categories."""
    soup = fetch_page(session, BASE_URL)
    links = {}
    for anchor in soup.select(".side_categories ul li ul li a"):
        name = anchor.get_text(strip=True)
        if name in CATEGORY_NAMES:
            links[name] = urljoin(BASE_URL, anchor["href"])

    missing = sorted(set(CATEGORY_NAMES) - set(links))
    if missing:
        raise RuntimeError(f"Could not find category links for: {', '.join(missing)}")
    return links


def scrape_category(
    session: requests.Session, category: str, first_page_url: str
) -> list[dict[str, str]]:
    """Scrape every listing page for one category."""
    rows = []
    page_url = first_page_url

    while page_url:
        soup = fetch_page(session, page_url)
        for book in soup.select("article.product_pod"):
            title_link = book.select_one("h3 a")
            price = book.select_one(".price_color")
            rating = book.select_one(".star-rating")
            availability = book.select_one(".availability")
            rows.append(
                {
                    "title": title_link.get("title", "") if title_link else "",
                    "price": price.get_text(strip=True) if price else "",
                    "star_rating": (
                        next(
                            (
                                value
                                for value in rating.get("class", [])
                                if value in RATING_VALUES
                            ),
                            "",
                        )
                        if rating
                        else ""
                    ),
                    "availability": (
                        availability.get_text(" ", strip=True) if availability else ""
                    ),
                    "category": category,
                }
            )

        next_link = soup.select_one("li.next a")
        page_url = urljoin(page_url, next_link["href"]) if next_link else ""

    return rows


def scrape_books() -> list[dict[str, str]]:
    """Scrape all selected categories."""
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Zepto-data-pipeline-learning-project/1.0"}
    )

    rows = []
    for category, url in find_category_urls(session).items():
        rows.extend(scrape_category(session, category, url))
    return rows


def parse_price(value: str) -> float | None:
    """Convert a displayed GBP price into a number."""
    match = re.search(r"(\d+(?:\.\d+)?)", value or "")
    return float(match.group(1)) if match else None


def parse_stock(value: str) -> bool | None:
    """Convert availability text into a boolean."""
    text = (value or "").strip().lower()
    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return None


def convert_to_inr(price_gbp: float) -> float:
    """Convert GBP to INR and round currency values to two decimal places."""
    converted = Decimal(str(price_gbp)) * GBP_TO_INR
    return float(converted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def clean_books(rows: list[dict[str, str]]) -> tuple[pd.DataFrame, int]:
    """Clean scraped rows and drop rows with an unparseable required field."""
    cleaned_rows = []
    for row in rows:
        price_gbp = parse_price(row.get("price", ""))
        rating = RATING_VALUES.get(row.get("star_rating", ""))
        in_stock = parse_stock(row.get("availability", ""))
        title = row.get("title", "").strip()
        category = row.get("category", "").strip()

        if not title or not category or price_gbp is None or rating is None or in_stock is None:
            continue

        cleaned_rows.append(
            {
                "title": title,
                "price_gbp": price_gbp,
                "rating": rating,
                "in_stock": in_stock,
                "price_inr": convert_to_inr(price_gbp),
                "category": category,
            }
        )

    frame = pd.DataFrame(
        cleaned_rows,
        columns=["title", "price_gbp", "rating", "in_stock", "price_inr", "category"],
    )
    if not frame.empty:
        frame = frame.astype(
            {
                "title": "string",
                "price_gbp": "float64",
                "rating": "int64",
                "in_stock": "bool",
                "price_inr": "float64",
                "category": "string",
            }
        )
    return frame, len(rows) - len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape and clean practice book data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "books_cleaned.csv",
    )
    args = parser.parse_args()

    raw_rows = scrape_books()
    books, dropped_count = clean_books(raw_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    books.to_csv(args.output, index=False)

    print(f"Scraped rows: {len(raw_rows)}")
    print(f"Clean rows: {len(books)}")
    print(f"Dropped malformed rows: {dropped_count}")
    print(f"Categories: {books['category'].nunique()}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
