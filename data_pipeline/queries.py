"""SQL queries used to explore the book database."""

QUERIES = {
    "1. Ten expensive in-stock books": """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        WHERE in_stock = 1
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "2. Distinct categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "3. Books in a middle GBP price range": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp, title;
    """,
    "4. Books with selected ratings": """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 15;
    """,
    "5. Ten highest-rated books in each category": """
        WITH ranked_books AS (
            SELECT
                b.title,
                b.rating,
                b.price_gbp,
                c.category_name AS category,
                ROW_NUMBER() OVER (
                    PARTITION BY c.category_name
                    ORDER BY b.rating DESC, b.price_gbp DESC, b.title
                ) AS category_rank
            FROM books AS b
            JOIN categories AS c ON b.category_id = c.category_id
        )
        SELECT title, rating, price_gbp, category, category_rank
        FROM ranked_books
        WHERE category_rank <= 10
        ORDER BY category, category_rank;
    """,
}

JOIN_QUERY_NAME = "5. Ten highest-rated books in each category"
