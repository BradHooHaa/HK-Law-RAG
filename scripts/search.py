#!/usr/bin/env python3
import argparse
import sqlite3
import textwrap
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "may",
    "must",
    "of",
    "on",
    "or",
    "required",
    "shall",
    "should",
    "the",
    "to",
    "about",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def prepare_fts_query(query):
    terms = []
    for term in query.replace("'", " ").split():
        cleaned = "".join(ch for ch in term if ch.isalnum() or ch in {"_", "-"})
        cleaned = cleaned.strip("-_")
        if len(cleaned) >= 2 and cleaned.lower() not in STOPWORDS:
            terms.append(cleaned)
    return " ".join(terms) or query


def search(db_path, query, limit):
    fts_query = prepare_fts_query(query)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT
            c.doc_name,
            c.title,
            c.doc_status,
            c.date,
            c.citation,
            c.source_url,
            c.text,
            bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    con.close()
    return rows


def print_results(rows):
    if not rows:
        print("No results found.")
        return

    for idx, row in enumerate(rows, start=1):
        print("=" * 88)
        print(f"{idx}. {row['citation']}")
        print(f"   Title: {row['title']}")
        print(f"   Status: {row['doc_status']} | Date: {row['date']}")
        if row["source_url"]:
            print(f"   Source: {row['source_url']}")
        print()
        print(textwrap.fill(row["text"], width=88))
        print()


def main():
    parser = argparse.ArgumentParser(description="Search the Hong Kong legislation FTS index.")
    parser.add_argument("query", help="Search query, for example: water authority licensed plumber")
    parser.add_argument("--db", default="data/hk_legislation_fts.db", help="SQLite database path.")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to show.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}. Run scripts/build_index.py first.")

    print_results(search(db_path, args.query, args.limit))


if __name__ == "__main__":
    main()
