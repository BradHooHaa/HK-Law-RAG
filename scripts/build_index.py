#!/usr/bin/env python3
import argparse
import hashlib
import re
import sqlite3
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "h": "http://www.xml.gov.hk/schemas/hklm/1.0",
    "dc": "http://purl.org/dc/elements/1.1/",
}

PRIMARY_UNIT_TAGS = {"section", "regulation", "rule", "article", "order"}
FALLBACK_UNIT_TAGS = {"paragraph"}


def local_name(tag):
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";")
    return text.strip()


def node_text(node):
    return clean_text(" ".join(node.itertext()))


def first_text(root, xpath):
    node = root.find(xpath, NS)
    return clean_text(" ".join(node.itertext())) if node is not None else ""


def make_source_url(identifier):
    if not identifier:
        return ""
    return "https://www.elegislation.gov.hk" + identifier


def xml_files(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".xml"):
                yield name


def iter_units(main):
    units = []
    for node in main.iter():
        if local_name(node.tag) in PRIMARY_UNIT_TAGS:
            units.append(node)
    if units:
        return units
    for node in main.iter():
        if local_name(node.tag) in FALLBACK_UNIT_TAGS:
            units.append(node)
    if units:
        return units
    return list(main)


def unit_ref(node):
    parts = []
    num = node.find("h:num", NS)
    heading = node.find("h:heading", NS)
    if num is not None:
        parts.append(node_text(num))
    if heading is not None:
        parts.append(node_text(heading))
    return clean_text(" ".join(parts))


def split_long_text(text, max_chars=1800, overlap=250):
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunk_start = start
        if chunk_start > 0:
            next_space = text.find(" ", chunk_start, min(chunk_start + 80, len(text)))
            if next_space != -1:
                chunk_start = next_space + 1
        chunk = text[chunk_start:end].strip()
        if start > 0 and chunk:
            chunk = "... " + chunk
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def parse_xml(zip_path, name):
    with zipfile.ZipFile(zip_path) as zf:
        root = ET.fromstring(zf.read(name))

    meta = {
        "doc_name": first_text(root, "h:meta/h:docName"),
        "doc_type": first_text(root, "h:meta/h:docType"),
        "doc_number": first_text(root, "h:meta/h:docNumber"),
        "doc_status": first_text(root, "h:meta/h:docStatus"),
        "date": first_text(root, "h:meta/dc:date"),
        "identifier": first_text(root, "h:meta/dc:identifier"),
        "language": first_text(root, "h:meta/dc:language"),
    }
    meta["source_url"] = make_source_url(meta["identifier"])

    short_title = first_text(root, ".//h:shortTitle")
    long_title = first_text(root, "h:main/h:longTitle")
    meta["title"] = short_title or long_title or meta["doc_name"]

    main = root.find("h:main", NS)
    if main is None:
        return []

    rows = []
    for unit_number, unit in enumerate(iter_units(main), start=1):
        text = node_text(unit)
        if len(text) < 30:
            continue

        ref = unit_ref(unit) or unit.get("name") or local_name(unit.tag)
        base_citation = clean_text(f"{meta['doc_name']} {ref}")
        for index, chunk in enumerate(split_long_text(text), start=1):
            body = clean_text(
                f"{meta['title']} | {base_citation} | Status: {meta['doc_status']} | "
                f"Date: {meta['date']} | {chunk}"
            )
            stable_id = hashlib.sha1(
                f"{name}:{unit_number}:{ref}:{index}:{chunk[:80]}".encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    **meta,
                    "id": stable_id,
                    "zip_member": name,
                    "unit_tag": local_name(unit.tag),
                    "unit_ref": ref,
                    "chunk_index": index,
                    "citation": base_citation,
                    "text": chunk,
                    "search_text": body,
                }
            )
    return rows


def init_db(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_fts;

        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            doc_name TEXT,
            doc_type TEXT,
            doc_number TEXT,
            title TEXT,
            doc_status TEXT,
            date TEXT,
            identifier TEXT,
            source_url TEXT,
            zip_member TEXT,
            unit_tag TEXT,
            unit_ref TEXT,
            chunk_index INTEGER,
            citation TEXT,
            text TEXT
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            search_text,
            content='chunks',
            content_rowid='rowid',
            tokenize='unicode61'
        );
        """
    )
    return con


def insert_rows(con, rows):
    con.executemany(
        """
        INSERT INTO chunks (
            id, doc_name, doc_type, doc_number, title, doc_status, date, identifier,
            source_url, zip_member, unit_tag, unit_ref, chunk_index, citation, text
        ) VALUES (
            :id, :doc_name, :doc_type, :doc_number, :title, :doc_status, :date, :identifier,
            :source_url, :zip_member, :unit_tag, :unit_ref, :chunk_index, :citation, :text
        )
        """,
        rows,
    )
    con.executemany(
        """
        INSERT INTO chunks_fts(rowid, search_text)
        SELECT rowid, :search_text FROM chunks WHERE id = :id
        """,
        rows,
    )


def main():
    parser = argparse.ArgumentParser(description="Build a local RAG search index for Hong Kong legislation XML.")
    parser.add_argument("--zip", required=True, help="Path to the downloaded legislation zip file.")
    parser.add_argument("--db", default="data/hk_legislation_fts.db", help="Output SQLite database path.")
    parser.add_argument("--limit", type=int, default=0, help="Only index the first N XML files, useful for testing.")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    db_path = Path(args.db)
    con = init_db(db_path)

    total_files = 0
    total_chunks = 0
    for name in xml_files(zip_path):
        total_files += 1
        rows = parse_xml(zip_path, name)
        if rows:
            insert_rows(con, rows)
            total_chunks += len(rows)
        if total_files % 100 == 0:
            con.commit()
            print(f"Indexed {total_files} XML files, {total_chunks} chunks...")
        if args.limit and total_files >= args.limit:
            break

    con.commit()
    con.close()
    print(f"Done. Indexed {total_files} XML files into {total_chunks} chunks.")
    print(f"Database: {db_path.resolve()}")


if __name__ == "__main__":
    main()
