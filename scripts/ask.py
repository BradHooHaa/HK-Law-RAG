#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from search import search


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_GLM_MODEL = "glm-4.7-flash"
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


SYSTEM_PROMPT = """You answer questions about Hong Kong legislation.
Use only the provided context.
If the context is insufficient, say what is missing.
Always cite the relevant Cap/section/date/source URL.
Do not give legal advice. Say that the answer is for legal information retrieval only.
Answer in the same language as the user's question."""


REWRITE_PROMPT = """Rewrite the user's question into concise English keyword queries for searching Hong Kong legislation.
Use terms likely to appear in statute text, such as creditor's petition, prescribed amount, statutory demand, liquidated sum, licence, offence, authority, employment contract.
Return only JSON in this shape:
{"queries":["query one","query two","query three"]}
Do not answer the question."""


def format_context(rows):
    blocks = []
    for idx, row in enumerate(rows, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] {row['citation']}",
                    f"Title: {row['title']}",
                    f"Status: {row['doc_status']}",
                    f"Date: {row['date']}",
                    f"Source: {row['source_url']}",
                    f"Text: {row['text']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_user_prompt(question, context):
    return f"""Question:
{question}

Context:
{context}"""


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def openai_client(api_key, base_url=None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The openai Python package is not installed.\n"
            "Install it with:\n"
            "python3 -m pip install openai"
        ) from exc

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def chat_text(client, model, system_prompt, user_prompt, temperature=0.2):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def call_openai(model, question, context):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "Set it first, for example:\n"
            "export OPENAI_API_KEY='your_api_key_here'"
        )

    client = openai_client(api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, context)},
        ],
    )

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Small compatibility fallback for SDK versions that expose raw output items.
    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def call_chat_completions(client, model, question, context):
    return chat_text(client, model, SYSTEM_PROMPT, build_user_prompt(question, context))


def answer_with_model(provider, model, question, context):
    if provider == "glm":
        api_key = os.environ.get("GLM_API_KEY")
        if not api_key:
            raise SystemExit(
                "GLM_API_KEY is not set.\n"
                "Set it first, for example:\n"
                "export GLM_API_KEY='your_glm_api_key_here'"
            )
        client = openai_client(api_key, base_url=GLM_BASE_URL)
        return call_chat_completions(client, model or DEFAULT_GLM_MODEL, question, context)

    return call_openai(model or DEFAULT_MODEL, question, context)


def parse_rewrite_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    queries = payload.get("queries", [])
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()]


def rewrite_queries(provider, model, question):
    if provider != "glm":
        return []

    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        return []

    client = openai_client(api_key, base_url=GLM_BASE_URL)
    text = chat_text(
        client,
        model or DEFAULT_GLM_MODEL,
        REWRITE_PROMPT,
        question,
        temperature=0,
    )
    return parse_rewrite_json(text)


def merge_rows(existing, new_rows, limit):
    seen = {row["citation"] + row["text"][:120] for row in existing}
    merged = list(existing)
    for row in new_rows:
        key = row["citation"] + row["text"][:120]
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
        if len(merged) >= limit:
            break
    return merged


def retrieve_rows(db_path, question, limit, provider, model, use_rewrite=True):
    search_limit = max(limit, 3)
    rows = search(db_path, question, search_limit)
    queries = []

    if use_rewrite and len(rows) < limit:
        queries = rewrite_queries(provider, model, question)
        for query in queries:
            rows = merge_rows(rows, search(db_path, query, search_limit), limit)
            if len(rows) >= limit:
                break

    return rows[:limit], queries


def print_context(rows):
    print("\nRetrieved context:")
    for idx, row in enumerate(rows, start=1):
        print(f"{idx}. {row['citation']} | {row['date']} | {row['source_url']}")


def print_queries(question, queries):
    print("\nSearch queries:")
    print(f"0. {question}")
    for idx, query in enumerate(queries, start=1):
        print(f"{idx}. {query}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask questions over the Hong Kong legislation RAG index.")
    parser.add_argument("question", help="Question to answer, for example: What licence is needed for dangerous goods?")
    parser.add_argument("--db", default="data/hk_legislation_fts.db", help="SQLite database path.")
    parser.add_argument("--limit", type=int, default=6, help="Number of retrieved chunks to use.")
    parser.add_argument("--provider", choices=["auto", "glm", "openai"], default="auto", help="Model provider to use.")
    parser.add_argument("--model", default=None, help="Model to use. Defaults to glm-4.7-flash for GLM or gpt-4.1-mini for OpenAI.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved citations before the answer.")
    parser.add_argument("--show-queries", action="store_true", help="Print the original and rewritten search queries.")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable model-assisted search query rewriting.")
    parser.add_argument("--context-only", action="store_true", help="Only retrieve context; do not call OpenAI.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}. Run scripts/build_index.py first.")

    provider = args.provider
    if provider == "auto":
        provider = "glm" if os.environ.get("GLM_API_KEY") else "openai"

    rows, queries = retrieve_rows(
        db_path,
        args.question,
        args.limit,
        provider,
        args.model,
        use_rewrite=not args.no_rewrite,
    )
    if not rows:
        raise SystemExit("No relevant legislation chunks found.")

    if args.show_queries:
        print_queries(args.question, queries)

    if args.show_context or args.context_only:
        print_context(rows)

    if args.context_only:
        return

    answer = answer_with_model(provider, args.model, args.question, format_context(rows))
    print(answer)


if __name__ == "__main__":
    main()
