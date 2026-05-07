#!/usr/bin/env python3
import argparse
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
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, context)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


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


def print_context(rows):
    print("\nRetrieved context:")
    for idx, row in enumerate(rows, start=1):
        print(f"{idx}. {row['citation']} | {row['date']} | {row['source_url']}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask questions over the Hong Kong legislation RAG index.")
    parser.add_argument("question", help="Question to answer, for example: What licence is needed for dangerous goods?")
    parser.add_argument("--db", default="data/hk_legislation_fts.db", help="SQLite database path.")
    parser.add_argument("--limit", type=int, default=6, help="Number of retrieved chunks to use.")
    parser.add_argument("--provider", choices=["auto", "glm", "openai"], default="auto", help="Model provider to use.")
    parser.add_argument("--model", default=None, help="Model to use. Defaults to glm-4.7-flash for GLM or gpt-4.1-mini for OpenAI.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved citations before the answer.")
    parser.add_argument("--context-only", action="store_true", help="Only retrieve context; do not call OpenAI.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}. Run scripts/build_index.py first.")

    rows = search(db_path, args.question, args.limit)
    if not rows:
        raise SystemExit("No relevant legislation chunks found. Try a more specific English query.")

    if args.show_context or args.context_only:
        print_context(rows)

    if args.context_only:
        return

    provider = args.provider
    if provider == "auto":
        provider = "glm" if os.environ.get("GLM_API_KEY") else "openai"

    answer = answer_with_model(provider, args.model, args.question, format_context(rows))
    print(answer)


if __name__ == "__main__":
    main()
