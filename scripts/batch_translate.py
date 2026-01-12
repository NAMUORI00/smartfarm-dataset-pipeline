#!/usr/bin/env python3
"""Batch translation script for wasabi corpus EN->KO."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

INPUT_FILE = Path(__file__).parent.parent / "output" / "wasabi_web_en.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "output" / "wasabi_en_ko_parallel.jsonl"
BATCH_SIZE = 20
SLEEP_BETWEEN = 0.5


def load_existing_ids(path: Path) -> set:
    """Load already translated IDs."""
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                ids.add(row.get("id"))
            except:
                pass
    return ids


def translate_text(client: OpenAI, text: str, model: str = "gemini-2.5-flash") -> str:
    """Translate English text to Korean."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional translator specializing in agricultural science. "
                    "Translate the following English text to Korean. "
                    "Maintain technical terminology accuracy. "
                    "For plant scientific names, keep the Latin name in parentheses. "
                    "Output only the translation, no explanations."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


def main():
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("API_KEY"),
    )

    # Load input
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    print(f"Total chunks to translate: {len(rows)}")

    # Load existing translations (resume support)
    existing_ids = load_existing_ids(OUTPUT_FILE)
    print(f"Already translated: {len(existing_ids)}")

    # Filter to only untranslated
    to_translate = [r for r in rows if r["id"] not in existing_ids]
    print(f"Remaining: {len(to_translate)}")

    if not to_translate:
        print("All translations complete!")
        return

    # Translate in batches
    batch = []
    translated = 0
    errors = 0

    for i, row in enumerate(to_translate):
        try:
            text_ko = translate_text(client, row["text"])
            result = {
                "id": row["id"],
                "text_en": row["text"],
                "text_ko": text_ko,
                "metadata": row.get("metadata", {}),
                "translation": {
                    "model": "gemini-2.5-flash",
                    "src_lang": "en",
                    "tgt_lang": "ko",
                },
            }
            batch.append(result)
            translated += 1

            # Progress
            if translated % 10 == 0:
                print(f"Progress: {translated}/{len(to_translate)} ({100*translated/len(to_translate):.1f}%)")

        except Exception as e:
            print(f"Error translating {row['id']}: {e}")
            errors += 1
            time.sleep(2)  # Wait longer on error
            continue

        # Flush batch
        if len(batch) >= BATCH_SIZE:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for r in batch:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            batch.clear()

        time.sleep(SLEEP_BETWEEN)

    # Final flush
    if batch:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for r in batch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nComplete! Translated: {translated}, Errors: {errors}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
