"""Run the per-file summarization pipeline against the `genuin/` folder.

Usage:
    python dev.py

Produces one short summary per file (~150-200 words), prints them to
the terminal, and writes the full result to `summary.json` — ready to
be fed into a knowledge-graph builder like Graphiti as episodes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app.graph.builder import build_graph

FOLDER = "genuin"
OUTPUT_FILE = Path("summary.json")


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


async def main() -> int:
    print(f"Summarizing folder: {FOLDER!r}")
    print("One LLM call per file, up to 5 in parallel.\n")

    graph = build_graph()
    try:
        state = await graph.ainvoke({"folder": FOLDER})
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1

    all_files = state.get("all_files", [])
    file_summaries = state.get("file_summaries", [])
    errors = state.get("errors", [])

    _print_header(f"Files in folder ({len(all_files)})")
    for f in all_files:
        print(f"  - {f['name']} ({f['size_bytes']} bytes)")

    for entry in file_summaries:
        _print_header(entry["name"])
        print(entry["summary"])

    _print_header(f"Done — {len(file_summaries)}/{len(all_files)} files summarized")

    if errors:
        _print_header(f"Warnings ({len(errors)})")
        for err in errors:
            print(f"  ! {err}")

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "folder": FOLDER,
                "all_files": all_files,
                "file_summaries": file_summaries,
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nPer-file summaries written to {OUTPUT_FILE.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
