"""Run the brand summarization pipeline directly and print the result.

Usage:
    python dev.py                 # summarizes the `genuin/` folder
    python dev.py nike            # summarizes `nike/` instead
    python dev.py genuin --json   # also writes a full trace to summary.json

This bypasses the HTTP layer entirely — it invokes the compiled LangGraph
in-process and prints the brand summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.graph.builder import build_graph


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run(folder: str, write_json: bool) -> int:
    print(f"Summarizing folder: {folder!r}")
    print("This will make 2 LLM calls (file selection + summarization).\n")

    graph = build_graph()
    try:
        state = graph.invoke({"folder": folder})
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1

    all_files = state.get("all_files", [])
    selected = state.get("selected_files", [])
    skipped = state.get("skipped_files", [])
    summary = state.get("summary", "")
    errors = state.get("errors", [])

    _print_header(f"Files in folder ({len(all_files)})")
    for f in all_files:
        print(f"  - {f['name']} ({f['size_bytes']} bytes)")

    _print_header(f"LLM selected ({len(selected)})")
    for f in selected:
        print(f"  + {f['name']:<50} {f['reason']}")

    _print_header(f"LLM skipped ({len(skipped)})")
    for f in skipped:
        print(f"  - {f['name']:<50} {f['reason']}")

    _print_header("Brand summary")
    print(summary or "(empty)")

    if errors:
        _print_header(f"Warnings ({len(errors)})")
        for err in errors:
            print(f"  ! {err}")

    if write_json:
        out_path = Path("summary.json")
        out_path.write_text(
            json.dumps(
                {
                    "folder": folder,
                    "all_files": all_files,
                    "selected_files": selected,
                    "skipped_files": skipped,
                    "summary": summary,
                    "errors": errors,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFull trace written to {out_path.resolve()}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the brand summarizer pipeline and print the result."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="genuin",
        help="Folder name under BASE_DIR to summarize (default: genuin)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write the full pipeline trace to summary.json",
    )
    args = parser.parse_args()
    sys.exit(run(args.folder, args.json))


if __name__ == "__main__":
    main()
