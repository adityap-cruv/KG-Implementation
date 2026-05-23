"""Run the file-scoring pipeline directly and print per-file scores.

Usage:
    python dev_score.py genuin summary.json
    python dev_score.py genuin summary.json --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["BASE_DIR"] = str(Path(__file__).resolve().parent)

from app.graph.builder import build_scoring_graph


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run(folder: str, summary_file: str, write_json: bool) -> int:
    print(f"Scoring folder: {folder!r}")
    print(f"Using company summary: {summary_file!r}\n")

    graph = build_scoring_graph()
    try:
        state = graph.invoke({"folder": folder, "summary_file": summary_file})
    except Exception as exc:
        print(f"\nScoring pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    scored_files = state.get("scored_files", [])
    errors = state.get("errors", [])

    _print_header(f"Scored files ({len(scored_files)})")
    for item in scored_files:
        print(
            f"  {item.get('rank', '-')}. {item['name']} "
            f"(weighted={item.get('weighted_score', '-')}, {item['size_bytes']} bytes)"
        )
        print(
            "    "
            f"authority={item['authority_score']} "
            f"foundational={item['foundational_value']} "
            f"currentness={item['currentness_score']} "
            f"entity_rel={item['entity_relationship_clarity']} "
            f"content={item['content_specificity']} "
            f"structure={item['document_structure_quality']}"
        )
        print(
            "    "
            f"conflict={item['conflict_risk']} "
            f"duplicate={item['duplicate_penalty']} "
            f"noise={item['noise_penalty']} "
            f"ambiguity={item['ambiguity_penalty']}"
        )
        print(f"    reason: {item['reason']}")

    if errors:
        _print_header(f"Warnings ({len(errors)})")
        for err in errors:
            print(f"  ! {err}")

    if write_json:
        out_path = Path("file_scores.json")
        out_path.write_text(
            json.dumps(
                {
                    "folder": folder,
                    "summary_file": summary_file,
                    "all_files": state.get("all_files", []),
                    "scored_files": scored_files,
                    "errors": errors,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFull scoring trace written to {out_path.resolve()}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the file scoring pipeline and print per-file scores."
    )
    parser.add_argument("folder", help="Folder name under BASE_DIR to score")
    parser.add_argument(
        "summary_file",
        help="Company summary text file in the brand folder, or summary.json under BASE_DIR",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write the full scoring trace to file_scores.json",
    )
    args = parser.parse_args()
    sys.exit(run(args.folder, args.summary_file, args.json))


if __name__ == "__main__":
    main()
