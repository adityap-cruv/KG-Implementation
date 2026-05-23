"""Rank an existing file_scores.json without making any LLM calls.

Usage:
    python rank_scores.py
    python rank_scores.py file_scores.json ranked_files.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ranking import rank_scored_files


def rank_scores(input_path: Path, output_path: Path, weights_path: Path | None) -> list[dict]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    weights = None
    if weights_path is not None:
        weights = json.loads(weights_path.read_text(encoding="utf-8"))

    scored_files = payload.get("scored_files", [])
    ranked_files = rank_scored_files(scored_files, weights)

    output_payload = {
        **payload,
        "ranked_files": ranked_files,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return ranked_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ranked_files.json from existing file_scores.json."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="file_scores.json",
        help="Input scoring JSON file",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="ranked_files.json",
        help="Output ranked JSON file",
    )
    parser.add_argument(
        "--weights",
        help="Optional JSON file containing ranking weights",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights) if args.weights else None
    ranked_files = rank_scores(Path(args.input), Path(args.output), weights_path)

    print(f"Ranked {len(ranked_files)} files.")
    for item in ranked_files[:10]:
        print(f"{item['rank']}. {item['weighted_score']} - {item['name']}")
        print(f"   {item['reason']}")
    print(f"\nSaved ranked output to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
