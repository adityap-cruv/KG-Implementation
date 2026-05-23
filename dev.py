"""Run the brand-summarizer pipeline against the `genuin/` folder.

Usage:
    python dev.py

Two modes, auto-dispatched:
  * If state/genuin.json does NOT exist  → ONBOARDING:
      identify foundational files → build base brand summary →
      summarize every file (grounded in base summary) →
      rank every file in ingestion order →
      write state/genuin.json

  * If state/genuin.json DOES exist      → UPDATE:
      reuse saved base summary (don't regenerate) →
      detect new files (files in folder not yet in state) →
      summarize ONLY the new files →
      rank the new files among themselves →
      append to state/genuin.json
"""

from __future__ import annotations

import asyncio
import sys

from app.graph.builder import build_onboarding_graph, build_update_graph
from app.state import load_state, storage_description

FOLDER = "genuin"


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


async def main() -> int:
    existing = load_state(FOLDER)
    mode = "update" if existing else "onboard"

    print(f"Folder:  {FOLDER!r}")
    print(f"Mode:    {mode!r}")
    print(f"Storage: {storage_description()}")
    if mode == "onboard":
        print(
            "Steps:  list → identify base files → build base summary → "
            "read all → summarize each → rank → save\n"
        )
        graph = build_onboarding_graph()
    else:
        print(
            "Steps:  load state → list → detect new → read new → "
            "summarize new → rank new → append + save\n"
        )
        graph = build_update_graph()

    try:
        state = await graph.ainvoke({"folder": FOLDER})
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1

    all_files = state.get("all_files", [])
    base_files = state.get("base_files", [])
    base_summary = state.get("base_summary", "")
    new_ranked = state.get("new_ranked_files", [])
    ranked_files = state.get("ranked_files", [])
    errors = state.get("errors", [])

    _print_header(f"Files in folder ({len(all_files)})")
    for f in all_files:
        print(f"  - {f['name']} ({f['size_bytes']} bytes)")

    if mode == "onboard":
        _print_header(f"Base files chosen ({len(base_files)})")
        for name in base_files:
            print(f"  * {name}")
        _print_header("Base brand summary")
        print(base_summary)

    _print_header(
        f"Newly processed this run ({len(new_ranked)})"
        if mode == "update"
        else f"Ingestion order ({len(new_ranked)} files)"
    )
    if not new_ranked:
        print("  (nothing new this run — folder is already fully processed)")
    else:
        for i, entry in enumerate(new_ranked, start=1):
            print(f"  #{i:>2}  {entry['name']:<55}  {entry['reason']}")

    if new_ranked:
        for entry in new_ranked:
            _print_header(entry["name"])
            print(f"(reason: {entry['reason']})\n")
            print(entry["summary"])

    _print_header(
        f"Done — full ranked list now has {len(ranked_files)} files; "
        f"{len(new_ranked)} added this run"
    )

    if errors:
        _print_header(f"Warnings ({len(errors)})")
        for err in errors:
            print(f"  ! {err}")

    print(f"\nState saved to {storage_description()} (key: folder={FOLDER!r})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
