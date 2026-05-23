# Brand Summarizer

A two-mode LangGraph pipeline that processes a folder of markdown files
about a brand into:

1. A **base brand summary** built from the foundational pages, and
2. A **per-file summary** for every file in the folder,
3. Ranked into a single **ingestion order** ready for Graphiti (or any
   sequential knowledge-graph builder).

State is persisted in **MongoDB**, one document per brand. Subsequent
runs only summarize and rank net-new files, reusing the saved base
summary.

## How it works

Two LangGraph state machines, auto-dispatched based on whether the brand
already has a document in Mongo:

### Onboarding (first run for a brand)

```
list_files → identify_base_files (LLM) → build_base_summary (LLM)
                                                ↓
                read_all_files → summarize_each (N parallel LLMs)
                                                ↓
                rank_files (LLM, grounded in base summary) → save_state → END
```

### Update (subsequent runs)

```
load_state → list_files → detect_new_files (no LLM)
                                ↓
                    read_new_files → summarize_each (M parallel LLMs)
                                                ↓
                rank_files (LLM, grounded in saved base) → save_state → END
```

## Setup

Requires Python 3.11+, [uv](https://github.com/astral-sh/uv), and a
running MongoDB (local or Atlas).

```bash
uv sync --extra dev
```

Set the following in `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=openai/gpt-oss-120b
BASE_DIR=/absolute/path/to/this/repo

MONGODB_URI="mongodb+srv://USER:PASSWORD@cluster.xyz.mongodb.net/?retryWrites=true&w=majority"
MONGODB_DATABASE=brand_summarizer
MONGODB_COLLECTION=brand_states
```

For local Mongo, use something like `mongodb://localhost:27017`.

Optional LLM knobs (defaults are sensible):

```
LLM_MAX_TOKENS=8192          # raise if you hit truncated JSON on big folders
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=1
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MONGODB_TIMEOUT_MS=10000
```

## Run — direct script (recommended)

```bash
python dev.py
```

Hard-coded to the `genuin/` folder. Auto-detects mode by looking up
the brand in Mongo.

- **First run** → onboarding. Picks base files, builds base summary,
  summarizes every file, ranks them, writes the document to Mongo.
- **Subsequent runs** → update. Pulls existing state, diffs the folder
  to find new files, summarizes only those, ranks them, appends to the
  saved `ranked_files`.

To force a fresh onboarding for a brand, delete its Mongo document:

```python
from app.state import delete_state
delete_state("genuin")
```

## Run — HTTP service

If you want an API endpoint instead:

```bash
uv run uvicorn app.main:app --reload --port 8001
```

```bash
curl -X POST http://127.0.0.1:8001/summarize \
  -H 'Content-Type: application/json' \
  -d '{"folder": "genuin"}'
```

Response shape:

```json
{
  "folder": "genuin",
  "mode": "onboard | update",
  "base_summary": "Genuin is a ...",
  "base_files": ["begenuin-com.md", ...],
  "all_files": [{"name": "...", "size_bytes": 1234}, ...],
  "ranked_files": [
    {"name": "...", "summary": "...", "reason": "..."},
    ...
  ],
  "newly_added": [
    {"name": "...", "summary": "...", "reason": "..."}
  ],
  "errors": []
}
```

Interactive docs at `http://127.0.0.1:8001/docs`.

## State document (MongoDB)

One document per brand in the `brand_states` collection:

```json
{
  "_id": ObjectId("..."),
  "folder": "genuin",
  "base_summary": "...",
  "base_files": ["...", ...],
  "ranked_files": [
    {"name": "...", "summary": "...", "reason": "..."}
  ],
  "errors": [],
  "created_at": "2026-05-23T...",
  "updated_at": "2026-05-23T..."
}
```

`folder` is uniquely indexed. The collection and database names come
from `.env`.

### Migrating from the old JSON files

If you have a legacy `state/<folder>.json` from a previous run (before
Mongo was wired in), `load_state(folder)` will auto-migrate it the
first time it's called and rename the source to
`state/<folder>.json.migrated` so it doesn't re-trigger. You can delete
the `.migrated` file once you've verified the Mongo document.

## Adding another brand

Drop a folder of markdown files at the project root (e.g. `nike/`),
then either `python dev.py` (after editing the hard-coded `FOLDER`
constant) or `POST /summarize {"folder": "nike"}`. Folder names are
validated against `^[a-zA-Z0-9_-]+$` and must resolve under
`BASE_DIR` (path-traversal safe).

Each brand gets its own document in Mongo, keyed by `folder`.

## Testing

```bash
uv run pytest
```

All tests use mocked LLM clients **and** an in-memory state-store
fixture — they never hit OpenRouter or Mongo. No live calls, no API
spend.

## Project layout

```
app/
├── main.py            # FastAPI app (POST /summarize, GET /health)
├── config.py          # Pydantic Settings (loads .env)
├── schemas.py         # Pydantic + TypedDict models
├── state.py           # MongoDB persistence (load/save/delete)
├── llm.py             # OpenRouter ChatOpenAI client factory
├── prompts.py         # All LLM prompt templates
└── graph/
    ├── builder.py     # build_onboarding_graph(), build_update_graph()
    └── nodes.py       # All node functions
tests/
├── conftest.py        # state_store + fixture_folder fixtures
├── test_nodes.py      # per-node unit tests
├── test_graph.py      # end-to-end graph tests for both modes
└── test_api.py        # FastAPI integration tests for both modes
dev.py                 # run the pipeline directly (no HTTP)
docs/superpowers/specs/
└── 2026-05-23-brand-summarizer-design.md  # original design spec
```
