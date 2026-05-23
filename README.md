# Brand Summarizer

LangGraph-driven FastAPI service that summarizes a brand from a folder of
markdown files.

## How it works

The pipeline is a four-node LangGraph state machine:

```
list_files → select_files (LLM) → read_files → summarize (LLM) → END
```

1. **list_files** — scans the requested folder under `BASE_DIR` for `.md` files.
2. **select_files** — sends only the filenames to the LLM, which returns
   structured JSON choosing the files most relevant for a brand summary
   (and explaining each include/skip).
3. **read_files** — loads the chosen files from disk.
4. **summarize** — sends the concatenated contents to the LLM and gets back
   a brand summary.

The full pipeline state — `all_files`, `selected_files`, `skipped_files`,
`summary`, `errors` — is returned in the response so you can inspect every
decision.

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv) (or pip).

```bash
uv sync --extra dev
```

Set the following in `.env` (the project ships with a working example):

```
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=openai/gpt-oss-120b
BASE_DIR=/absolute/path/to/this/repo
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST http://localhost:8000/summarize \
  -H 'Content-Type: application/json' \
  -d '{"folder": "genuin"}'
```

## Response shape

```json
{
  "folder": "genuin",
  "all_files": [{"name": "...", "size_bytes": 1234}, ...],
  "selected_files": [{"name": "...", "reason": "..."}, ...],
  "skipped_files":  [{"name": "...", "reason": "..."}, ...],
  "summary": "Genuin is a creator-first video platform that ...",
  "errors": []
}
```

## Adding another brand

Drop a new folder of markdown files at the project root (e.g. `nike/`),
then `POST /summarize {"folder": "nike"}`. Folder names are validated
against `^[a-zA-Z0-9_-]+$` and must resolve under `BASE_DIR`.

## Testing

```bash
uv run pytest
```

All tests use mocked LLM clients — no live API calls.

## Project layout

```
app/
├── main.py            # FastAPI app
├── config.py          # Pydantic Settings (loads .env)
├── schemas.py         # Pydantic + TypedDict models
├── llm.py             # OpenRouter ChatOpenAI client factory
├── prompts.py         # Selection + summarization prompts
└── graph/
    ├── builder.py     # build_graph(): compiled LangGraph
    └── nodes.py       # 4 node functions
tests/
├── test_nodes.py      # per-node unit tests
├── test_graph.py      # end-to-end graph test
└── test_api.py        # FastAPI integration tests
```
