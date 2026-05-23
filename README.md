# Brand Summarizer

A LangGraph-driven pipeline that summarizes a brand from a folder of
markdown files. Two ways to run it: a one-shot script (`dev.py`) or a
FastAPI HTTP service.

## How it works

The pipeline is a four-node LangGraph state machine:

```
list_files → select_files (LLM) → read_files → summarize (LLM) → END
```

1. **list_files** — scans the requested folder under `BASE_DIR` for `.md` files.
2. **select_files** — sends only the filenames to the LLM, which returns
   JSON choosing the files most relevant for a brand summary (and explaining
   each include/skip).
3. **read_files** — loads the chosen files from disk.
4. **summarize** — sends the concatenated contents to the LLM and gets back
   a brand summary.

The full pipeline state — `all_files`, `selected_files`, `skipped_files`,
`summary`, `errors` — is returned so you can inspect every decision.

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

Optional knobs (defaults are sensible):

```
LLM_MAX_TOKENS=8192          # raise for very large folders
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=1
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Run — direct script (recommended)

The fastest way: run the pipeline in-process and print the summary to
the terminal.

```bash
python dev.py                 # summarize the genuin/ folder
python dev.py nike            # summarize nike/ instead
python dev.py genuin --json   # also write full trace to summary.json
```

Sample output (truncated):

```
========================================================================
Files in folder (20)
========================================================================
  - begenuin-com-company.md (4965 bytes)
  - begenuin-com-genai.md (7774 bytes)
  ...

========================================================================
LLM selected (15)
========================================================================
  + begenuin-com-company.md          Company overview and mission
  + begenuin-com-genai.md            Generative AI product details
  ...

========================================================================
LLM skipped (5)
========================================================================
  - begenuin-com-privacy.md          Legal privacy policy
  - begenuin-com-terms.md            Legal terms of service
  ...

========================================================================
Brand summary
========================================================================
Genuin is a 2021-born technology platform that lets brands rebuild the
social experience on their own terms...
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
  "all_files": [{"name": "...", "size_bytes": 1234}, ...],
  "selected_files": [{"name": "...", "reason": "..."}, ...],
  "skipped_files":  [{"name": "...", "reason": "..."}, ...],
  "summary": "Genuin is a ...",
  "errors": []
}
```

Interactive docs at `http://127.0.0.1:8001/docs`.

## Adding another brand

Drop a new folder of markdown files at the project root (e.g. `nike/`),
then either `python dev.py nike` or `POST /summarize {"folder": "nike"}`.
Folder names are validated against `^[a-zA-Z0-9_-]+$` and must resolve
under `BASE_DIR` (path-traversal safe).

## Testing

```bash
uv run pytest
```

All tests use mocked LLM clients — no live API calls, no API spend.

## Notes on the LLM call

- The selection step uses JSON-object mode (`response_format={"type": "json_object"}`)
  rather than tool/function calling. This is more reliable across providers,
  especially for reasoning models like `gpt-oss-120b` that burn many tokens
  internally before emitting the final answer.
- `LLM_MAX_TOKENS` defaults to 8192 to give the reasoning model headroom.
  If you see truncated-JSON errors on a very large folder, raise this.
- `temperature=0` keeps selection and summary deterministic across runs.

## Project layout

```
app/
├── main.py            # FastAPI app (POST /summarize, GET /health)
├── config.py          # Pydantic Settings (loads .env)
├── schemas.py         # Request/response models + PipelineState
├── llm.py             # OpenRouter ChatOpenAI client factory
├── prompts.py         # Selection + summarization prompt templates
└── graph/
    ├── builder.py     # build_graph(): compiled LangGraph
    └── nodes.py       # list_files, select_files, read_files, summarize
tests/
├── conftest.py        # shared fixtures (env, fixture_folder)
├── test_nodes.py      # per-node unit tests
├── test_graph.py      # end-to-end graph test
└── test_api.py        # FastAPI integration tests
dev.py                 # run the pipeline directly (no HTTP)
docs/superpowers/specs/
└── 2026-05-23-brand-summarizer-design.md  # the design spec
```
