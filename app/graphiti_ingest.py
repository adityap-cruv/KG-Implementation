"""Self-hosted graphiti-core ingestion: markdown folder -> local Neo4j KG.

No Zep Cloud, no episode quotas. Uses your OPENAI_API_KEY directly and
writes to a Neo4j running on localhost (Docker — see README).

Each `add_episode` call applies the reusable Pydantic ontology so the LLM
is forced to extract structured attributes (founded_year, role, etc.).

Usage:
    python ingest.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args, get_origin

# Silence the noise BEFORE importing neo4j / graphiti modules.
# - neo4j 5 logs IF-NOT-EXISTS index errors and missing-property warnings
#   that are benign during graphiti's normal ingest flow.
# - graphiti_core logs verbose debug info per episode.
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.CRITICAL)
logging.getLogger("graphiti_core").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import hashlib
import inspect
import json

import frontmatter
from dotenv import load_dotenv
from json_repair import repair_json
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError as GraphitiRateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts.models import Message
import openai as openai_sdk
from pydantic import BaseModel

import argparse
import uuid as _uuid_mod

from app.ontology import ENTITY_TYPES, EDGE_TYPES, HUB_RELATIONS
from app.state import load_state, save_state
from app.config import settings

load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

# Single-provider setup via OpenRouter (chat completions + embeddings,
# proxies all OpenAI models). One key, one base URL — used for both the
# chat LLM and the embedder.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# On OpenRouter, OpenAI models require the "openai/" prefix.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")
INPUT_DIR = Path(os.getenv("INPUT_DIR", "")).expanduser()

CHUNK_CHARS = 15000
CONCURRENCY = int(os.getenv("CONCURRENCY", "1"))

TRACE_DIR = Path(__file__).parent / "trace_out"
MANIFEST_PATH = Path(__file__).parent / "manifest.json"


def _hash_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    """Manifest schema:
    {
        "files": {
            "<slug>": {
                "hash": "<sha256>",
                "url": "<source url>",
                "ingested_utc": "<iso>",
                "episode_uuids": ["<uuid>", ...]
            },
            ...
        }
    }
    """
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return {"files": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))

# Every LLM call appends one dict here. Dumped to trace_out/llm_calls.jsonl.
_LLM_CALLS: list[dict] = []


def _require_openrouter_key() -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY must be set in .env")
    return OPENROUTER_API_KEY


def make_llm_client() -> "LoggingOpenAIClient":
    """Chat LLM via OpenRouter (/v1/chat/completions with json_schema)."""
    return LoggingOpenAIClient(config=LLMConfig(
        api_key=_require_openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        model=LLM_MODEL,
    ))


def make_embedder() -> OpenAIEmbedder:
    """Embeddings via OpenRouter (/v1/embeddings, OpenAI-compatible)."""
    return OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=_require_openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        embedding_model=EMBEDDING_MODEL,
    ))


def make_cross_encoder() -> OpenAIRerankerClient:
    """Cross-encoder reranker via OpenRouter — used during hybrid search to
    re-rank candidate entities/edges. Uses chat.completions just like the
    main LLM client, so OpenRouter-compatible."""
    return OpenAIRerankerClient(config=LLMConfig(
        api_key=_require_openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        model=LLM_MODEL,
    ))


def _tolerate_bare_list(response, response_model):
    """Open-source models (gpt-oss-120b, Llama, Qwen, etc.) sometimes ignore
    the json_schema response format and return a bare JSON array — e.g.
    `[{...}, {...}]` — when graphiti's schema expects an object wrapping
    that array — e.g. `{"extracted_entities": [{...}, {...}]}`.

    When we detect this exact mismatch (response is a list, response_model
    has exactly one list-typed field), wrap the list under that field. This
    rescues otherwise-failing chunks without changing the model."""
    if response_model is None or not isinstance(response, list):
        return response
    list_fields = [
        name for name, field in response_model.model_fields.items()
        if get_origin(field.annotation) is list
    ]
    if len(list_fields) == 1:
        return {list_fields[0]: response}
    return response


def _coerce_to_annotation(val, annotation):
    """Coerce val to fit a Pydantic field annotation, biased toward Neo4j-
    safe primitives. Handles str | None, list[str] | None, list[int],
    and the Optional[T] unwrapping case. gpt-oss-120b often returns lists
    where strings were declared and dicts where lists were declared —
    this flattens them so node.save() doesn't choke on a nested map.

    The Optional-unwrap MUST check the union origin: get_args(list[int])
    returns (int,) and would falsely look like an Optional unwrap if we
    keyed only on `len(non_none) == 1`."""
    import types as _types
    import typing as _typing
    origin = get_origin(annotation)
    is_union = origin is _typing.Union or origin is _types.UnionType
    if is_union:
        union_args = get_args(annotation)
        is_optional = type(None) in union_args
        non_none = [a for a in union_args if a is not type(None)]
        target = non_none[0] if len(non_none) == 1 else annotation
    else:
        is_optional = False
        target = annotation
    target_origin = get_origin(target)

    if val is None:
        # required list[T] with null → [] (gpt-oss-120b emits null for empty)
        if target_origin is list and not is_optional:
            return []
        return None

    if target is str:
        if isinstance(val, str):
            return val
        if isinstance(val, (int, float, bool)):
            return str(val)
        if isinstance(val, list):
            return ", ".join(
                _coerce_to_annotation(v, str) for v in val if v is not None
            )
        if isinstance(val, dict):
            return json.dumps(val, ensure_ascii=False)
        return str(val)

    if target_origin is list:
        elt_args = get_args(target)
        elt_type = elt_args[0] if elt_args else str
        # Empty-string-for-list is the most common drift mode.
        if val == "":
            return []
        if not isinstance(val, list):
            val = [val]
        return [_coerce_to_annotation(v, elt_type) for v in val]

    # Unknown / scalar annotation we don't model — keep Neo4j-safe.
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return val


def _sanitize_to_schema(response, response_model):
    """gpt-oss-120b returns extra attribute keys outside the response
    schema, nested dicts where scalars were declared, or occasionally a
    list-of-dicts where a single dict was expected. Three downstream
    failures result:
      1. Neo4j save crashes — properties must be primitives or arrays of
         primitives, never maps (see graphiti_core/nodes.py:570).
      2. graphiti validates via response_model(**llm_response) and raises
         on schema mismatch (extract_attributes_from_nodes line 811,
         edge dedupe line 721).
      3. `EntityType(**[{...}])` raises TypeError when LLM returns a list.

    Behavior: if response is a list-of-dicts, merge them into one dict
    (last-write-wins). Drop any key not declared on response_model. Then
    coerce remaining values to match their declared annotation."""
    if response_model is None:
        return response
    if isinstance(response, list):
        merged: dict[str, Any] = {}
        for item in response:
            if isinstance(item, dict):
                merged.update(item)
        response = merged
    if not isinstance(response, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for field_name, field_info in response_model.model_fields.items():
        if field_name not in response:
            continue
        cleaned[field_name] = _coerce_to_annotation(
            response[field_name], field_info.annotation
        )
    return cleaned


def _ensure_edge_facts(response):
    """gpt-oss-120b sometimes drops the required `fact` field on every
    edge in an ExtractedEdges payload — Pydantic then rejects the entire
    chunk and graphiti loses all its relationships. Synthesize a minimal
    fact from source/relation/target when missing so the edges survive.
    The synthesized text is plain ("Genuin partners with Mastercard")
    but readable; better than dropping the chunk."""
    if not isinstance(response, dict):
        return response
    edges = response.get("edges")
    if not isinstance(edges, list):
        return response
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if edge.get("fact"):
            continue
        src = edge.get("source_entity_name", "")
        tgt = edge.get("target_entity_name", "")
        rel = (edge.get("relation_type") or "RELATED_TO").replace("_", " ").lower()
        synthesized = f"{src} {rel} {tgt}".strip()
        edge["fact"] = synthesized or "relationship"
    return response


class LoggingOpenAIClient(OpenAIGenericClient):
    """OpenAI-compatible client (chat.completions + json_schema response
    format) with logging. Inherits from OpenAIGenericClient — NOT OpenAIClient —
    so it works against any OpenAI-compatible endpoint (OpenRouter, vLLM,
    ollama, etc.). OpenAIClient is OpenAI-only because it uses /v1/responses.

    Captures every (messages, response) pair to _LLM_CALLS so we can dump
    the REAL prompts and responses graphiti-core issued during ingest.

    Three rescues for gpt-oss-120b's loose adherence to json_schema:
      1. _tolerate_bare_list  — wraps bare JSON arrays into the schema's
         wrapper field when the model skips the wrapping object.
      2. _sanitize_node_attributes — drops hallucinated attribute keys and
         flattens nested dicts so Neo4j accepts the values.
      3. _ensure_edge_facts — synthesizes the required `fact` field on
         edges when the model omits it.
    Plus a json_repair fallback in _generate_response for the model's
    occasional malformed JSON (unterminated strings, missing commas)."""

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens=DEFAULT_MAX_TOKENS,
                                 model_size=ModelSize.medium):
        """Mirrors OpenAIGenericClient._generate_response but falls back to
        json_repair when the LLM emits malformed JSON. The parent raises
        json.JSONDecodeError on every parse failure, which forces graphiti
        to retry the entire (expensive) LLM call up to MAX_RETRIES times
        and then drop the chunk. Most parse failures are trivial truncations
        json_repair fixes locally — no extra API spend."""
        openai_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ("user", "system"):
                openai_messages.append({"role": m.role, "content": m.content})
        try:
            response_format: dict[str, Any] = {"type": "json_object"}
            if response_model is not None:
                json_schema = response_model.model_json_schema()
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": getattr(response_model, "__name__", "structured_response"),
                        "schema": json_schema,
                    },
                }
            api_response = await self.client.chat.completions.create(
                model=self.model or "gpt-4.1-mini",
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )
            result = api_response.choices[0].message.content or ""
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError as parse_err:
                repaired = repair_json(result, return_objects=True)
                # json_repair returns '' when it can't recover anything
                if repaired == "" or repaired is None:
                    raise parse_err
                parsed = repaired
            # gpt-oss-120b sometimes emits a bare scalar (single float,
            # string, bool) instead of a JSON object — graphiti then does
            # `ResponseModel(**parsed)` and raises an opaque TypeError.
            # Raise here so the parent's retry loop catches it and asks
            # the LLM to try again with an error-context message.
            if response_model is not None and not isinstance(parsed, (dict, list)):
                raise ValueError(
                    f"Expected JSON object/array matching "
                    f"{response_model.__name__}, got {type(parsed).__name__}: "
                    f"{parsed!r}"
                )
            return parsed
        except openai_sdk.RateLimitError as exc:
            raise GraphitiRateLimitError from exc
        except Exception as exc:
            logging.getLogger("graphiti_core.llm_client").error(
                f"Error in generating LLM response: {exc}"
            )
            raise

    async def generate_response(self, messages, response_model=None,
                                max_tokens=None, model_size=None,
                                group_id=None, prompt_name=None,
                                **extra_kwargs):
        started = time.time()
        # The parent will mutate messages (appends schema, language hints).
        # Snapshot BEFORE the call so we record what graphiti-core decided
        # before mutation; we'll also snapshot after to show the final prompt.
        before_snapshot = [
            {"role": getattr(m, "role", None), "content": getattr(m, "content", "")}
            for m in messages
        ]
        # Only forward kwargs that the caller actually set, so the parent's
        # own defaults (e.g. ModelSize.medium) kick in when graphiti-core
        # calls us with no value. Forwarding None overrides those defaults
        # and breaks `model_size.value` access downstream.
        passthrough_kwargs: dict = {}
        if max_tokens is not None:
            passthrough_kwargs["max_tokens"] = max_tokens
        if model_size is not None:
            passthrough_kwargs["model_size"] = model_size
        if group_id is not None:
            passthrough_kwargs["group_id"] = group_id
        if prompt_name is not None:
            passthrough_kwargs["prompt_name"] = prompt_name
        # Forward forward-compat kwargs graphiti-core adds in newer versions
        # (e.g. attribute_extraction). Without this, signature mismatches
        # raise TypeError on every chunk.
        passthrough_kwargs.update(extra_kwargs)
        try:
            response = await super().generate_response(
                messages=messages, response_model=response_model,
                **passthrough_kwargs,
            )
            # Auto-rescue bare-list responses from open-source models
            # before they reach graphiti's strict Pydantic unpacking.
            response = _tolerate_bare_list(response, response_model)
            # Per-prompt rescues for gpt-oss-120b's schema drift.
            if prompt_name in (
                "extract_nodes.extract_attributes",
                "extract_edges.extract_attributes",
                "dedupe_edges.resolve_edge",
            ):
                response = _sanitize_to_schema(response, response_model)
            elif prompt_name == "extract_edges.edge":
                response = _ensure_edge_facts(response)
            error = None
        except Exception as exc:
            response = None
            error = repr(exc)
            raise
        finally:
            after_snapshot = [
                {"role": getattr(m, "role", None), "content": getattr(m, "content", "")}
                for m in messages
            ]
            schema = None
            if response_model is not None and issubclass(response_model, BaseModel):
                try:
                    schema = response_model.model_json_schema()
                except Exception:
                    schema = None
            _LLM_CALLS.append({
                "call_index": len(_LLM_CALLS) + 1,
                "prompt_name": prompt_name,
                "model_size": getattr(model_size, "value", str(model_size)),
                "model": getattr(self, "model", None),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.time() - started,
                "request_messages_before_mutation": before_snapshot,
                "request_messages_final_sent_to_llm": after_snapshot,
                "response_model": response_model.__name__ if response_model else None,
                "response_model_schema": schema,
                "response": response,
                "error": error,
            })
        return response


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= size:
                buf = para
            else:
                for i in range(0, len(para), size):
                    chunks.append(para[i:i + size])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def load_markdown_files(input_dir: Path) -> list[tuple[str, str, str]]:
    if not input_dir.exists():
        print(f"ERROR: INPUT_DIR does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    items = []
    for md_path in sorted(input_dir.glob("*.md")):
        if md_path.name == "INDEX.md":
            continue
        post = frontmatter.load(md_path)
        url = post.metadata.get("url", str(md_path))
        body = post.content.strip()
        if not body:
            continue
        items.append((md_path.stem, url, body))
    return items


async def ingest_one(graphiti: Graphiti, sem: asyncio.Semaphore,
                     slug: str, url: str, chunk_idx: int,
                     chunk: str) -> tuple[bool, str | None]:
    """Add a single chunk as an episode. Returns (success, episode_uuid)."""
    async with sem:
        try:
            result = await graphiti.add_episode(
                name=f"{slug}#{chunk_idx}",
                episode_body=f"Source URL: {url}\n\n{chunk}",
                source=EpisodeType.text,
                source_description=url,
                reference_time=datetime.now(timezone.utc),
                entity_types=ENTITY_TYPES,
            )
            episode_uuid = getattr(getattr(result, "episode", None), "uuid", None)
            return True, episode_uuid
        except Exception as exc:
            print(f"    ! {slug}#{chunk_idx} failed: {exc}")
            return False, None


async def main() -> int:
    """Read ranked_files from MongoDB, ingest each markdown file in ranked
    order, then build the 2-layer brand scaffold (Brand + 7 hubs)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", default="genuin_1",
        help="Brand folder under BASE_DIR (must match the folder ranked in MongoDB)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest all ranked files, ignoring previously-recorded progress",
    )
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: set OPENROUTER_API_KEY in .env", file=sys.stderr)
        return 1

    state = load_state(args.folder)
    if not state:
        print(f"ERROR: no state for folder '{args.folder}' in MongoDB. "
              f"Run dev.py first to produce the ranking.", file=sys.stderr)
        return 1
    ranked = state.get("ranked_files", [])
    if not ranked:
        print("ERROR: state has no ranked_files. Run dev.py to populate.",
              file=sys.stderr)
        return 1
    folder_path = settings.BASE_DIR / args.folder
    if not folder_path.exists():
        print(f"ERROR: folder not found: {folder_path}", file=sys.stderr)
        return 1

    ingested = {} if args.force else state.get("ingested_files", {})
    to_ingest = [
        e for e in ranked
        if args.force or not ingested.get(e["name"], {}).get("complete")
    ]

    print(f"Folder:        {args.folder}")
    print(f"Source path:   {folder_path}")
    print(f"Neo4j:         {NEO4J_URI}")
    print(f"Concurrency:   {CONCURRENCY}")
    print(f"Ranked files:  {len(ranked)}")
    print(f"Already done:  {sum(1 for e in ranked if ingested.get(e['name'], {}).get('complete'))}")
    print(f"To ingest now: {len(to_ingest)}\n")

    graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                        llm_client=make_llm_client(),
                        embedder=make_embedder(),
                        cross_encoder=make_cross_encoder())
    try:
        await graphiti.build_indices_and_constraints()
    except Exception as exc:
        if "AlreadyExists" not in str(exc):
            raise

    started = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)

    # Brand-wide context produced by the LangGraph ranking step. Injected
    # into every chunk's source_description so the extraction LLM has
    # consistent global context for entity disambiguation.
    base_summary = (state.get("base_summary") or "").strip()
    base_context = (
        f"Brand context: {base_summary}\n\n" if base_summary else ""
    )

    for i, entry in enumerate(to_ingest, 1):
        fname = entry["name"]
        fpath = folder_path / fname
        if not fpath.exists():
            print(f"  [{i}/{len(to_ingest)}] ! missing on disk: {fname}")
            continue
        post = frontmatter.load(str(fpath))
        text = post.content.strip()
        if not text:
            print(f"  [{i}/{len(to_ingest)}] ! empty: {fname}")
            continue
        chunks = chunk_text(text)
        print(f"  [{i}/{len(to_ingest)}] {fname} — {len(chunks)} chunk(s)")

        source_desc = (
            f"{base_context}"
            f"This file's role: {entry.get('reason', 'no reason')}. "
            f"Source file: {fname}."
        )
        episode_uuids: list[str] = []
        file_started = time.time()
        for ci, chunk in enumerate(chunks, 1):
            async with sem:
                t0 = time.time()
                try:
                    r = await graphiti.add_episode(
                        name=f"{Path(fname).stem}#{ci}",
                        episode_body=chunk,
                        source=EpisodeType.text,
                        source_description=source_desc,
                        reference_time=datetime.now(timezone.utc),
                        entity_types=ENTITY_TYPES,
                        edge_types=EDGE_TYPES,
                    )
                    uid = getattr(getattr(r, "episode", None), "uuid", None)
                    if uid:
                        episode_uuids.append(uid)
                    print(f"      ✓ chunk {ci}/{len(chunks)}  ({time.time()-t0:.1f}s)")
                except Exception as exc:
                    print(f"      ! chunk {ci}/{len(chunks)} failed: {exc}")

        ingested[fname] = {
            "episode_uuids": episode_uuids,
            "complete": len(episode_uuids) == len(chunks),
            "ingested_utc": datetime.now(timezone.utc).isoformat(),
        }
        state["ingested_files"] = ingested
        save_state(args.folder, state)
        print(f"      ★ {fname} done ({len(episode_uuids)}/{len(chunks)} chunks, "
              f"{time.time()-file_started:.0f}s)")

    if to_ingest:
        print(f"\nIngestion done in {time.time()-started:.0f}s "
              f"({len(_LLM_CALLS)} LLM calls captured)")

    print("\nBuilding 2-layer brand scaffold...")
    all_episode_uuids = [
        u for f in ingested.values() for u in f.get("episode_uuids", [])
    ]
    await _build_brand_scaffold(graphiti.driver, all_episode_uuids)

    await graphiti.close()
    print(f"\nDone. View at http://localhost:7475 (login: {NEO4J_USER}/{NEO4J_PASSWORD})")
    return 0


async def _build_brand_scaffold(driver, episode_uuids: list[str]) -> None:
    """Auto-detect primary brand from this run's episodes, build the
    per-brand 2-layer scaffold (Brand + 7 hubs + Layer-1 edges), attach
    every extracted entity to its type's hub via :INCLUDES, and strip
    direct Brand↔entity edges so Brand has exactly 7 edges."""
    if not episode_uuids:
        print("  (no episodes — skipping scaffold)")
        return

    recs, _, _ = await driver.execute_query(
        "MATCH (ep:Episodic)-[:MENTIONS]->(b:Brand) "
        "WHERE ep.uuid IN $eps "
        "RETURN b.uuid AS u, b.name AS n, count(DISTINCT ep) AS m "
        "ORDER BY m DESC LIMIT 1",
        eps=episode_uuids,
    )
    if not recs:
        print("  (no :Brand entities found — skipping scaffold)")
        return

    detected_uuid, brand_name = recs[0]["u"], recs[0]["n"]
    hub_recs, _, _ = await driver.execute_query(
        "MATCH (b:Brand:Hub {name:$n}) RETURN b.uuid AS u LIMIT 1",
        n=brand_name,
    )
    if hub_recs:
        brand_uuid = hub_recs[0]["u"]
        if brand_uuid != detected_uuid:
            await driver.execute_query(
                "MATCH (dup:Brand) WHERE NOT dup:Hub AND toLower(dup.name)=toLower($n) "
                "DETACH DELETE dup",
                n=brand_name,
            )
    else:
        brand_uuid = detected_uuid
        await driver.execute_query(
            "MATCH (b:Brand {uuid:$u}) SET b:Hub, b.is_primary_brand=true",
            u=brand_uuid,
        )
    print(f"  Primary brand: {brand_name}")

    for label, hub_name, rel in HUB_RELATIONS:
        await driver.execute_query(
            f"MATCH (b:Brand:Hub {{uuid:$bu}}) "
            f"MERGE (h:Hub:{label} {{brand_uuid:$bu}}) "
            f"ON CREATE SET h.uuid=$hu, h.name=$hn, "
            f"              h.is_hub=true, h.created_at=datetime() "
            f"MERGE (b)-[:{rel}]->(h) "
            f"WITH h "
            f"MATCH (ep:Episodic) WHERE ep.uuid IN $eps "
            f"MATCH (ep)-[:MENTIONS]->(e:{label}) WHERE NOT e:Hub "
            f"MERGE (h)-[:INCLUDES]->(e)",
            bu=brand_uuid,
            hn=f"{brand_name} {hub_name}",
            hu=str(_uuid_mod.uuid4()),
            eps=episode_uuids,
        )

    await driver.execute_query(
        "MATCH (h:Hub:Partnerships {brand_uuid:$u}) "
        "MATCH (ep:Episodic) WHERE ep.uuid IN $eps "
        "MATCH (ep)-[:MENTIONS]->(b:Brand) WHERE NOT b:Hub AND b.uuid<>$u "
        "MERGE (h)-[:INCLUDES]->(b)",
        u=brand_uuid, eps=episode_uuids,
    )
    await driver.execute_query(
        "MATCH (b:Brand:Hub {uuid:$u})-[r]->(e) "
        "WHERE NOT e:Hub AND NOT e:Episodic DELETE r",
        u=brand_uuid,
    )
    await driver.execute_query(
        "MATCH (s)-[r]->(b:Brand:Hub {uuid:$u}) "
        "WHERE NOT s:Hub AND NOT s:Episodic DELETE r",
        u=brand_uuid,
    )
    print(f"  ✓ scaffold built — '{brand_name}' + 7 hubs, "
          "entities attached, Brand isolated to 7 hub edges")


async def write_trace_artifacts(graphiti: Graphiti,
                                docs: list[tuple[str, str, str]]) -> None:
    """Dump the full LLM trace for this ingest: every input file, every
    actual graphiti-core prompt template, every node and edge in Neo4j,
    and every LLM call we captured. NO truncation."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    # input.md — every file's full content + chunk plan
    parts = [
        f"INPUT_DIR: {INPUT_DIR}",
        f"files_loaded: {len(docs)}",
        f"total_chars: {sum(len(b) for _, _, b in docs)}",
        f"chunk_size_chars: {CHUNK_CHARS}",
        "",
    ]
    for slug, url, body in docs:
        chunks = chunk_text(body)
        parts += [
            "=" * 78,
            f"slug: {slug}",
            f"url: {url}",
            f"body_chars: {len(body)}",
            f"chunks_planned: {len(chunks)}",
            "-" * 78,
            body,
            "",
        ]
        for i, ch in enumerate(chunks, 1):
            parts += [f"--- chunk {i}/{len(chunks)} ---", ch, ""]
    (TRACE_DIR / "input.md").write_text("\n".join(parts))

    # prompt.txt — dump the actual graphiti-core prompt module source files.
    # These contain every template graphiti-core uses to build the prompt
    # messages it actually sends to the LLM. The runtime-mutated final
    # messages are in llm_calls.jsonl (more useful for "what the LLM saw").
    import graphiti_core.prompts as gp_prompts
    prompts_dir = Path(inspect.getfile(gp_prompts)).parent
    prompt_chunks = []
    for pf in sorted(prompts_dir.glob("*.py")):
        if pf.name in ("__init__.py", "lib.py", "models.py",
                       "prompt_helpers.py", "snippets.py"):
            continue
        prompt_chunks.append(f"===== FILE: graphiti_core/prompts/{pf.name} =====")
        prompt_chunks.append(pf.read_text())
        prompt_chunks.append("")
    (TRACE_DIR / "prompt.txt").write_text("\n".join(prompt_chunks))

    # output.md — query Neo4j for every node and every edge, no filtering
    nodes_query = (
        "MATCH (n) "
        "RETURN labels(n) AS labels, properties(n) AS props"
    )
    edges_query = (
        "MATCH (a)-[r]->(b) "
        "RETURN type(r) AS type, properties(r) AS props, "
        "       properties(a) AS source, properties(b) AS target"
    )
    nodes_out, edges_out = [], []
    try:
        records, _, _ = await graphiti.driver.execute_query(nodes_query)
        for rec in records:
            nodes_out.append({"labels": rec["labels"], "properties": rec["props"]})
    except Exception as e:
        nodes_out = [{"error": repr(e)}]
    try:
        records, _, _ = await graphiti.driver.execute_query(edges_query)
        for rec in records:
            edges_out.append({
                "type": rec["type"],
                "properties": rec["props"],
                "source": rec["source"],
                "target": rec["target"],
            })
    except Exception as e:
        edges_out = [{"error": repr(e)}]

    out_payload = {
        "neo4j_uri": NEO4J_URI,
        "total_nodes": len(nodes_out),
        "total_edges": len(edges_out),
        "nodes": nodes_out,
        "edges": edges_out,
    }
    (TRACE_DIR / "output.md").write_text(
        "# Complete LLM output — every node and edge graphiti-core wrote to Neo4j\n\n"
        "```json\n"
        + json.dumps(out_payload, indent=2, ensure_ascii=False, default=str)
        + "\n```\n"
    )

    # llm_calls.jsonl — every actual LLM call captured by LoggingOpenAIClient
    with (TRACE_DIR / "llm_calls.jsonl").open("w") as fh:
        for call in _LLM_CALLS:
            fh.write(json.dumps(call, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
