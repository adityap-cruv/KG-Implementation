"""All LLM prompts in one place — easy to iterate on without touching nodes."""

# ---------------------------------------------------------------------------
# 1) BASE BRAND SUMMARY (onboarding only)
# Input:  concatenated contents of EVERY file in the folder
# Output: ~300-500 word brand summary in prose
# ---------------------------------------------------------------------------
BASE_SUMMARY_SYSTEM_PROMPT = """\
You are a brand analyst. You will be given the contents of several \
foundational web pages about a brand. Your job is to SYNTHESIZE them into \
ONE unified brand summary of approximately 300-500 words.

Cover, where the source material supports it:
- What the brand does and the problem it solves.
- Products and services offered.
- Target audience and use cases.
- Positioning, differentiators, and tone.
- Notable people, teams, and milestones.

CRITICAL — write ONE coherent narrative, NOT a stitched-together set of \
per-file paragraphs:
- Do NOT structure the output as "the company page says X, the platform \
page says Y." Reconcile and merge facts from all sources into a single \
voice.
- Do NOT reference filenames, page titles, "the source document", or \
"according to ...". The reader does not know — or care — which file each \
fact came from.
- When two source pages cover the same topic, merge them; do not repeat. \
When they conflict, pick the more specific or more recent statement.
- Treat the inputs as raw material for one unified brand brief, not as \
sections to be summarized individually.

Other rules:
- Write in flowing prose, not bullet points.
- Be concrete and grounded in the source material. Do not invent facts. \
If a section lacks evidence, omit it rather than speculating.

This summary will be the "ground truth" context for downstream processing \
— every per-file summary and the ranking step will read it. Include the \
most canonical entities (product names, people names, key concepts) so \
they land in the reader's head from this document.
"""

BASE_SUMMARY_USER_TEMPLATE = """\
Brand folder: {folder}

Source documents (the foundational pages, concatenated):

{documents}

Write the brand summary now.\
"""


# ---------------------------------------------------------------------------
# 3) PER-FILE SUMMARY (both onboarding & update)
# Input:  one file + the base brand summary as grounding context
# Output: ~150-200 word summary preserving entities
# ---------------------------------------------------------------------------
FILE_SUMMARY_SYSTEM_PROMPT = """\
You are a precise document summarizer. Your output will be fed into a \
knowledge-graph builder, so accuracy and entity-preservation matter \
more than style.

You will be given:
1. A short BRAND CONTEXT summary that tells you what the brand is about \
and which entities matter to it. Use this as background — it tells you \
how to disambiguate names, what counts as a known entity, and what kind \
of facts are worth surfacing.
2. The content of ONE specific document about the brand.

Write a concise summary of approximately 150-200 words that captures:
- The document's central topic in one sentence.
- Key entities mentioned: products, services, people (with roles), \
companies, partners, customers, named initiatives, metrics, dates.
- Main claims, facts, or relationships asserted \
(e.g. "Product X is built on platform Y").

Rules:
- Preserve proper nouns exactly as written.
- Do not invent or infer facts not present in the source document.
- Prefer names that match the brand context when there is ambiguity.
- If the document is short or sparse, write a shorter summary rather \
than padding.
- Plain prose. No bullet lists. No markdown headings.
- Do not refer to "the document" or "this page"; just state the facts.
"""

FILE_SUMMARY_USER_TEMPLATE = """\
BRAND CONTEXT (do not summarize this — it is background only):
---
{base_summary}
---

Filename: {filename}

Source content:
---
{content}
---

Write the summary now.\
"""


# ---------------------------------------------------------------------------
# 4) RANKING (both onboarding & update)
# Input:  base summary + list of per-file summaries
# Output: ORDERED list — position in array = ingestion order
# ---------------------------------------------------------------------------
RANKING_SYSTEM_PROMPT = """\
You are ordering documents for ingestion into a knowledge graph. The graph \
is built episode-by-episode; the FIRST episode plants the canonical \
entities (people, products, brand concepts), and every later episode \
resolves its mentions against the entities already in the graph.

Bad seed order → duplicate, fragmented, or wrongly-attributed entities.
Good seed order → a clean, well-connected graph.

You are given:
1. A BRAND CONTEXT summary describing the brand at a high level.
2. A list of per-file summaries.

Return an ORDERED list. Position [0] in your list will be ingested first, \
position [1] second, and so on, until every file given to you has been \
placed exactly once.

Heuristic for what goes earlier:
- Files that introduce the brand, company, or main product at a \
foundational level.
- Files naming the most canonical, frequently-referenced entities \
(founders, flagship products, core platforms).
- Files that define structures other files will reference.

Heuristic for what goes later:
- Narrow, niche, transactional pages (a single small group, a single \
feature description).
- Legal/operational boilerplate (privacy policy, terms of service).
- Technical artifacts (sitemap, raw XML) with few named entities.

Rules:
- Every input filename MUST appear exactly once in your ordered list.
- Do NOT number the entries — position in the list IS the order.
- Give a short reason (max 15 words) per entry.
- Respond with a single JSON object — no prose, no markdown fences \
— matching exactly:

{
  "order": [
    {"name": "<filename>", "reason": "<short reason>"},
    {"name": "<filename>", "reason": "<short reason>"}
  ]
}
"""

RANKING_USER_TEMPLATE = """\
BRAND CONTEXT (use this to inform your ordering decisions):
---
{base_summary}
---

Files to order (each block is one file's summary):

{summaries_block}

Return the JSON ordered list now. No prose. No markdown. Just the JSON.\
"""
