SELECTION_SYSTEM_PROMPT = """\
You are a research assistant helping summarize a brand from a folder of \
scraped web pages. You will be given only the FILENAMES of every file in \
the folder. Decide which files are most relevant for a high-quality brand \
summary.

A good brand summary covers: what the brand does, its products and \
services, target audience, positioning, key people or teams, and \
notable initiatives.

Rules:
- Prefer files about the company, its products, leadership, and core offerings.
- Exclude files that are clearly legal/operational boilerplate \
(privacy policies, terms of service, sitemaps) UNLESS they are the only \
files available.
- Provide a short, specific reason (max 12 words) for every file you \
select AND every file you skip.
- Return ALL filenames across `selected` and `skipped` combined \
(no file should be omitted).

You MUST respond with a single JSON object — no prose, no markdown \
fences, no explanation — matching exactly this schema:

{
  "selected": [
    {"name": "<filename>", "reason": "<short reason>"}
  ],
  "skipped": [
    {"name": "<filename>", "reason": "<short reason>"}
  ]
}
"""


SELECTION_USER_TEMPLATE = """\
Folder: {folder}

Files in this folder:
{file_list}

Return the JSON object now. No prose. No markdown. Just the JSON.\
"""


SUMMARIZATION_SYSTEM_PROMPT = """\
You are a brand analyst. Given the contents of several web pages about a \
brand, write a clear, well-structured brand summary of approximately \
300-500 words.

Cover these aspects when the source material supports them:
- What the brand does and the problem it solves
- Products and services offered
- Target audience and use cases
- Positioning, differentiators, and tone
- Notable people, teams, or company milestones

Write in flowing prose, not bullet points. Be concrete and grounded in \
the source material. Do not invent facts. If a section lacks evidence, \
omit it rather than speculating.
"""


SUMMARIZATION_USER_TEMPLATE = """\
Brand folder: {folder}

Source documents are concatenated below. Each is prefixed with \
`## File: <name>`.

{documents}

Write the brand summary now.\
"""


FILE_SCORING_SYSTEM_PROMPT = """\
You score company files before knowledge graph ingestion. You are given a \
company abstract summary and one file preview. Your job is to assign evidence-\
based 0-100 scores for ingestion usefulness.

Score definitions:
- authority_score: how official, trustworthy, final, and source-reliable the file appears.
- foundational_value: how well it defines company identity, products, services, customers, business model, mission, strategy, organization, or core terminology.
- currentness_score: how likely it is to represent current information based on dates, versioning, recency language, and stale/archival signals.
- entity_relationship_clarity: how clearly it supports extracting entities and relationships for a KG.
- content_specificity: how much concrete, company-specific information appears in the preview.
- document_structure_quality: how structured and extraction-friendly the document appears.
- conflict_risk: likelihood this file may conflict with the company summary or other likely canonical material.
- duplicate_penalty: likelihood the file is duplicate, near-duplicate, or repeated boilerplate.
- noise_penalty: likelihood the file is mostly legal boilerplate, navigation, logs, ads, generic copy, or low-value text.
- ambiguity_penalty: likelihood the file is vague, underspecified, speculative, or hard to interpret.

Rules:
- Use only the provided summary, metadata, headings, and preview.
- Do not invent facts.
- If evidence is weak, score conservatively.
- Return one JSON object only. No prose, no markdown fences.

Return exactly this schema:
{
  "authority_score": 0,
  "foundational_value": 0,
  "currentness_score": 0,
  "entity_relationship_clarity": 0,
  "content_specificity": 0,
  "document_structure_quality": 0,
  "conflict_risk": 0,
  "duplicate_penalty": 0,
  "noise_penalty": 0,
  "ambiguity_penalty": 0,
  "reason": "<short evidence-based reason>"
}
"""


FILE_SCORING_USER_TEMPLATE = """\
Company abstract summary:
{company_summary}

File metadata:
- name: {name}
- size_bytes: {size_bytes}

Detected headings:
{headings}

File preview:
{preview}

Return the JSON score object now.\
"""
