"""Brand-KG ontology for graphiti self-hosted.

Aligned with brand_kg_schema.txt (project root). Every extracted entity is
classified as exactly ONE of the 7 brand-schema types:

    Brand, Product, People, Business, Marketing, Audience, Partnerships

Engineering is intentionally absent — engineering-dominant docs (API specs,
ad-prioritization algorithms, telemetry frameworks) are routed to SKIP by
the triage layer until the Engineering bucket is refined.

This file is what drives the actual labels in Neo4j. brand_kg_schema.txt is
documentation; this file is the executable schema graphiti uses at
extraction time.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field


class Brand(BaseModel):
    """The brand itself — the root identity node. Usually 1-2 per graph
    (the primary brand, and maybe a parent/sub-brand)."""

    founded_year: str | None = Field(
        default=None,
        description="Year the brand was founded or established "
                    "(extract from phrases like 'founded in YEAR', 'established YEAR').",
    )
    headquarters: str | None = Field(
        default=None,
        description="Headquarters city / region / country.",
    )
    mission: str | None = Field(
        default=None,
        description="Stated mission or purpose of the brand.",
    )
    tagline: str | None = Field(
        default=None,
        description="Brand tagline or slogan, if explicitly stated.",
    )
    industry: str | None = Field(
        default=None,
        description="Industry or sector (e.g. 'video tech', 'fintech', 'ad-tech').",
    )


class Product(BaseModel):
    """Products, services, SKUs, features, platforms THAT THE PRIMARY BRAND
    ITSELF OFFERS, CREATES, BUILDS, OR SELLS — i.e. the brand's own
    externally-facing offerings.

    GOOD examples (Product):
      - "Octo Canvas"      — Genuin's creator workflow product
      - "Genuin Monetize"  — Genuin's ad infrastructure product
      - "Genuin SDK"       — Genuin's developer toolkit
      - "Genuin Embeds"    — Genuin's embeddable widget

    BAD examples (NOT Product — use Engineering for these):
      - "Amazon S3"        — a third-party service Genuin USES, not offers
      - "PostgreSQL"       — a database Genuin runs on, not a Genuin product
      - "Kafka"            — infrastructure Genuin uses
      - "Stripe"           — a payment vendor Genuin integrates with

    Rule of thumb: if Genuin's marketing page lists it as something
    customers buy/use FROM Genuin → Product. If it's something Genuin's
    engineers PICKED from outside → Engineering."""

    category: str | None = Field(
        default=None,
        description="Product category — e.g. 'platform', 'SDK', 'ad format', "
                    "'API', 'mobile app', 'enterprise tool'.",
    )
    launched_year: str | None = Field(
        default=None,
        description="Year the product was launched or released.",
    )
    pricing: str | None = Field(
        default=None,
        description="Pricing model — e.g. 'free', 'subscription', 'usage-based', "
                    "'enterprise contract'.",
    )
    availability: str | None = Field(
        default=None,
        description="Where/how available — e.g. 'iOS+web', 'enterprise only', 'beta'.",
    )


class People(BaseModel):
    """Named individuals — founders, leadership, employees, advisors, board
    members, public figures, spokespeople."""

    role: str | None = Field(
        default=None,
        description="Exact job title or role as written in the text "
                    "(e.g. 'CEO', 'Senior VP - Global', 'Head of Engineering').",
    )
    seniority_tier: str | None = Field(
        default=None,
        description=(
            "Seniority tier inferred from the role. MUST be exactly one of:\n"
            "  - 'FOUNDER'  : Founder, Co-Founder, Founding Partner\n"
            "  - 'C_SUITE'  : CEO, CFO, CTO, CMO, COO, CIO, CHRO, President, Chairman\n"
            "  - 'SVP'      : SVP, EVP, Executive Vice President\n"
            "  - 'VP'       : VP, Vice President, Chief of Staff\n"
            "  - 'DIRECTOR' : Director, Head of <function>, Group Director\n"
            "  - 'MANAGER'  : Manager, Senior Manager, Lead, Principal\n"
            "  - 'IC'       : Engineer, Analyst, Designer, Associate (individual contributor)\n"
            "  - 'ADVISOR'  : Advisor, Board Member, Independent Director\n"
            "  - 'OTHER'    : Role does not fit any of the above"
        ),
    )
    affiliation: str | None = Field(
        default=None,
        description="Primary organization the person is affiliated with. "
                    "Reporting lines in the People hierarchy only span "
                    "individuals sharing the same affiliation.",
    )


class Business(BaseModel):
    """Business / organizational / financial entities — legal entities,
    funding rounds, RACI matrices, business units, financials, M&A history,
    compliance items, internal operations."""

    business_type: str | None = Field(
        default=None,
        description="Specific kind — one of: 'funding_round', 'business_unit', "
                    "'RACI', 'legal_entity', 'financial_metric', 'compliance', "
                    "'M&A', 'operations', 'org_structure'.",
    )
    year: str | None = Field(
        default=None,
        description="Year associated with this business entity (e.g. founding "
                    "year of unit, funding round year, M&A close year).",
    )


class Marketing(BaseModel):
    """Outbound brand communication — campaigns, ads, social posts, press
    releases, blog/content, brand voice, channels, sponsorships-as-campaigns,
    events used for promotion."""

    channel: str | None = Field(
        default=None,
        description="Channel — e.g. 'TikTok', 'Instagram', 'website', "
                    "'in-network', 'YouTube', 'podcast', 'email', 'OOH', 'TV'.",
    )
    campaign_year: str | None = Field(
        default=None,
        description="Year the campaign ran, if mentioned.",
    )


class Audience(BaseModel):
    """The people on the receiving end — customer segments, personas,
    demographics, reviews, sentiment, competitors, market positioning,
    industry context, press coverage."""

    segment_type: str | None = Field(
        default=None,
        description="Segment kind — one of: 'demographic', 'persona', "
                    "'competitor', 'market_segment', 'industry_context', "
                    "'press_coverage', 'review'.",
    )


class Engineering(BaseModel):
    """Technical / engineering entities. Includes BOTH the brand's own
    technical work AND third-party tools the brand uses.

    GOOD examples (Engineering):
      - "Amazon S3"        — third-party storage Genuin USES (not Genuin's product)
      - "PostgreSQL"       — third-party database Genuin runs on
      - "Kafka"            — third-party streaming infra Genuin uses
      - "Stripe API"       — third-party payment integration
      - "Genuin video pipeline"     — Genuin's own internal infrastructure
      - "Genuin Adaptive Intelligence"  — Genuin's internal R&D project
      - "Embed.js"         — internal codebase/library

    BAD examples (NOT Engineering — use Product instead):
      - "Octo Canvas"      — Genuin's externally-sold product
      - "Genuin SDK"       — Genuin's offering, sold to customers

    Rule of thumb: third-party tools/services the brand consumes → here.
    The brand's own customer-facing offering → Product."""

    component_type: str | None = Field(
        default=None,
        description="Kind of engineering component — one of: 'tech_stack', "
                    "'infrastructure', 'API', 'database', 'platform', "
                    "'codebase', 'patent', 'security', 'roadmap', 'tool'.",
    )


class Partnerships(BaseModel):
    """External partner relationships — tech integrations, sponsorships,
    co-brands, suppliers, distributors, resellers, agencies, M&A
    counter-parties, joint ventures, affiliate networks."""

    partner_type: str | None = Field(
        default=None,
        description="Kind of partnership — one of: 'tech_integration', "
                    "'sponsorship', 'co_brand', 'supplier', 'distributor', "
                    "'reseller', 'agency', 'M&A', 'joint_venture', 'affiliate'.",
    )
    partnership_year: str | None = Field(
        default=None,
        description="Year the partnership started or was announced.",
    )


# Pass this dict to graphiti.add_episode(..., entity_types=ENTITY_TYPES).
# Order matters slightly — graphiti's extraction may prefer earlier types
# when uncertain. Brand is first because it's the most specific (only the
# primary brand should land here).
ENTITY_TYPES = {
    "Brand":         Brand,
    "Product":       Product,
    "People":        People,
    "Business":      Business,
    "Engineering":   Engineering,
    "Marketing":     Marketing,
    "Audience":      Audience,
    "Partnerships":  Partnerships,
}


# =============================================================================
# EDGE_TYPES — parsed from brand_kg_schema.txt at import time
# =============================================================================
# brand_kg_schema.txt is the SINGLE SOURCE OF TRUTH for the relation
# vocabulary. We read it once at import and extract every `[:RELATION]`
# pattern. The resulting EDGE_TYPES dict is passed to
# graphiti.add_episode(edge_types=EDGE_TYPES), which constrains the
# extraction LLM to this vocabulary instead of inventing names like
# RELATES_TO. Engineering-related relations are skipped.
#
# To add or remove a relation: edit brand_kg_schema.txt. Re-running
# ingest picks up the changes automatically — no code edits needed.

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "brand_kg_schema.txt"
)

_REL_PATTERN = re.compile(
    r'\(\s*([A-Za-z]+)\s*\)\s*-\[\s*:\s*([A-Z][A-Z_]*)\s*\]->\s*\(\s*([A-Za-z]+)\s*\)'
)


def _parse_edge_types_from_schema() -> dict[str, type[BaseModel]]:
    """Read brand_kg_schema.txt, extract `(Src) -[:REL]-> (Tgt)` patterns,
    build one empty Pydantic class per unique relation name. The
    docstring includes the source→target examples so the LLM has
    context. Lines mentioning Engineering are skipped because Engineering
    is not an active node type."""
    if not _SCHEMA_PATH.exists():
        return {}

    # relation_name -> set of (src_type, tgt_type) pairs from the schema
    relations: dict[str, set[tuple[str, str]]] = {}
    for line in _SCHEMA_PATH.read_text().splitlines():
        for src, rel, tgt in _REL_PATTERN.findall(line):
            relations.setdefault(rel, set()).add((src, tgt))

    edge_types: dict[str, type[BaseModel]] = {}
    for name in sorted(relations):
        pairs = sorted(relations[name])
        examples = ", ".join(f"({s})→({t})" for s, t in pairs[:4])
        doc = (
            f"Relationship {name}. From brand_kg_schema.txt; "
            f"valid source→target signatures: {examples}."
        )
        cls = type(name, (BaseModel,), {"__doc__": doc})
        edge_types[name] = cls
    return edge_types


EDGE_TYPES = _parse_edge_types_from_schema()


# Map from entity-type label to (hub_relation_from_brand, hub_display_name).
# Drives both Layer-1 edges (Brand -> Hub) and the INCLUDES attachments
# from each hub to its specific entities.
HUB_RELATIONS: list[tuple[str, str, str]] = [
    # (hub label,   hub display name,  Layer-1 relation from Brand)
    ("Product",      "Products",        "OFFERS"),
    ("People",       "People",          "HAS_PERSON"),
    ("Business",     "Business",        "OPERATES_AS"),
    ("Engineering",  "Engineering",     "BUILT_WITH"),
    ("Marketing",    "Marketing",       "COMMUNICATES_VIA"),
    ("Audience",     "Audience",        "SERVES"),
    ("Partnerships", "Partnerships",    "HAS_PARTNERSHIP"),
]
