import pytest

from app.ranking import calculate_weighted_score, rank_scored_files, validate_weights


def _score(name: str, authority: int, foundational: int) -> dict:
    return {
        "name": name,
        "size_bytes": 100,
        "authority_score": authority,
        "foundational_value": foundational,
        "currentness_score": 60,
        "entity_relationship_clarity": 70,
        "content_specificity": 80,
        "document_structure_quality": 90,
        "conflict_risk": 20,
        "duplicate_penalty": 10,
        "noise_penalty": 0,
        "ambiguity_penalty": 10,
        "reason": "test reason",
    }


def test_calculate_weighted_score_uses_expected_formula():
    scored_file = _score("company.md", 90, 80)

    assert calculate_weighted_score(scored_file) == 75.0


def test_rank_scored_files_sorts_descending_and_assigns_rank():
    ranked_files = rank_scored_files(
        [
            _score("lower.md", 50, 50),
            _score("higher.md", 90, 90),
        ]
    )

    assert [item["name"] for item in ranked_files] == ["higher.md", "lower.md"]
    assert [item["rank"] for item in ranked_files] == [1, 2]
    assert ranked_files[0]["weighted_score"] > ranked_files[1]["weighted_score"]


def test_calculate_weighted_score_accepts_custom_weights():
    scored_file = _score("company.md", 90, 80)
    weights = {
        "authority_score": 1,
        "foundational_value": 0,
        "entity_relationship_clarity": 0,
        "currentness_score": 0,
        "content_specificity": 0,
        "document_structure_quality": 0,
        "risk_penalties": 0,
    }

    assert calculate_weighted_score(scored_file, weights) == 90


def test_validate_weights_rejects_missing_keys():
    with pytest.raises(ValueError):
        validate_weights({"authority_score": 1})
