RANKING_WEIGHTS = {
    "authority_score": 0.25,
    "foundational_value": 0.20,
    "entity_relationship_clarity": 0.20,
    "currentness_score": 0.10,
    "content_specificity": 0.10,
    "document_structure_quality": 0.10,
    "risk_penalties": 0.05,
}


def validate_weights(weights: dict) -> dict:
    required_keys = set(RANKING_WEIGHTS)
    missing = required_keys - set(weights)
    extra = set(weights) - required_keys

    if missing:
        raise ValueError(f"Missing ranking weight keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown ranking weight keys: {sorted(extra)}")

    normalized_weights = {key: float(value) for key, value in weights.items()}
    for key, value in normalized_weights.items():
        if value < 0:
            raise ValueError(f"Ranking weight cannot be negative: {key}={value}")

    return normalized_weights


def calculate_weighted_score(scored_file: dict, weights: dict | None = None) -> float:
    active_weights = validate_weights(weights or RANKING_WEIGHTS)
    risk_penalties = (
        scored_file["conflict_risk"]
        + scored_file["duplicate_penalty"]
        + scored_file["noise_penalty"]
        + scored_file["ambiguity_penalty"]
    ) / 4

    weighted_score = (
        scored_file["authority_score"] * active_weights["authority_score"]
        + scored_file["foundational_value"] * active_weights["foundational_value"]
        + scored_file["entity_relationship_clarity"]
        * active_weights["entity_relationship_clarity"]
        + scored_file["currentness_score"] * active_weights["currentness_score"]
        + scored_file["content_specificity"] * active_weights["content_specificity"]
        + scored_file["document_structure_quality"]
        * active_weights["document_structure_quality"]
        - risk_penalties * active_weights["risk_penalties"]
    )

    return round(weighted_score, 2)


def rank_scored_files(scored_files: list[dict], weights: dict | None = None) -> list[dict]:
    ranked_files = [
        {
            **scored_file,
            "weighted_score": calculate_weighted_score(scored_file, weights),
        }
        for scored_file in scored_files
    ]
    ranked_files.sort(key=lambda item: item["weighted_score"], reverse=True)

    for index, ranked_file in enumerate(ranked_files, start=1):
        ranked_file["rank"] = index

    return ranked_files
