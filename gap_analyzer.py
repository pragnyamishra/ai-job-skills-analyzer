"""
Gap Analyzer
Compares market-demanded skills against resume skills using both
exact matching and semantic similarity via the vector store.
"""

from monitoring import logger

SIMILARITY_THRESHOLD = 0.82


def compute_gaps(
    market_skills: list[dict],
    resume_skills: list[dict],
    vector_store=None,
) -> dict:
    """Compare market skills against resume skills.

    Uses exact string match first, then falls back to vector similarity
    when a vector_store is provided.

    Returns a dict with matched_skills, skill_gaps, match_score, etc.
    """

    resume_names_lower = {s.get("skill", "").lower().strip() for s in resume_skills}

    matched = []
    gaps = []

    for ms in market_skills:
        skill_name = ms.get("skill", "").strip()
        if not skill_name:
            continue

        # 1) Exact / case-insensitive match
        if skill_name.lower() in resume_names_lower:
            matched.append(ms)
            continue

        # 2) Semantic match via Pinecone
        if vector_store is not None:
            try:
                similar = vector_store.find_similar(
                    skill_name, top_k=3, source_filter="resume"
                )
                found = False
                for hit in similar:
                    if hit["score"] >= SIMILARITY_THRESHOLD:
                        matched.append(ms)
                        found = True
                        break
                if found:
                    continue
            except Exception as e:
                logger.warning(f"Vector similarity lookup failed for '{skill_name}': {e}")

        # 3) Not matched -> gap
        gaps.append(ms)

    # Sort gaps: High importance first, then by job_count descending
    priority_map = {"High": 3, "Medium": 2, "Low": 1}
    gaps.sort(
        key=lambda x: (
            priority_map.get(x.get("importance", "Low"), 0),
            x.get("job_count", 0),
        ),
        reverse=True,
    )

    total = max(len(market_skills), 1)
    score = round(len(matched) / total * 100, 1)

    logger.info(f"Gap analysis: {len(matched)} matched, {len(gaps)} gaps, score={score}%")

    return {
        "matched_skills": matched,
        "skill_gaps": gaps,
        "match_score": score,
        "total_market_skills": len(market_skills),
        "total_matched": len(matched),
        "total_gaps": len(gaps),
    }
