"""
Vector Store
Manages skill embeddings in Pinecone for semantic similarity matching.
Uses sentence-transformers (all-MiniLM-L6-v2) for free local embeddings.
"""

import os
import hashlib
from monitoring import logger

# Lazy-loaded globals to avoid slow imports on every module load
_model = None
_pc = None

DIMENSION = 384
INDEX_NAME = "job-skills"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded")
    return _model


def _get_pinecone():
    global _pc
    if _pc is None:
        from pinecone import Pinecone
        api_key = os.getenv("PINECONE_API_KEY", "")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set")
        _pc = Pinecone(api_key=api_key)
    return _pc


def _make_id(role: str, source: str, skill_name: str) -> str:
    """Create a deterministic short ID to avoid duplicates."""
    raw = f"{role}|{source}|{skill_name}".lower()
    return hashlib.md5(raw.encode()).hexdigest()[:16]


class SkillVectorStore:
    """Pinecone-backed skill embedding store."""

    def __init__(self):
        pc = _get_pinecone()

        existing = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing:
            from pinecone import ServerlessSpec
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Created Pinecone index '{INDEX_NAME}'")

        self.index = pc.Index(INDEX_NAME)
        self.model = _get_model()

    def embed_skills(self, skills: list[dict], role: str, source: str = "jd"):
        """Embed and upsert a list of skill dicts into Pinecone.

        Each skill dict must have at least a 'skill' key.
        source should be 'jd' for market skills or 'resume' for user skills.
        """
        if not skills:
            return

        vectors = []
        for s in skills:
            name = s.get("skill", "").strip()
            if not name:
                continue
            embedding = self.model.encode(name).tolist()
            vectors.append(
                {
                    "id": _make_id(role, source, name),
                    "values": embedding,
                    "metadata": {
                        "skill": name,
                        "role": role,
                        "source": source,
                        "category": s.get("category", "Technical"),
                        "importance": s.get("importance", "Medium"),
                        "job_count": s.get("job_count", 0),
                        "proficiency": s.get("proficiency", ""),
                    },
                }
            )

        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i : i + 100])

        logger.info(f"Upserted {len(vectors)} skill embeddings (source={source}, role={role})")

    def find_similar(self, query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict]:
        """Find skills semantically similar to the query string."""
        embedding = self.model.encode(query).tolist()

        filters = {}
        if source_filter:
            filters["source"] = {"$eq": source_filter}

        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filters if filters else None,
        )

        return [
            {"skill": m.metadata.get("skill", ""), "score": m.score, "metadata": m.metadata}
            for m in results.matches
        ]

    def get_role_skills(self, role: str, source: str = "jd", top_k: int = 50) -> list[dict]:
        """Retrieve all stored skills for a given role and source."""
        dummy = self.model.encode(role).tolist()
        results = self.index.query(
            vector=dummy,
            top_k=top_k,
            include_metadata=True,
            filter={"role": {"$eq": role}, "source": {"$eq": source}},
        )
        return [
            {"skill": m.metadata.get("skill", ""), "score": m.score, "metadata": m.metadata}
            for m in results.matches
        ]

    def clear_role(self, role: str):
        """Delete all vectors for a given role (use before re-analyzing)."""
        try:
            self.index.delete(filter={"role": {"$eq": role}})
            logger.info(f"Cleared vectors for role '{role}'")
        except Exception as e:
            logger.warning(f"Could not clear role vectors: {e}")
