"""Hybrid Memory Agent combining Episodic Vector Memory (Qdrant) and User Profile (Feast).

Bonus Challenge — Lab 19: Vector Feature Store.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from feast import FeatureStore

ROOT = Path(__file__).resolve().parent.parent
FEAST_REPO = ROOT / "app" / "feast_repo"


class HybridMemoryAgent:
    """Combines episodic memory (dense vector retrieval) and user profile (feature store)."""

    def __init__(self, collection_name: str = "agent_episodic_memory"):
        self.collection_name = collection_name
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        self.point_counter = 0

        # Connect to Feast Feature Store (SQLite online store)
        self.fs = FeatureStore(repo_path=str(FEAST_REPO))
        self.profile_features = [
            "user_profile_features:reading_speed_wpm",
            "user_profile_features:preferred_language",
            "user_profile_features:topic_affinity",
            "query_velocity_features:queries_last_hour",
            "query_velocity_features:distinct_topics_24h",
        ]

    def remember(self, text: str, user_id: str = "u_001", metadata: dict[str, Any] | None = None) -> int:
        """Add a new piece of episodic memory for this user."""
        vec = next(self.embedder.embed([text])).tolist()
        self.point_counter += 1
        point_id = self.point_counter

        payload = {
            "user_id": user_id,
            "text": text,
            **(metadata or {}),
        }

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            ],
        )
        return point_id

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories + user profile features, assembling unified LLM context."""
        # 1. Fetch user profile + velocity from Feast online store
        profile_dict = {}
        try:
            raw_features = self.fs.get_online_features(
                features=self.profile_features,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            profile_dict = {k: raw_features[k][0] for k in raw_features if raw_features[k]}
        except Exception as e:
            profile_dict = {"error": str(e)}

        reading_speed = profile_dict.get("reading_speed_wpm", 200)
        language = profile_dict.get("preferred_language", "vi")
        affinity = profile_dict.get("topic_affinity", "general")
        queries_hour = profile_dict.get("queries_last_hour", 0)
        distinct_topics = profile_dict.get("distinct_topics_24h", 1)

        # 2. Vector search in Qdrant with tenant isolation (user_id filter)
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )

        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=user_filter,
            limit=top_k,
        ).points

        # 3. Assemble rich context prompt for generation
        memory_lines = []
        for i, hit in enumerate(search_result, 1):
            score = hit.score
            text = hit.payload.get("text", "")
            memory_lines.append(f"  [{i}] (similarity: {score:.3f}) {text}")

        memories_block = "\n".join(memory_lines) if memory_lines else "  (No relevant episodic memories found)"

        context = (
            f"=== USER CONTEXT (Feast Feature Store) ===\n"
            f"User ID           : {user_id}\n"
            f"Preferred Language: {language} | Reading Speed: {reading_speed} wpm\n"
            f"Topic Affinity    : {affinity}\n"
            f"Session Velocity  : {queries_hour} queries/hr across {distinct_topics} distinct topics\n"
            f"\n"
            f"=== EPISODIC RECALL (Qdrant Vector Store, query={query!r}) ===\n"
            f"{memories_block}\n"
            f"\n"
            f"=== SYNTHESIS GUIDELINES FOR LLM ===\n"
            f"- Reply in {language} matching user's reading pace ({reading_speed} wpm density).\n"
            f"- Ground response in recalled memories above, biased toward topic '{affinity}'."
        )
        return context
