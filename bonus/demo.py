"""Demo script executing 5 test queries with the HybridMemoryAgent.

Bonus Challenge — Lab 19: Vector Feature Store.
Must exit with code 0 and output context assemblies for 5 distinct queries.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("=" * 70)
    print("HYBRID MEMORY AGENT DEMO: Episodic Vector Memory + Feast Feature Store")
    print("=" * 70)

    agent = HybridMemoryAgent()
    user_id = "u_001"

    # Seed initial episodic memories for user u_001
    print(f"\n[1] Seeding episodic memories for user '{user_id}'...")
    memories = [
        "Đã hoàn thành đọc tài liệu Kubernetes Architecture và cách triển khai StatefulSet trên multi-node cluster.",
        "Ghi chú dự án: Thiết kế giải pháp auto-scaling (HPA) trên đám mây dựa trên Custom Metrics và Prometheus.",
        "Đánh giá bảo mật: Cấu hình mã hoá dữ liệu lưu trữ (encryption at rest) và quản lý bí mật qua Vault cho hệ thống microservices.",
        "Nghiên cứu mô hình ngôn ngữ lớn: Tối ưu hóa bộ nhớ đệm ngữ nghĩa Semantic Cache với ngưỡng tương đồng Cosine 0.85.",
        "Kế hoạch quý 4: Nâng cấp cụm database PostgreSQL và thiết lập cơ chế đồng bộ Point-in-Time replication.",
    ]

    for m in memories:
        agent.remember(m, user_id=user_id)
    print(f"  -> Successfully stored {len(memories)} memory chunks in Qdrant vector collection.")

    # 5 Test queries demonstrating different retrieval behaviors
    queries = [
        (
            "Query 1 (Direct Episodic Vector Hit)",
            "Tôi đã đọc gì về Kubernetes?",
            "Expects high-similarity recall matching Kubernetes StatefulSet and multi-node cluster.",
        ),
        (
            "Query 2 (Profile Context & Topic Affinity)",
            "Recommend đọc gì tiếp theo?",
            "Leverages Feast topic_affinity and reading_speed_wpm to guide recommendation generation.",
        ),
        (
            "Query 3 (Recent Velocity & Session Activity)",
            "Tôi đang quan tâm gì gần đây?",
            "Utilizes Feast queries_last_hour and distinct_topics_24h to characterize session state.",
        ),
        (
            "Query 4 (Paraphrased Query without verbatim terms)",
            "Tài liệu về phương pháp tự động mở rộng hạ tầng theo lưu lượng?",
            "Demonstrates dense semantic recall mapping 'mở rộng hạ tầng theo lưu lượng' to 'auto-scaling (HPA)'.",
        ),
        (
            "Query 5 (Mixed Query: Episodic + Domain Profile)",
            "Cho tôi summary về cloud security và mã hóa dữ liệu?",
            "Synthesizes episodic Vault/encryption memories with Feast 'cloud' affinity profile.",
        ),
    ]

    print("\n" + "=" * 70)
    print("RUNNING 5 TEST QUERIES")
    print("=" * 70)

    for i, (q_type, q_text, expectation) in enumerate(queries, 1):
        print(f"\n>>> TEST [{i}/5]: {q_type}")
        print(f"    User Query : {q_text!r}")
        print(f"    Expectation: {expectation}")
        print("-" * 70)
        assembled_context = agent.recall(q_text, user_id=user_id, top_k=2)
        print(assembled_context)
        print("-" * 70)

    print("\n[SUCCESS] All 5 queries completed successfully with 0 errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
