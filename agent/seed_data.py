"""Seed the memory database with initial quantum computing sector knowledge."""

SEED_FACTS = [
    {"ticker": "IONQ", "type": "technology", "content": "IonQ uses trapped-ion quantum computing approach. Current system: Forte Enterprise with 36 algorithmic qubits."},
    {"ticker": "IONQ", "type": "roadmap", "content": "IonQ targets 1024 qubits by 2028. Key milestone: achieve broad quantum advantage."},
    {"ticker": "RGTI", "type": "technology", "content": "Rigetti uses superconducting transmon qubits. Current system: Ankaa-3 with 84 qubits."},
    {"ticker": "RGTI", "type": "roadmap", "content": "Rigetti targets 4000 qubits by 2026. Focus on modular architecture."},
    {"ticker": "QBTS", "type": "technology", "content": "D-Wave uses quantum annealing (not gate-based). Current system: Advantage2 with 1200+ qubits."},
    {"ticker": "QBTS", "type": "competitive", "content": "D-Wave is the only commercial quantum annealing company. Limited to optimization problems."},
    {"ticker": "QNT", "type": "technology", "content": "Quantinuum (Honeywell subsidiary) uses trapped-ion approach. Highest quantum volume in industry."},
    {"ticker": "QNT", "type": "milestone", "content": "Quantinuum demonstrated first fault-tolerant quantum computation in 2024."},
    {"ticker": "IBM", "type": "technology", "content": "IBM uses superconducting qubits. Current system: Heron with 133 qubits. Roadmap targets 100K qubits by 2033."},
    {"ticker": "HON", "type": "exposure", "content": "Honeywell owns majority stake in Quantinuum. Quantum is ~5% of Honeywell's strategic focus."},
    # Competitive dynamics
    {"ticker": "IONQ", "type": "competitive", "content": "Trapped-ion vs superconducting is the primary technology rivalry. Trapped-ion has higher fidelity but slower gate speeds."},
    {"ticker": "RGTI", "type": "competitive", "content": "Superconducting approach benefits from semiconductor manufacturing expertise. Faster gates but lower coherence times."},
]


if __name__ == "__main__":
    """Run this script directly to seed the memory database."""
    from .memory import MemoryStore
    from .config import MEMORY_DB_PATH

    store = MemoryStore(MEMORY_DB_PATH)
    for fact in SEED_FACTS:
        store.store_knowledge(
            ticker=fact["ticker"],
            fact_type=fact["type"],
            content=fact["content"],
            source="seed",
            confidence=0.9
        )
    print(f"Seeded {len(SEED_FACTS)} facts into memory.")
    print(f"Memory stats: {store.get_memory_stats()}")
