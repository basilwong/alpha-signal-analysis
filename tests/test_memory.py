import tempfile
import unittest
from pathlib import Path

from src.memory import MemoryStore, infer_tags


class MemoryStoreTest(unittest.TestCase):
    def test_add_and_search_returns_relevant_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.jsonl")
            store.add(
                text="IonQ announced a trapped-ion error correction milestone.",
                source="news",
                title="IonQ milestone",
                observed_at="2026-06-01T00:00:00+00:00",
            )

            results = store.search("Does this IonQ error correction update matter?", limit=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["title"], "IonQ milestone")

    def test_infer_tags_detects_tickers(self):
        self.assertEqual(infer_tags("IONQ and RGTI moved on quantum news"), ["IONQ", "RGTI"])


if __name__ == "__main__":
    unittest.main()
