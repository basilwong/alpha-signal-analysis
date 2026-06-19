import tempfile
import unittest
from pathlib import Path

from src.tracker import RunTracker


class RunTrackerTest(unittest.TestCase):
    def test_records_trade_eval_and_backtest_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = RunTracker(Path(tmp))

            trade = tracker.record_trade(
                strategy="memory-agent-v1",
                symbol="IONQ",
                side="buy",
                quantity=25,
                price=42.5,
                source="paper",
                metadata={"signal_score": 1.2},
            )
            evaluation = tracker.record_eval(
                name="signal-json-schema",
                status="pass",
                metrics={"valid_rate": 0.98},
                metadata={"sample_size": 50},
            )
            backtest = tracker.record_backtest(
                strategy="memory-agent-v1",
                dataset="quantum-news-2026",
                start_date="2026-06-01",
                end_date="2026-06-15",
                metrics={"sharpe": 1.1, "max_drawdown": -0.08},
            )

            self.assertEqual(trade["kind"], "trade")
            self.assertEqual(evaluation["kind"], "eval")
            self.assertEqual(backtest["kind"], "backtest")

            self.assertEqual(len(tracker.list_records("trades")), 1)
            self.assertEqual(len(tracker.list_records("evals")), 1)
            self.assertEqual(len(tracker.list_records("backtests")), 1)

    def test_rejects_invalid_trade_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = RunTracker(Path(tmp))

            with self.assertRaises(ValueError):
                tracker.record_trade(
                    strategy="memory-agent-v1",
                    symbol="IONQ",
                    side="hold",
                    quantity=25,
                    price=42.5,
                )


if __name__ == "__main__":
    unittest.main()
