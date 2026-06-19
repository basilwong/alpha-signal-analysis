"""Reusable sector context for quantum computing signal analysis."""

from src.config import ADJACENT_TICKERS, INACTIVE_TICKERS, PURE_PLAY_TICKERS


TECHNOLOGY_DYNAMICS = {
    "trapped ion": "Bullish for IONQ/QNT/HON when the event validates trapped-ion hardware; company-specific wins can be competitive.",
    "superconducting": "Bullish for RGTI/IBM when the event validates superconducting hardware; broad validation can help smaller pure plays.",
    "quantum annealing": "Most relevant to QBTS and optimization workflows.",
    "error correction": "Usually validates gate-based approaches broadly, but score magnitude depends on commercial proximity.",
    "government funding": "Usually sector-positive unless funding is explicitly zero-sum.",
}


def format_sector_context() -> str:
    """Return compact sector context for prompts."""
    lines = ["Quantum computing ticker universe:"]
    lines.append("Active pure plays:")
    for ticker, info in PURE_PLAY_TICKERS.items():
        lines.append(
            f"- {ticker}: {info['name']}, {info['technology']}, score range +/-{info['max_score']}"
        )

    lines.append("Adjacent active tickers:")
    for ticker, info in ADJACENT_TICKERS.items():
        lines.append(
            f"- {ticker}: {info['name']}, {info['technology']}, capped at +/-{info['max_score']}"
        )

    lines.append("Inactive tickers:")
    for ticker, reason in INACTIVE_TICKERS.items():
        lines.append(f"- {ticker}: always 0.0 unless the user explicitly changes scope. Reason: {reason}")

    lines.append("Technology dynamics:")
    for topic, rule in TECHNOLOGY_DYNAMICS.items():
        lines.append(f"- {topic}: {rule}")

    return "\n".join(lines)
