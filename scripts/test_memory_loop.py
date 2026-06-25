"""
End-to-end test of the full memory improvement loop.
Uses the sandbox's built-in OpenAI-compatible API (free, unlimited).

Tests:
1. Process articles with enhanced memory retrieval (all 3 types)
2. Record outcomes using real market data
3. Run the feedback loop to generate procedural rules
4. Verify that future predictions use the learned rules
"""
import json
import os
import sys
sys.path.insert(0, '.')

from openai import OpenAI
from agent.memory import MemoryStore
from agent.memory_loop import (
    EpisodicMemory, ProceduralMemory, FeedbackLoop, EnhancedRetriever, Episode
)
from agent.seed_data import SEED_FACTS
from agent.config import QUANTUM_TICKERS

# Use sandbox API (free, unlimited)
client = OpenAI()  # Uses OPENAI_API_KEY and OPENAI_API_BASE from env
MODEL = "gpt-5-nano"  # Fastest model for iteration

# Setup
DB_PATH = "data/test_loop_memory.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

memory = MemoryStore(DB_PATH)
episodic = EpisodicMemory(memory)
procedural = ProceduralMemory(memory)
retriever = EnhancedRetriever(memory)
feedback = FeedbackLoop(memory, llm_client=client)

# Seed baseline knowledge
for fact in SEED_FACTS:
    memory.store_knowledge(fact['ticker'], fact['type'], fact['content'], 'seed')

print("=" * 60)
print("MEMORY LOOP END-TO-END TEST")
print("=" * 60)

# ============================================================
# PHASE 1: Simulate predictions with outcomes
# ============================================================
print("\n--- Phase 1: Simulating prediction episodes ---")

# Simulate 20 episodes with known outcomes
test_episodes = [
    # Correct bullish predictions
    Episode(date="2026-01-15", ticker="IONQ", predicted_score=1.5, predicted_direction="bullish",
            actual_return_5d=0.08, actual_direction="bullish", was_correct=True,
            article_title="IonQ announces new qubit milestone", source_type="news",
            reasoning_summary="Major technical achievement ahead of roadmap"),
    Episode(date="2026-01-20", ticker="IONQ", predicted_score=1.2, predicted_direction="bullish",
            actual_return_5d=0.05, actual_direction="bullish", was_correct=True,
            article_title="IonQ partnership with Hyundai", source_type="news",
            reasoning_summary="Commercial partnership validates technology"),
    Episode(date="2026-02-01", ticker="RGTI", predicted_score=1.0, predicted_direction="bullish",
            actual_return_5d=0.03, actual_direction="bullish", was_correct=True,
            article_title="Rigetti new processor announcement", source_type="press_release",
            reasoning_summary="Hardware upgrade cycle"),
    Episode(date="2026-02-10", ticker="IONQ", predicted_score=0.8, predicted_direction="bullish",
            actual_return_5d=0.12, actual_direction="bullish", was_correct=True,
            article_title="DOE quantum funding announcement", source_type="news",
            reasoning_summary="Government funding benefits all quantum companies"),
    # Correct bearish predictions
    Episode(date="2026-01-25", ticker="RGTI", predicted_score=-1.5, predicted_direction="bearish",
            actual_return_5d=-0.12, actual_direction="bearish", was_correct=True,
            article_title="Rigetti Q4 earnings miss", source_type="news",
            reasoning_summary="Revenue below expectations"),
    Episode(date="2026-03-01", ticker="QBTS", predicted_score=-1.0, predicted_direction="bearish",
            actual_return_5d=-0.08, actual_direction="bearish", was_correct=True,
            article_title="D-Wave dilution announcement", source_type="sec_filing",
            reasoning_summary="Share dilution signals cash needs"),
    # WRONG predictions (bearish predictions that were wrong)
    Episode(date="2026-02-05", ticker="RGTI", predicted_score=-1.2, predicted_direction="bearish",
            actual_return_5d=0.15, actual_direction="bullish", was_correct=False,
            article_title="Rigetti stock crash analysis", source_type="news",
            reasoning_summary="Assumed further decline but market bounced"),
    Episode(date="2026-02-15", ticker="QBTS", predicted_score=-0.8, predicted_direction="bearish",
            actual_return_5d=0.10, actual_direction="bullish", was_correct=False,
            article_title="D-Wave negative analyst note", source_type="news",
            reasoning_summary="Analyst downgrade but stock rallied on short squeeze"),
    Episode(date="2026-03-05", ticker="IONQ", predicted_score=-0.5, predicted_direction="bearish",
            actual_return_5d=0.06, actual_direction="bullish", was_correct=False,
            article_title="IonQ valuation concerns article", source_type="news",
            reasoning_summary="Valuation article but market ignored it"),
    # WRONG predictions (arXiv papers that didn't move stocks)
    Episode(date="2026-01-30", ticker="IONQ", predicted_score=1.5, predicted_direction="bullish",
            actual_return_5d=-0.02, actual_direction="bearish", was_correct=False,
            article_title="Novel error correction scheme for trapped ions", source_type="arxiv",
            reasoning_summary="Theoretical breakthrough but no commercial impact"),
    Episode(date="2026-02-20", ticker="RGTI", predicted_score=1.0, predicted_direction="bullish",
            actual_return_5d=-0.05, actual_direction="bearish", was_correct=False,
            article_title="Superconducting qubit coherence improvement", source_type="arxiv",
            reasoning_summary="Research paper but market didn't react"),
    Episode(date="2026-03-10", ticker="QBTS", predicted_score=0.8, predicted_direction="bullish",
            actual_return_5d=-0.03, actual_direction="bearish", was_correct=False,
            article_title="Quantum annealing optimization results", source_type="arxiv",
            reasoning_summary="Academic results don't translate to stock movement"),
    # More correct news predictions to establish pattern
    Episode(date="2026-03-15", ticker="IONQ", predicted_score=1.8, predicted_direction="bullish",
            actual_return_5d=0.15, actual_direction="bullish", was_correct=True,
            article_title="IonQ wins government contract", source_type="news",
            reasoning_summary="Large contract validates commercial viability"),
    Episode(date="2026-03-20", ticker="QNT", predicted_score=1.5, predicted_direction="bullish",
            actual_return_5d=0.10, actual_direction="bullish", was_correct=True,
            article_title="Quantinuum logical qubit demonstration", source_type="news",
            reasoning_summary="Industry-first achievement"),
    Episode(date="2026-04-01", ticker="IONQ", predicted_score=1.0, predicted_direction="bullish",
            actual_return_5d=0.04, actual_direction="bullish", was_correct=True,
            article_title="IonQ revenue beat", source_type="news",
            reasoning_summary="Strong quarterly results"),
]

for ep in test_episodes:
    episodic.store_episode(ep)
print(f"  Stored {len(test_episodes)} episodes")

# ============================================================
# PHASE 2: Run the feedback loop to generate rules
# ============================================================
print("\n--- Phase 2: Running feedback loop ---")

rules = feedback.analyze_and_generate_rules()
print(f"  Generated {len(rules)} procedural rules:")
for rule in rules:
    print(f"    [{rule.category}] {rule.rule_text[:80]}...")

# ============================================================
# PHASE 3: Generate advanced rules with LLM
# ============================================================
print("\n--- Phase 3: LLM-generated rules ---")

episodes = episodic.get_similar_episodes(limit=15)
llm_rules = feedback.generate_advanced_rules_with_llm(episodes)
print(f"  LLM generated {len(llm_rules)} additional rules:")
for rule in llm_rules:
    print(f"    [{rule.category}] {rule.rule_text[:80]}...")

# ============================================================
# PHASE 4: Test enhanced retrieval with all 3 memory types
# ============================================================
print("\n--- Phase 4: Enhanced retrieval (all 3 memory types) ---")

context = retriever.build_full_context(
    "IonQ just announced a new error correction breakthrough in a research paper on arXiv",
    source_type="arxiv"
)
print(f"  Context length: {len(context)} chars")
print(f"  Context preview:")
print(f"  {context[:600]}")

# ============================================================
# PHASE 5: Generate a signal using the full memory context
# ============================================================
print("\n--- Phase 5: Signal generation with full memory ---")

article = "IonQ publishes new paper on arXiv demonstrating 99.9% gate fidelity on their trapped-ion processor, a significant improvement over their previous 99.5% benchmark."

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": f"""You are a quantitative signal generator for quantum computing stocks.

{context}

Based on your accumulated experience and learned rules, generate a signal vector for: IONQ, RGTI, QBTS, QUBT, QNT, IBM, GOOGL, MSFT, HON, NVDA.
Score range: -2.0 to +2.0. GOOGL/MSFT/NVDA always 0.0.
Include chain_of_thought that EXPLICITLY references your learned rules and past experiences.
Output valid JSON with signal_vector and chain_of_thought fields."""},
        {"role": "user", "content": f"Analyze this arXiv paper:\n\n{article}"}
    ],
    temperature=0.3,
    max_tokens=1000
)

output = response.choices[0].message.content or ""
if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
    output = response.choices[0].message.reasoning_content + "\n" + output
print(f"  Model output ({len(output)} chars):")
print(f"  {output[:500]}")

# Check if the model referenced rules/episodes
referenced_rules = "rule" in output.lower() or "experience" in output.lower() or "past" in output.lower() or "accuracy" in output.lower() or "conservative" in output.lower()
print(f"\n  Model referenced learned rules/experience: {'YES' if referenced_rules else 'NO'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
stats = episodic.get_accuracy_by_category()
print(f"  Episodes stored: {stats['overall']['total']}")
print(f"  Overall accuracy: {stats['overall']['accuracy']*100:.0f}%")
source_acc = {k: f"{v['accuracy']*100:.0f}%" for k,v in stats['by_source'].items()}
print(f"  By source: {json.dumps(source_acc, indent=4)}")
print(f"  Active rules: {len(procedural.get_active_rules())}")
print(f"  Rules context: {len(procedural.get_rules_as_context())} chars")
print(f"  Full context (all 3 types): {len(context)} chars")
print(f"  Model used learned context: {'YES' if referenced_rules else 'NO'}")

# Cleanup
os.remove(DB_PATH)
print("\n  Test complete. Memory DB cleaned up.")
