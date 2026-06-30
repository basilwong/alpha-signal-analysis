# Alpha Signal Memory Agent

A persistent memory agent for quantum computing stock signal generation, built for the **Qwen Cloud Global AI Hackathon (Memory Agent Track)**.

The system ingests unstructured alternative data across seven sources (news, arXiv papers, patent filings, insider transactions, job postings, GitHub activity, and conference presentations) and translates them into actionable cross-sectional signal vectors for the quantum computing sector. 

The core innovation is a three-part persistent memory loop (semantic, episodic, and procedural memory) that enables the agent to autonomously accumulate domain knowledge, learn from its own prediction errors, and generate increasingly accurate trading signals over time without any model fine-tuning.

---

## The Core Hypothesis

In academic finance, the **Information Diffusion Hypothesis** states that highly technical or complex information diffuses slowly into asset prices because the average market participant lacks the specialized knowledge to interpret it. This is especially true for the quantum computing sector, where pure-play companies (IonQ, Rigetti, D-Wave) and diversified giants (IBM, Google, Microsoft) announce breakthroughs using dense physics terminology (e.g., "gate fidelity," "logical qubits," "error correction thresholds").

Our hypothesis is that a **stateless language model** struggles to value these announcements because it lacks historical sector context. However, an **agent with persistent memory** can connect dots across time and different data streams. By maintaining a running ledger of company capabilities, past predictions, and learned behavioral rules, the memory-augmented agent can capture the earliest possible alpha signals as information diffuses into the market.

---

## System Architecture

The platform consists of four main layers:

```
User → Gradio UI → Memory Retrieval → DashScope API (qwen-plus) → Signal Vector
                        ↑                                              ↓
                  SQLite Memory ← Knowledge Extraction ← Parse Response
                        ↑
                  Feedback Loop (episodic + procedural rules)
```

### 1. Data Ingestion (7 Alternative Sources)
The agent monitors seven data streams to build its knowledge base:
- **News Articles**: Google News RSS feeds for real-time market sentiment.
- **arXiv Papers**: Technical preprints in `quant-ph` to capture scientific milestones early.
- **Patent Filings**: USPTO PatentsView API to track early technological intellectual property.
- **Insider Transactions**: SEC EDGAR full-text search to monitor executive buying/selling (Form 4).
- **Job Postings**: Scraping hiring activity for key technical roles (e.g., "error correction engineer").
- **GitHub Activity**: Commit velocity and release history in key open-source quantum repos (Qiskit, Cirq, PennyLane).
- **Conference Presentations**: Tracking abstracts from major physics and computer science conferences.

### 2. The Persistent Memory Store
Built on a local SQLite database, the memory layer implements three distinct cognitive structures:
- **Semantic Memory**: Stores factual domain knowledge (e.g., "IBM's Heron processor has 133 qubits").
- **Episodic Memory**: Stores past prediction experiences and their actual market outcomes (e.g., "On 2026-03-15, I predicted IONQ bullish +1.5, actual return was +12% over 5 days").
- **Procedural Memory**: Stores behavioral rules learned from experience (e.g., "Be conservative on arXiv papers because scientific breakthroughs take months to impact stock prices").

### 3. The Feedback Loop
The self-improvement engine closes the loop by:
1. Recording predictions and matching them to actual 5-day abnormal returns (using Yahoo Finance market data).
2. Computing running accuracy statistics by source, ticker, and direction.
3. Generating statistical rules (e.g., "news predictions are 70% accurate, arXiv is 0%").
4. Calling Qwen Cloud API to generate advanced behavioral rules from recent episodes (LLM-as-a-Judge).
5. Injecting these rules into the system prompt for all future analyses.

---

## 6-Way Evaluation Results

We conducted a rigorous walk-forward evaluation on 200 chronological articles (Jan-Jun 2026) across six model configurations to isolate the value of memory and fine-tuning.

### Main Performance Metrics

#### 8B Base Model (no memory)
- Information Coefficient (IC) @5d: +0.0473 (p=0.1361)
- Direction Accuracy @5d: 53.8%
- Summary: Weak positive signal, barely better than random guessing.

#### 8B Model + Memory (Iterative Loop)
- Information Coefficient (IC) @5d: **+0.1068** (p<0.001, highly significant)
- Direction Accuracy @5d: **56.8%**
- Summary: The best overall directional predictor. Memory provides a **+125% improvement in IC** over the base model.

#### 14B Base Model (no memory)
- Information Coefficient (IC) @5d: +0.0059 (p=0.8514)
- Direction Accuracy @5d: 49.9%
- Summary: Zero predictive power. The larger model is too hedged and fails to capture directional signals.

#### 14B Model + Memory
- Information Coefficient (IC) @5d: -0.0066 (p=n/a)
- Direction Accuracy @5d: 53.2%
- Summary: Memory improves direction accuracy but the IC remains near zero.

#### 14B Fine-Tuned Model (no memory)
- Information Coefficient (IC) @5d: +0.0104 (p=0.7365)
- Direction Accuracy @5d: 51.9%
- Summary: Fine-tuning provides a negligible improvement over the base 14B model.

#### 14B Fine-Tuned Model + Memory
- Information Coefficient (IC) @5d: -0.0041 (p=n/a)
- Direction Accuracy @5d: 52.9%
- Summary: No meaningful improvement.

---

## Key Takeaways

1. **Memory beats scale**: The small 8B model with persistent memory (+0.107 IC) significantly outperforms the larger 14B base model (+0.006 IC) and the fine-tuned 14B model (+0.010 IC).
2. **Memory is a zero-cost alternative to fine-tuning**: The memory agent requires zero training compute, zero fine-tuning infrastructure, and runs entirely on a free API tier, yet closes 100% of the performance gap.
3. **The self-improvement loop works**: Across the 4 batches of 50 articles, the agent's direction accuracy improved steadily from **46% (Batch 1)** to **58% (Batch 4)** as it accumulated episodic memories and generated behavioral rules from its mistakes.

---

## Tech Stack
- **Model**: `qwen-plus-2025-07-28` via DashScope API (Alibaba Cloud)
- **Memory Store**: SQLite with TTL-based forgetting
- **Frontend**: Gradio (deployed on HuggingFace Spaces)
- **Deployment**: HuggingFace Spaces + Alibaba Cloud ECS (ap-southeast-1, Singapore)
- **Data Ingestion**: USPTO API, SEC EDGAR, Google News RSS, GitHub API, arXiv API
- **Market Data**: Yahoo Finance (`yfinance`)

---

## Deployment Proof
The backend is configured to run on an Alibaba Cloud ECS instance in the Singapore region:
- **Instance Type**: `ecs.t5-lc1m1.small` (burstable, free tier)
- **VPC ID**: `vpc-t4nx6ufy2rbp78muoj6s8`
- **Region**: `ap-southeast-1` (Singapore)
- **Inference**: DashScope API (Alibaba Cloud Model Studio)

The Terraform configuration is available in `infra/main.tf` and the deployment script is in `deploy.sh`.
