# Project TODO

## Phase 1: MVP (Standard SFT + LoRA) — Target: June 15

### Data Collection
- [ ] Run `collect_articles.py` with sample articles (immediate test)
- [ ] Collect 50+ real arXiv papers via API
- [ ] Collect 50+ real news articles via RSS feeds
- [ ] Collect SEC filings for quantum tickers (IONQ, RGTI, QBTS)
- [ ] Target: 200+ raw articles in `data/raw/articles.jsonl`

### Training Data Generation (Teacher Model)
- [ ] Test teacher pipeline with 5 articles (verify qwen3-max output quality)
- [ ] Review and validate teacher outputs manually (spot check 10-20 examples)
- [ ] Run full generation on all 200+ articles
- [ ] Split dataset 90/10 into train/validation
- [ ] Target: `data/training/quantum_alpha_train.jsonl` ready

### Fine-Tuning (Modal)
- [ ] Verify Modal environment works (test_setup passes)
- [ ] Upload training data to Modal volume
- [ ] Run fine-tuning (standard QLoRA, LoRA rank 64, 4 epochs)
- [ ] Verify training loss converges
- [ ] Push model to HF Hub: `basilwong/quantum-alpha-qwen3-8b`

### Evaluation
- [ ] Run base qwen3-8b on validation set (baseline metrics)
- [ ] Run fine-tuned model on validation set (comparison metrics)
- [ ] Calculate: Sentiment Accuracy, Event F1, Ticker Jaccard, JSON pass rate
- [ ] Document improvement delta

### App Integration
- [ ] Wire fine-tuned model into Gradio Server `/analyze_news` endpoint
- [ ] Implement `/get_signals` with live data ingestion
- [ ] Implement `/get_briefing` with daily narrative generation
- [ ] Implement `/get_sector_overview` with aggregated sentiment
- [ ] Test end-to-end: paste article → get structured signal back

### Frontend Polish
- [ ] Populate dashboard with real signals
- [ ] Fix any UI bugs
- [ ] Ensure responsive design works
- [ ] Add loading states and error handling

### Submission (Build Small Hackathon — June 15)
- [ ] Deploy to HF Space: `build-small-hackathon/quantum-alpha-intelligence`
- [ ] Verify app runs on ZeroGPU
- [ ] Record 2-minute demo video
- [ ] Write social media post
- [ ] Submit: Space link + video + social post

---

## Phase 2: Experimentation (Post-MVP) — Target: July 9

### Alternative Fine-Tuning Approaches
- [ ] Experiment: DoRA (`use_dora=True`) — compare metrics vs standard LoRA
- [ ] Experiment: Curriculum ordering (sort training data by complexity)
- [ ] Experiment: GRPO with price-based reward signal
- [ ] Experiment: Larger training dataset (500+ examples)
- [ ] Document performance comparison across approaches

### Memory Agent (Qwen Cloud Hackathon)
- [ ] Set up ChromaDB vector store for persistent memory
- [ ] Implement memory ingestion (store past analyses)
- [ ] Implement memory retrieval (RAG for contextual recall)
- [ ] Implement memory forgetting (TTL for outdated info)
- [ ] Deploy backend on Alibaba Cloud ECS
- [ ] Connect to qwen3-max API for V1 inference

### Qwen Cloud Hackathon Submission (July 9)
- [ ] Create public GitHub repo with open-source license
- [ ] Create architecture diagram
- [ ] Record 3-minute demo video
- [ ] Write blog post (for bonus prize)
- [ ] Submit on DevPost: repo + video + deck + description

---

## Completed
- [x] Research hackathon rules and requirements
- [x] Set up GitHub repo (basilwong/quantum-alpha-intelligence)
- [x] Create HF Space (build-small-hackathon/quantum-alpha-intelligence)
- [x] Build frontend skeleton (custom trading terminal UI)
- [x] Write Gradio Server app skeleton with API endpoints
- [x] Research prior art (NLP alpha generation papers)
- [x] Research evaluation methodology
- [x] Research hackathon/credit opportunities
- [x] Confirm Qwen Cloud API access (Singapore, free tier, qwen3-max working)
- [x] Confirm Modal access (authenticated, workspace connected)
- [x] Write Modal fine-tuning script
- [x] Write teacher data generation pipeline
- [x] Write article collection script
- [x] Write end-to-end system design document
