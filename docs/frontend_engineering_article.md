# Frontend Engineering: Deploying a Custom Trading Terminal on Hugging Face Spaces

## The Challenge

The Quantum Alpha Intelligence platform needed a custom frontend. Not a Gradio widget grid, not a Streamlit dashboard, but a purpose built terminal style interface with interactive Plotly charts, tabbed navigation, a welcome overlay, concurrent inference feeds, and a FastAPI backend serving JSON endpoints. The problem: Hugging Face Spaces with ZeroGPU expects a Gradio app, manages port binding internally, and wraps `demo.launch()` with its own runtime. Our custom HTML/CSS/JS frontend with a FastAPI backend kept crashing on deploy.

## The Port Binding Problem

On HF Spaces with `sdk: gradio`, the runtime occupies port 7860 before your code even runs. The `spaces` package patches Gradio's `launch()` method to handle GPU allocation and authentication. When we tried the standard pattern of mounting Gradio onto FastAPI using `gr.mount_gradio_app()` and then calling `demo.launch()`, the Spaces runtime had already claimed port 7860. Our app would crash with:

```
OSError: Cannot find empty port in range: 7860-7860
```

We tried several approaches that failed:

1. **Removing `server_port=7860`** from `demo.launch()` still failed because the runtime's port range is locked to 7860.
2. **Adding `ssr_mode=False`** avoided the Node.js SSR proxy error but didn't solve the underlying port conflict.
3. **Using `gr.mount_gradio_app(app, demo, path="/gradio")`** then calling `demo.launch()` created a circular dependency where Gradio tried to start its own uvicorn server on a port that was already occupied by the very runtime that was supposed to host it.

The fundamental issue: `gr.mount_gradio_app()` is designed for when YOU control the server (running uvicorn yourself). On HF Spaces, the runtime controls the server. You cannot call both `gr.mount_gradio_app()` and `demo.launch()` because `launch()` tries to start a second server.

## The Solution: `gradio.Server` Mode

Gradio 5.29+ introduced `gradio.Server`, a class that inherits directly from FastAPI. Instead of creating a FastAPI app and trying to bolt Gradio onto it, you use Gradio's own FastAPI subclass that handles all the port management, queue infrastructure, and ZeroGPU integration internally.

```python
from gradio import Server
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = Server(title="Quantum Alpha Intelligence API")

@app.get("/api/models")
async def get_models():
    return JSONResponse({"models": [...]})

@app.get("/")
async def serve_index():
    return FileResponse("frontend_v2/index.html")

app.mount("/static", StaticFiles(directory="frontend_v2"), name="static")

# For ZeroGPU support, use @app.api() for queued endpoints
@app.api(name="analyze")
def analyze_via_gradio(text: str, source: str, model_name: str, enable_thinking: bool) -> str:
    return gpu_inference(text, source, model_name, enable_thinking)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
```

This works because `Server.launch()` is aware of the HF Spaces runtime. It doesn't try to start a competing server. It registers itself with the existing infrastructure, and all your custom FastAPI routes (`/api/*`, `/`, `/static/*`) coexist with Gradio's internal endpoints (`/gradio_api/*`) on the same port.

The key insight: `gradio.Server` is not "Gradio mounted on FastAPI." It IS FastAPI, with Gradio's queue and GPU allocation built in. Your custom routes are first class citizens, not afterthoughts mounted onto someone else's app.

## ZeroGPU Integration

For live inference on ZeroGPU hardware (`zero-a10g`), the `@spaces.GPU` decorator needs to wrap the function that touches the GPU. We kept this pattern but routed it through both a Gradio API endpoint (for proper queue/auth handling) and a standard FastAPI POST endpoint (for our custom frontend to call):

```python
try:
    import spaces
    @spaces.GPU
    def gpu_inference(text, source, model_name, enable_thinking):
        return _do_inference(text, source, model_name, enable_thinking)
except ImportError:
    def gpu_inference(text, source, model_name, enable_thinking):
        return _do_inference(text, source, model_name, enable_thinking)

# Gradio API endpoint (queue + ZeroGPU auth)
@app.api(name="analyze")
def analyze_via_gradio(text: str, source: str, model_name: str, enable_thinking: bool) -> str:
    return gpu_inference(text, source, model_name, enable_thinking)

# Standard FastAPI endpoint (for our custom frontend)
@app.post("/api/analyze")
async def analyze(request: Request):
    body = await request.json()
    result = gpu_inference(body["text"], body["source"], body["model"], body["enable_thinking"])
    return JSONResponse(json.loads(result))
```

A critical deployment issue: the model repository (`basilwong/quantum-alpha-openreasoning-7b-grpo`) was initially set to private. The Space could not download the model weights for inference, returning a generic 500 Internal Server Error. Making the model public resolved this. We also increased `max_new_tokens` to 10,000 because the GRPO model outputs detailed per-ticker reasoning that was getting truncated at lower limits.

## The Frontend Architecture

The frontend is pure HTML/CSS/JS with no build step. Three files: `index.html`, `styles.css`, `app.js`. Served as static files from the same origin as the API. No CORS issues, no proxy needed, no framework overhead.

### Design System: Terminal Aesthetic

The visual language communicates "quantitative finance infrastructure" through a terminal inspired design system:

- **Pure black backgrounds** (`#0c0c0c`, `#121212`, `#1a1a1a`) with no gradients
- **Monospace typography** (JetBrains Mono) for all text, including headings and body
- **Green accent** (`#00ff88`) for active states, borders, titles, and interactive elements
- **Minimal border radius** (2px) for sharp, technical edges
- **Uppercase headers** with letter-spacing for a data terminal feel
- **Outlined buttons** (green border, transparent background) that invert to solid green on hover
- **No font weight variation** (all 400) to maintain the monospace terminal consistency

CSS custom properties define the entire color system, making it trivial to adjust:

```css
:root {
    --bg-primary: #0c0c0c;
    --bg-secondary: #121212;
    --bg-card: #1a1a1a;
    --border: #2a2a2a;
    --accent-green: #00ff88;
    --accent-red: #ff4444;
    --text-primary: #d4d4d4;
    --text-secondary: #808080;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace;
    --radius: 2px;
}
```

### Welcome Overlay

A full screen overlay appears on every visit, requiring users to click through before accessing the dashboard. This serves as the product pitch:

- Title and tagline (NLP-Driven Alpha Signal for Quantitative Trading)
- Four stat cards: IC@5d (+0.157), statistical significance (p=0.006), ticker coverage (9), model iterations (7)
- Description of what the signal does and how it's consumed
- Market context with a link to Citadel seeking alternative signal sources
- "Explore" CTA button at the top so it's immediately visible without scrolling

The overlay is scrollable (`overflow-y: auto`) for mobile screens where the content exceeds viewport height. An "About This Project" button on the Evaluation Dashboard re-opens the overlay for returning users.

## Tab Structure and Product Positioning

The tab order positions the product as a fine-tuning evaluation tool for quant signal development:

1. **Evaluation Dashboard** (landing page after overlay)
2. **Live Signal Debugging**
3. **Historical Prediction Analysis**
4. **Sector Map**

### Tab 1: Evaluation Dashboard

The landing page shows IC metrics across all 13 model iterations. Two context cards define IC and explain how to read the dashboard, making it accessible to users without a quant finance background.

The Signal Decay Curve plots IC at horizons of 1, 2, 5, 10, and 20 days. Bright colored markers indicate statistical significance (p < 0.10), gray markers indicate noise. The legend sits below the chart in horizontal orientation to avoid occluding data.

**Default state:** Only 3 models are checked on load (the best fine-tuned model and both teacher models). This prevents visual overload while clearly showing the student-beats-teacher result. Users can toggle on additional models to explore the full training progression.

The IC Comparison Table provides exact numbers with color coding. Model names use explicit descriptive labels:

```
Nemotron-7B (SFT + GRPO, Manus Teacher)
Nemotron-7B (SFT, GPT-5.5 Teacher)
Manus (Teacher, Direct)
Nemotron-14B (Base, No Fine-Tuning)
```

This naming convention immediately communicates the base model, training method, and teacher without requiring version number lookup.

### Tab 2: Live Signal Debugging

This tab supports **concurrent, non-blocking analyses** displayed as a scrollable feed. The design allows rapid iteration:

- Input form stays at the top (model selector, source type, thinking toggle, textarea)
- Clicking Analyze immediately adds a "loading" card to the feed and clears the input
- The request fires asynchronously so you can submit the next article without waiting
- Results populate into each card as they return
- Newest results appear at the top (prepend order)

Each feed card shows:
- Header: model badge (green outline), source badge, timestamp, latency
- Article preview (first 150 chars)
- Inline signal bar chart (horizontal bars, green/red, sorted by score)
- Metadata line (event type, horizon, decay, novelty)
- Expandable "Raw JSON" and "Thinking Trace" sections

The feed handles multiple signal formats gracefully:
- V7d GRPO outputs: `{signal_vector: {IONQ: {score: 1.8, reasoning: "..."}}}`
- V4 Baseline outputs: `{IONQ: 1.5, RGTI: 0.8, ...}`
- Parse errors display the raw output instead of crashing

A "Clear Feed" button resets the session. The feed resets naturally on page refresh.

### Tab 3: Historical Prediction Analysis

This tab loads precomputed predictions from all model iterations and displays them against realized market outcomes. The event selector contains 388+ articles from Jan to Jun 2026.

**Model filter checkboxes** allow toggling which models appear in the signal comparison chart. The chart re-renders instantly on checkbox change without refetching data.

Three price charts provide proper context for evaluating predictions:

1. **Raw Price Movement** shows cumulative returns with SPY overlaid as a dashed benchmark line.
2. **Abnormal Returns vs. Market (SPY)** computes stock return minus SPY return at each day. This is the actual metric that IC is evaluated against. A stock going up 5% when the market went up 7% is bearish alpha, not bullish.
3. **Abnormal Returns vs. Sector (Equal-Weight Quantum Basket)** computes stock return minus the average of pure play quantum tickers (IONQ, RGTI, QBTS, QUBT). This shows whether the model correctly identified relative winners within the sector versus just riding a sector wide move.

### The Article Index Alignment Problem

During development, we discovered that the Manus Teacher predictions used a completely different `article_idx` numbering than the other models. The same `article_idx=0` pointed to different articles across files. When displaying side by side comparisons, the Manus model appeared to be analyzing a D-Wave article when the UI said it was showing an IonQ article.

The fix: re-indexed the Manus predictions by matching on article title to the canonical ordering used by the fine-tuned model. Articles that only existed in the Manus dataset (172 out of 421) were dropped. The remaining 249 now correctly align.

### Tab 4: Sector Map

A reference tab showing the quantum computing sector structure: signal weights by company, technology clusters, and signal propagation rules.

## Mobile Responsive Design

The terminal theme required careful mobile adaptation since monospace fonts and uppercase text consume more horizontal space. All responsive rules are CSS-only (no JS changes), applied through 8 `@media` query blocks:

**Tablets (768px and below):**
- Header: smaller logo, hackathon badge hidden, status text hidden
- Tab navigation: horizontal scroll with hidden scrollbar, smaller text, `flex-shrink: 0` prevents tab compression
- Content padding reduced from 24px to 12px
- Eval context cards stack to single column
- Checkbox groups wrap with smaller pill sizes
- Tables get horizontal scroll on overflow
- Feed card headers stack vertically (badges above, status below)
- Model details grid goes single column
- Sector clusters go to 2-column grid
- Footer stacks vertically

**Phones (480px and below):**
- Welcome overlay stats go 2x2 grid, CTA button becomes full width
- Tab buttons shrink further (11px, 8px padding)
- Live controls stack vertically (model, source, thinking each full width)
- Analyze button spans full width
- Feed signal bars use tighter grid (36px ticker column)
- Chart min-height reduced to 220px
- Content padding reduced to 8px sides

**Key mobile patterns:**
- `overflow-x: auto` with `-webkit-overflow-scrolling: touch` on all chart containers enables horizontal swipe
- `white-space: nowrap` on tab buttons prevents text wrapping while allowing scroll
- `word-wrap: break-word` on article titles prevents overflow in the event selector
- The welcome overlay uses `overflow-y: auto` so all content is accessible regardless of viewport height
- The CTA button is positioned near the top of the overlay so it's immediately visible without scrolling

## Deployment Pipeline

The deployment workflow uses `huggingface_hub` Python library:

```python
from huggingface_hub import HfApi
api = HfApi(token=TOKEN)

# Upload files
api.upload_file(path_or_fileobj='app_server_fixed.py', path_in_repo='app.py',
                repo_id=SPACE_ID, repo_type='space')

# Restart and verify
api.restart_space(SPACE_ID)
info = api.space_info(SPACE_ID)
assert info.runtime.stage == 'RUNNING'
```

Each iteration follows the same loop: modify locally, upload to Space, restart, wait 90 seconds, check status. If `RUNTIME_ERROR`, read the error message, fix, repeat. If `RUNNING`, verify endpoints with requests.

The Space configuration:
- SDK: `gradio` (version 6.16.0)
- Hardware: `zero-a10g` (ZeroGPU, NVIDIA A10G)
- Python: 3.12
- App file: `app.py`

## Evaluation Script: Handling Multiple Signal Formats

The evaluation script (`eval/run_multi_model_eval.py`) computes IC across all 13 models. Different models output signals in different formats:

- **Dict with score objects:** `{IONQ: {score: 1.5, reasoning: "..."}}`
- **Dict with float values:** `{IONQ: 1.5}`
- **List format:** `[{ticker: "IONQ", score: 1.5, reasoning: "..."}]`

The script normalizes all formats before computing Spearman rank correlation:

```python
if isinstance(signal_vector, list):
    sv_dict = {}
    for item in signal_vector:
        if isinstance(item, dict) and "ticker" in item:
            sv_dict[item["ticker"]] = item
    signal_vector = sv_dict

val = signal_vector[ticker]
if isinstance(val, dict):
    predicted_score = val.get("score", 0)
elif isinstance(val, (int, float)):
    predicted_score = val
predicted_score = float(predicted_score)
```

## What We Learned

1. **`gradio.Server` is the correct pattern for custom frontends on HF Spaces.** Not `mount_gradio_app`, not running uvicorn yourself, not Docker SDK. Server mode gives you full FastAPI control while letting Gradio manage the infrastructure.

2. **ZeroGPU authentication flows through Gradio's internal mechanisms.** If you bypass Gradio entirely (e.g., running raw uvicorn), ZeroGPU quota tracking breaks. The `@spaces.GPU` decorator needs to be called within Gradio's execution context.

3. **Model repos must be public for the Space to download them.** A private model repo causes a generic 500 error with no useful error message in the response. The Space runtime logs show the actual download failure.

4. **`max_new_tokens` matters for verbose models.** The GRPO model outputs detailed reasoning per ticker. At 1024 tokens, the JSON gets truncated mid-object. We set it to 10,000 to be safe.

5. **Data alignment across model prediction files cannot be assumed.** Different model runs may use different article orderings. Always join on a stable key (title, date) rather than assuming positional indices match.

6. **Plotly legends with 7+ traces need to be moved outside the chart.** Horizontal orientation below the plot (`orientation: 'h', y: -0.25`) keeps the data visible.

7. **Default to less, let users add more.** With 13 models, showing all checkboxes checked on load makes the decay curve unreadable. Defaulting to 3 (best model + teachers) tells the story clearly while inviting exploration.

8. **Non-blocking concurrent requests are essential for debugging.** The feed pattern (fire-and-forget fetch, render on completion) lets you submit 5 articles in rapid succession and compare outputs as they arrive. This is far more useful than a blocking single-result UI.

9. **Terminal aesthetics work well for quant tools.** Monospace fonts, green-on-black, sharp corners, and uppercase headers communicate "infrastructure" and "precision" to the target audience (quant developers, ML engineers at trading firms).

10. **CSS-only responsive design is safe.** All mobile adaptations were pure media queries appended to the stylesheet. No JS logic changes, no HTML restructuring. This means zero risk of breaking desktop behavior.

## Final Architecture

```
HF Space (zero-a10g)
├── app.py (gradio.Server)
│   ├── GET /                    → FileResponse(index.html)
│   ├── GET /static/*            → StaticFiles(frontend_v2/)
│   ├── GET /api/models          → JSON model list (13 models)
│   ├── GET /api/events          → JSON event list
│   ├── GET /api/prediction      → JSON prediction + price + benchmark data
│   ├── GET /api/prediction_comparison → JSON cross-model comparison
│   ├── GET /api/eval_metrics    → JSON IC/decay results (13 models)
│   ├── GET /api/sector_data     → JSON sector structure
│   ├── POST /api/analyze        → Live GPU inference (max 10K tokens)
│   └── @app.api("analyze")      → Gradio queue endpoint (ZeroGPU)
├── frontend_v2/
│   ├── index.html               → Welcome overlay + 4-tab terminal UI
│   ├── styles.css               → Terminal theme + 8 responsive breakpoints
│   └── app.js                   → Feed system, Plotly charts, tab logic
├── data/eval/
│   ├── predictions_v7d_grpo_clean.jsonl
│   ├── predictions_v7b_clean.jsonl
│   ├── predictions_v7c_clean.jsonl
│   ├── predictions_openreasoning7b_v7a.jsonl
│   ├── predictions_openreasoning7b_v6.jsonl
│   ├── predictions_openreasoning7b_v4.jsonl
│   ├── predictions_v8_sft_fixed.jsonl
│   ├── predictions_v8_grpo.jsonl
│   ├── predictions_manus_teacher_v2.jsonl
│   ├── predictions_codex_teacher.jsonl
│   ├── predictions_base_7b_fixed.jsonl
│   ├── predictions_base_14b_fixed.jsonl
│   ├── predictions_base_32b_fixed.jsonl
│   └── results_multi_model.json
└── data/market/*.parquet         → Price data (SPY + 10 quantum tickers)
```

The live deployment serves at [build-small-hackathon-quantum-alpha-intelligence.hf.space](https://build-small-hackathon-quantum-alpha-intelligence.hf.space), with the Nemotron-7B GRPO model (IC@5d = +0.157, p = 0.006) available for real time inference on ZeroGPU.
