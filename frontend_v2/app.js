// Alpha Signal Analysis — Frontend Application

const API_BASE = '';  // Same origin

// ============================================================
// WELCOME OVERLAY
// ============================================================

document.getElementById('welcome-enter-btn').addEventListener('click', () => {
    const overlay = document.getElementById('welcome-overlay');
    overlay.style.opacity = '0';
    overlay.style.transition = 'opacity 0.3s ease';
    setTimeout(() => {
        overlay.classList.add('hidden');
        overlay.style.opacity = '';
    }, 300);
});

document.getElementById('show-welcome-btn').addEventListener('click', () => {
    const overlay = document.getElementById('welcome-overlay');
    overlay.classList.remove('hidden');
    overlay.scrollTop = 0;
});

// ============================================================
// TAB NAVIGATION
// ============================================================

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    });
});

// Expandable sections
document.querySelectorAll('.expand-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const target = document.getElementById(btn.dataset.target);
        target.classList.toggle('open');
        btn.textContent = target.classList.contains('open')
            ? btn.textContent.replace('▾', '▴')
            : btn.textContent.replace('▴', '▾');
    });
});

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

const MODEL_COLORS = {
    // Fine-tuned (Manus Teacher)
    'Nemotron-7B (SFT + GRPO, Manus Teacher)': '#10b981',
    'Nemotron-7B (Best-of-4 SFT, Manus Teacher)': '#22c55e',
    'Nemotron-7B (SFT + DPO, Manus Teacher)': '#8b5cf6',
    'Nemotron-7B (SFT + Thinking, Manus Teacher)': '#a78bfa',
    'Nemotron-7B (SFT + Bearish, Manus Teacher)': '#f472b6',
    'Nemotron-7B (SFT, Manus Teacher)': '#f59e0b',
    // Fine-tuned (GPT-5.5 Teacher)
    'Nemotron-7B (SFT, GPT-5.5 Teacher)': '#fb923c',
    'Nemotron-7B (SFT + GRPO, GPT-5.5 Teacher)': '#84cc16',
    // Teachers
    'Manus (Teacher, Direct)': '#06b6d4',
    'GPT-5.5 (Teacher, Direct)': '#3b82f6',
    // Base models
    'Nemotron-7B (Base, No Fine-Tuning)': '#ef4444',
    'Nemotron-14B (Base, No Fine-Tuning)': '#dc2626',
    'Nemotron-32B (Base, No Fine-Tuning)': '#991b1b',
};

function getModelColor(name) {
    return MODEL_COLORS[name] || '#64748b';
}

function createSignalChart(signalVector, containerId, title = 'Signal Vector') {
    const tickers = Object.keys(signalVector).sort((a, b) => {
        const scoreA = signalVector[a]?.score || 0;
        const scoreB = signalVector[b]?.score || 0;
        return scoreA - scoreB;
    });

    const scores = tickers.map(t => signalVector[t]?.score || 0);
    const colors = scores.map(s => s >= 0 ? '#10b981' : '#ef4444');
    const hoverText = tickers.map(t => {
        const entry = signalVector[t] || {};
        return `${t}: ${(entry.score || 0).toFixed(3)}<br>${entry.reasoning || ''}`;
    });

    Plotly.newPlot(containerId, [{
        type: 'bar',
        y: tickers,
        x: scores,
        orientation: 'h',
        marker: { color: colors },
        hovertext: hoverText,
        hoverinfo: 'text',
    }], {
        title: { text: title, font: { color: '#e2e8f0', size: 14 } },
        xaxis: { title: 'Score', color: '#94a3b8', gridcolor: '#1e2a3a', zerolinecolor: '#3b82f6' },
        yaxis: { color: '#e2e8f0' },
        plot_bgcolor: '#0f1629',
        paper_bgcolor: '#1a2035',
        margin: { l: 60, r: 20, t: 40, b: 40 },
        height: 300,
    }, { responsive: true });
}

// ============================================================
// TAB 1: LIVE ANALYSIS
// ============================================================

let feedCount = 0;

async function initLiveTab() {
    const resp = await fetch(`${API_BASE}/api/models`);
    const data = await resp.json();
    const select = document.getElementById('live-model');
    select.innerHTML = '';
    data.models.filter(m => m.live_inference).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name}`;
        select.appendChild(opt);
    });
}

function updateFeedCount() {
    document.getElementById('feed-count').textContent = `${feedCount} ${feedCount === 1 ? 'analysis' : 'analyses'}`;
}

function createFeedCard(id, model, source, textPreview) {
    const feed = document.getElementById('live-feed');
    // Remove empty state
    const empty = feed.querySelector('.feed-empty');
    if (empty) empty.remove();

    const card = document.createElement('div');
    card.className = 'feed-card';
    card.id = `feed-card-${id}`;
    card.innerHTML = `
        <div class="feed-card-header">
            <div class="feed-card-meta">
                <span class="feed-model-badge">${model}</span>
                <span class="feed-source-badge">${source}</span>
                <span class="feed-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <span class="feed-status loading-status">Analyzing...</span>
        </div>
        <div class="feed-card-preview">${textPreview}</div>
        <div class="feed-card-body" id="feed-body-${id}"></div>
    `;
    feed.prepend(card);
    return card;
}

function renderFeedResult(id, data) {
    const card = document.getElementById(`feed-card-${id}`);
    if (!card) return;

    const status = card.querySelector('.feed-status');
    const body = document.getElementById(`feed-body-${id}`);

    if (data.error) {
        status.textContent = 'Error';
        status.className = 'feed-status error-status';
        body.innerHTML = `<div class="feed-error">${data.error}</div>`;
        return;
    }

    status.textContent = `${data.latency_ms}ms`;
    status.className = 'feed-status success-status';

    const signal = data.signal || {};

    // Handle signal parse errors from backend
    if (signal.error) {
        body.innerHTML = `<div class="feed-error">${signal.error}</div>
            ${signal.raw ? `<details class="feed-details"><summary>Raw Output</summary><pre class="feed-json">${signal.raw}</pre></details>` : ''}`;
        return;
    }

    // Handle both formats: {signal_vector: {IONQ: {score: ...}}} and flat {IONQ: 1.5, ...}
    let sv = signal.signal_vector || {};
    const possibleTickers = ['IONQ', 'RGTI', 'QBTS', 'QUBT', 'IBM', 'GOOGL', 'MSFT', 'HON', 'NVDA'];
    if (typeof sv === 'object' && !Array.isArray(sv)) {
        // Normalize: if values are plain numbers, wrap them
        Object.keys(sv).forEach(t => {
            if (typeof sv[t] === 'number') {
                sv[t] = { score: sv[t] };
            }
        });
    }
    if (Object.keys(sv).length === 0) {
        // Try flat format (V4 model returns {IONQ: 1.5, RGTI: 0.8, ...})
        const flatKeys = Object.keys(signal).filter(k => possibleTickers.includes(k));
        if (flatKeys.length > 0) {
            flatKeys.forEach(t => { sv[t] = { score: signal[t] }; });
        }
    }
    const tickers = Object.keys(sv).filter(t => possibleTickers.includes(t)).sort((a, b) => (sv[b]?.score || 0) - (sv[a]?.score || 0));

    // Check if all scores are zero (no signal)
    const allZero = tickers.every(t => (sv[t]?.score || 0) === 0);

    // Build inline signal bars
    const signalBars = tickers.map(t => {
        const score = sv[t]?.score || 0;
        const color = score >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
        const width = Math.min(Math.abs(score) * 40, 80);
        return `<div class="feed-signal-row">
            <span class="feed-ticker">${t}</span>
            <div class="feed-bar-container">
                <div class="feed-bar" style="width:${width}%;background:${color};${score < 0 ? 'margin-left:auto;' : ''}"></div>
            </div>
            <span class="feed-score" style="color:${color}">${score >= 0 ? '+' : ''}${score.toFixed(2)}</span>
        </div>`;
    }).join('');

    const meta = [
        signal.event_type ? `Event: ${signal.event_type}` : '',
        signal.time_horizon ? `Horizon: ${signal.time_horizon}` : '',
        signal.signal_decay ? `Decay: ${signal.signal_decay}` : '',
        signal.information_novelty ? `Novelty: ${signal.information_novelty}` : '',
    ].filter(Boolean).join(' | ');

    // Get rationale text (model uses different field names)
    const rationale = signal.signal_rationale || signal.signal_reasoning || signal.technical_translation || '';

    body.innerHTML = `
        ${allZero ? '<div class="feed-no-signal">NO SIGNAL — Model determined no actionable trading signal from this input.</div>' : ''}
        <div class="feed-signal-chart">${signalBars}</div>
        ${meta ? `<div class="feed-meta-line">${meta}</div>` : ''}
        ${rationale ? `<div class="feed-translation">${rationale}</div>` : ''}
        <details class="feed-details">
            <summary>Raw JSON</summary>
            <pre class="feed-json">${JSON.stringify(signal, null, 2)}</pre>
        </details>
        ${data.thinking ? `<details class="feed-details"><summary>Thinking Trace</summary><pre class="feed-json">${data.thinking}</pre></details>` : ''}
    `;
}

document.getElementById('live-analyze-btn').addEventListener('click', () => {
    const text = document.getElementById('live-input').value.trim();
    if (!text) return;

    const model = document.getElementById('live-model').value;
    const source = document.getElementById('live-source').value;
    const thinking = document.getElementById('live-thinking').checked;

    feedCount++;
    updateFeedCount();
    const id = feedCount;
    const preview = text.length > 150 ? text.substring(0, 150) + '...' : text;
    createFeedCard(id, model, source, preview);

    // Clear input for next analysis
    document.getElementById('live-input').value = '';

    // Fire and forget (non-blocking)
    fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source, model, enable_thinking: thinking }),
    })
    .then(r => {
        if (!r.ok) {
            return r.text().then(t => { throw new Error(`Server error (${r.status}): ${t.substring(0, 100)}`); });
        }
        return r.json();
    })
    .then(data => renderFeedResult(id, data))
    .catch(e => renderFeedResult(id, { error: e.message }));
});

document.getElementById('feed-clear-btn').addEventListener('click', () => {
    feedCount = 0;
    updateFeedCount();
    document.getElementById('live-feed').innerHTML = '<div class="feed-empty">No analyses yet. Paste an article above and click Analyze.</div>';
});

// ============================================================
// TAB 2: HISTORICAL PREDICTIONS
// ============================================================

let allEvents = [];

async function initHistoricalTab() {
    // Load events from the first available model
    const resp = await fetch(`${API_BASE}/api/events?model=Nemotron-7B (SFT, Manus Teacher)`);
    const data = await resp.json();
    allEvents = data.events || [];

    const select = document.getElementById('hist-event-select');
    select.innerHTML = '';
    allEvents.forEach(e => {
        const opt = document.createElement('option');
        opt.value = e.article_idx;
        opt.textContent = `${e.date} | ${e.title}`;
        select.appendChild(opt);
    });

    if (allEvents.length > 0) {
        loadHistoricalEvent(allEvents[0].article_idx);
    }
}

document.getElementById('hist-event-select').addEventListener('change', (e) => {
    loadHistoricalEvent(parseInt(e.target.value));
});

async function loadHistoricalEvent(articleIdx) {
    // Show loading state
    const resultsPanel = document.getElementById('hist-results');
    resultsPanel.style.opacity = '0.5';

    // Get comparison across all models
    const resp = await fetch(`${API_BASE}/api/prediction_comparison?article_idx=${articleIdx}`);
    const data = await resp.json();

    resultsPanel.style.opacity = '1';

    // Find article info
    const event = allEvents.find(e => e.article_idx === articleIdx);
    const infoDiv = document.getElementById('hist-article-info');
    if (event) {
        infoDiv.innerHTML = `<div class="title">${event.title}</div><div class="meta">${event.date} | ${event.source}</div>`;
    }

    // Build comparison chart (grouped bar)
    const models = Object.keys(data.models);
    const tickers = ['IONQ', 'RGTI', 'QBTS', 'QUBT', 'IBM', 'GOOGL', 'MSFT', 'HON', 'NVDA'];

    const traces = models.map(modelName => {
        const signal = data.models[modelName]?.signal?.signal_vector || {};
        return {
            name: modelName,
            type: 'bar',
            y: tickers,
            x: tickers.map(t => signal[t]?.score || 0),
            orientation: 'h',
            marker: { color: getModelColor(modelName) },
        };
    });

    Plotly.newPlot('hist-comparison-chart', traces, {
        title: { text: 'Signal Comparison Across Models', font: { color: '#e2e8f0', size: 14 } },
        barmode: 'group',
        xaxis: { title: 'Score', color: '#94a3b8', gridcolor: '#1e2a3a', zerolinecolor: '#3b82f6' },
        yaxis: { color: '#e2e8f0' },
        plot_bgcolor: '#0f1629',
        paper_bgcolor: '#1a2035',
        legend: { font: { color: '#e2e8f0' } },
        margin: { l: 60, r: 20, t: 40, b: 40 },
        height: 350,
    }, { responsive: true });

    // Price charts (raw, market-adjusted, sector-adjusted)
    // Use a model with the most predictions for reliable price data lookup
    const priceModel = 'Nemotron-7B (SFT, Manus Teacher)';
    if (models.length > 0) {
        const eventsResp = await fetch(`${API_BASE}/api/events?model=${encodeURIComponent(priceModel)}`);
        const eventsData = await eventsResp.json();
        const matchIdx = eventsData.events.findIndex(e => e.article_idx === articleIdx);
        if (matchIdx >= 0) {
            const predData = await (await fetch(`${API_BASE}/api/prediction?model=${encodeURIComponent(priceModel)}&idx=${matchIdx}`)).json();
            const priceData = predData.price_data || {};
            const benchmarkData = predData.benchmark_data || {};
            const purePlayTickers = ['IONQ', 'RGTI', 'QBTS', 'QUBT'];

            // Compute returns for each ticker
            const tickerReturns = {};
            Object.entries(priceData).forEach(([ticker, pd]) => {
                if (pd.values && pd.values.length > 0) {
                    const basePrice = pd.values[0];
                    tickerReturns[ticker] = {
                        dates: pd.dates,
                        returns: pd.values.map(v => ((v - basePrice) / basePrice) * 100),
                    };
                }
            });

            // Compute SPY benchmark returns
            let spyReturns = null;
            if (benchmarkData.SPY && benchmarkData.SPY.values.length > 0) {
                const spyBase = benchmarkData.SPY.values[0];
                spyReturns = {
                    dates: benchmarkData.SPY.dates,
                    returns: benchmarkData.SPY.values.map(v => ((v - spyBase) / spyBase) * 100),
                };
            }

            // Compute sector basket returns (equal-weight of pure-play quantum tickers)
            let sectorReturns = null;
            const sectorTickers = purePlayTickers.filter(t => tickerReturns[t]);
            if (sectorTickers.length > 0) {
                const maxLen = Math.min(...sectorTickers.map(t => tickerReturns[t].returns.length));
                const avgReturns = [];
                for (let i = 0; i < maxLen; i++) {
                    const avg = sectorTickers.reduce((sum, t) => sum + tickerReturns[t].returns[i], 0) / sectorTickers.length;
                    avgReturns.push(avg);
                }
                sectorReturns = {
                    dates: tickerReturns[sectorTickers[0]].dates.slice(0, maxLen),
                    returns: avgReturns,
                };
            }

            const chartLayout = (title, yTitle) => ({
                title: { text: title, font: { color: '#e2e8f0', size: 14 } },
                xaxis: { color: '#94a3b8', gridcolor: '#1e2a3a' },
                yaxis: { title: yTitle, color: '#94a3b8', gridcolor: '#1e2a3a', zerolinecolor: '#3b82f6' },
                plot_bgcolor: '#0f1629',
                paper_bgcolor: '#1a2035',
                legend: { font: { color: '#e2e8f0', size: 11 }, orientation: 'h', x: 0, y: -0.25, xanchor: 'left', yanchor: 'top' },
                margin: { l: 50, r: 20, t: 40, b: 80 },
                height: 320,
            });

            // Chart 1: Raw cumulative returns (+ SPY as dashed line)
            const rawTraces = Object.entries(tickerReturns).slice(0, 5).map(([ticker, tr]) => ({
                name: ticker,
                type: 'scatter',
                mode: 'lines',
                x: tr.dates,
                y: tr.returns,
                line: { width: 2 },
            }));
            if (spyReturns) {
                rawTraces.push({
                    name: 'SPY (Market)',
                    type: 'scatter',
                    mode: 'lines',
                    x: spyReturns.dates,
                    y: spyReturns.returns,
                    line: { width: 2, dash: 'dash', color: '#64748b' },
                });
            }
            if (rawTraces.length > 0) {
                Plotly.newPlot('hist-price-chart', rawTraces,
                    chartLayout('Cumulative Return (%) | Dashed = SPY Benchmark', 'Return (%)'),
                    { responsive: true });
            }

            // Chart 2: Abnormal returns vs market (stock return - SPY return)
            if (spyReturns) {
                const abnormalTraces = Object.entries(tickerReturns).slice(0, 5).map(([ticker, tr]) => {
                    const maxLen = Math.min(tr.returns.length, spyReturns.returns.length);
                    const abnormal = tr.returns.slice(0, maxLen).map((r, i) => r - spyReturns.returns[i]);
                    return {
                        name: ticker,
                        type: 'scatter',
                        mode: 'lines',
                        x: tr.dates.slice(0, maxLen),
                        y: abnormal,
                        line: { width: 2 },
                    };
                });
                Plotly.newPlot('hist-abnormal-chart', abnormalTraces,
                    chartLayout('Abnormal Return (%) = Stock Return \u2212 SPY Return', 'Abnormal Return (%)'),
                    { responsive: true });
            }

            // Chart 3: Abnormal returns vs sector basket
            if (sectorReturns) {
                const sectorAbnormalTraces = Object.entries(tickerReturns).slice(0, 5).map(([ticker, tr]) => {
                    const maxLen = Math.min(tr.returns.length, sectorReturns.returns.length);
                    const abnormal = tr.returns.slice(0, maxLen).map((r, i) => r - sectorReturns.returns[i]);
                    return {
                        name: ticker,
                        type: 'scatter',
                        mode: 'lines',
                        x: tr.dates.slice(0, maxLen),
                        y: abnormal,
                        line: { width: 2 },
                    };
                });
                Plotly.newPlot('hist-sector-abnormal-chart', sectorAbnormalTraces,
                    chartLayout('Abnormal Return (%) = Stock Return \u2212 Quantum Sector Avg', 'Abnormal Return (%)'),
                    { responsive: true });
            }
        }
    }

    // Model details
    const detailsDiv = document.getElementById('hist-model-details');
    detailsDiv.innerHTML = models.map(modelName => {
        const signal = data.models[modelName]?.signal || {};
        const time = data.models[modelName]?.time_seconds || 0;
        return `<div class="model-detail-card">
            <h4>${modelName} <span style="color:var(--text-muted);font-size:11px;">(${time.toFixed(1)}s)</span></h4>
            <p style="font-size:12px;color:var(--text-secondary);margin-top:4px;">${signal.signal_rationale || signal.technical_translation || 'No rationale available'}</p>
        </div>`;
    }).join('');
}

// ============================================================
// TAB 3: EVALUATION DASHBOARD
// ============================================================

let evalData = {};

async function initEvalTab() {
    const resp = await fetch(`${API_BASE}/api/eval_metrics`);
    evalData = await resp.json();

    const container = document.getElementById('eval-model-checkboxes');
    container.innerHTML = '';
    // Only show the best model and teacher models checked by default
    const defaultChecked = [
        'Nemotron-7B (SFT + GRPO, Manus Teacher)',
        'Manus (Teacher, Direct)',
        'GPT-5.5 (Teacher, Direct)',
    ];
    Object.keys(evalData).forEach(modelName => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = modelName;
        checkbox.checked = defaultChecked.includes(modelName);
        checkbox.addEventListener('change', updateEvalCharts);
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(` ${modelName}`));
        container.appendChild(label);
    });

    updateEvalCharts();
}

function updateEvalCharts() {
    const checked = [...document.querySelectorAll('#eval-model-checkboxes input:checked')].map(c => c.value);

    if (checked.length === 0) {
        document.getElementById('eval-summary').textContent = 'Select at least one model.';
        return;
    }

    // Decay chart
    const traces = checked.map(modelName => {
        const results = evalData[modelName] || {};
        const decay = results.decay_curve || [];
        return {
            name: modelName,
            type: 'scatter',
            mode: 'lines+markers',
            x: decay.map(d => d.horizon),
            y: decay.map(d => d.ic),
            line: { color: getModelColor(modelName), width: 2 },
            marker: {
                size: 10,
                color: decay.map(d => d.p_value < 0.1 ? getModelColor(modelName) : '#555555'),
                line: { width: 1, color: 'white' },
            },
            hovertext: decay.map(d => `${modelName}<br>IC=${d.ic?.toFixed(4)}<br>p=${d.p_value?.toFixed(4)}`),
            hoverinfo: 'text',
        };
    });

    traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [0, 25],
        y: [0, 0],
        line: { color: '#64748b', dash: 'dash', width: 1 },
        showlegend: false,
    });

    Plotly.newPlot('eval-decay-chart', traces, {
        title: { text: 'Signal Decay (IC by Holding Period) | Bright = p<0.10', font: { color: '#e2e8f0', size: 14 } },
        xaxis: { title: 'Holding Period (Days)', color: '#94a3b8', gridcolor: '#1e2a3a' },
        yaxis: { title: 'Information Coefficient', color: '#94a3b8', gridcolor: '#1e2a3a' },
        plot_bgcolor: '#0f1629',
        paper_bgcolor: '#1a2035',
        legend: { font: { color: '#e2e8f0', size: 11 }, orientation: 'h', x: 0, y: -0.25, xanchor: 'left', yanchor: 'top' },
        margin: { l: 60, r: 20, t: 50, b: 100 },
        height: 450,
    }, { responsive: true });

    // Table
    let tableHTML = '<table><thead><tr><th>Model</th><th>IC +1d</th><th>IC +5d</th><th>IC +10d</th><th>IC +20d</th><th>Dir Acc</th><th>N</th></tr></thead><tbody>';
    checked.forEach(modelName => {
        const results = evalData[modelName] || {};
        const decay = {};
        (results.decay_curve || []).forEach(d => { decay[d.horizon] = d; });
        const acc = results.direction_accuracy_5d?.accuracy;

        const formatIC = (h) => {
            const d = decay[h];
            if (!d) return '<td>--</td>';
            const cls = d.ic > 0 ? 'positive' : 'negative';
            const sig = d.p_value < 0.05 ? ' significant' : '';
            return `<td class="${cls}${sig}">${d.ic >= 0 ? '+' : ''}${d.ic?.toFixed(4)}</td>`;
        };

        tableHTML += `<tr><td style="color:${getModelColor(modelName)};font-weight:600;">${modelName}</td>`;
        tableHTML += formatIC(1) + formatIC(5) + formatIC(10) + formatIC(20);
        tableHTML += `<td>${acc ? (acc * 100).toFixed(1) + '%' : '--'}</td>`;
        tableHTML += `<td>${results.n_predictions || '--'}</td></tr>`;
    });
    tableHTML += '</tbody></table>';
    document.getElementById('eval-table').innerHTML = tableHTML;

    // Summary
    const best = checked.reduce((best, m) => {
        const ic5 = (evalData[m]?.decay_curve || []).find(d => d.horizon === 5)?.ic || 0;
        return ic5 > (best.ic || 0) ? { name: m, ic: ic5 } : best;
    }, { name: '', ic: 0 });
    document.getElementById('eval-summary').innerHTML = `Comparing <strong>${checked.length}</strong> model(s). Best IC at +5d: <strong style="color:${getModelColor(best.name)}">${best.name}</strong> (IC=${best.ic?.toFixed(4)})`;
}

// ============================================================
// TAB 4: SECTOR MAP
// ============================================================

async function initSectorTab() {
    const resp = await fetch(`${API_BASE}/api/sector_data`);
    const data = await resp.json();

    // Signal weight chart
    const tickers = Object.entries(data.tickers)
        .sort((a, b) => b[1].signal_weight - a[1].signal_weight);

    const clusterColors = {
        'Trapped Ion': '#10b981',
        'Superconducting': '#3b82f6',
        'Annealing': '#f59e0b',
        'Topological': '#eab308',
        'Neutral Atom': '#8b5cf6',
        'Adjacent': '#06b6d4',
    };

    Plotly.newPlot('sector-weight-chart', [{
        type: 'bar',
        y: tickers.map(([t, d]) => `${t} (${d.name})`),
        x: tickers.map(([t, d]) => d.signal_weight * 100),
        orientation: 'h',
        marker: { color: tickers.map(([t, d]) => clusterColors[d.cluster] || '#64748b') },
        text: tickers.map(([t, d]) => `${(d.signal_weight * 100).toFixed(0)}%`),
        textposition: 'outside',
        hovertext: tickers.map(([t, d]) => `${t}<br>${d.name}<br>Cluster: ${d.cluster}<br>Weight: ${(d.signal_weight * 100).toFixed(0)}%`),
        hoverinfo: 'text',
    }], {
        title: { text: 'Signal Weight (how much quantum news moves each stock)', font: { color: '#e2e8f0', size: 14 } },
        xaxis: { title: 'Signal Weight (%)', color: '#94a3b8', gridcolor: '#1e2a3a', range: [0, 110] },
        yaxis: { color: '#e2e8f0' },
        plot_bgcolor: '#0f1629',
        paper_bgcolor: '#1a2035',
        margin: { l: 140, r: 60, t: 40, b: 40 },
        height: 350,
    }, { responsive: true });

    // Cluster cards
    const clustersDiv = document.getElementById('sector-clusters');
    clustersDiv.innerHTML = Object.entries(data.clusters).map(([cluster, tickerList]) => `
        <div class="cluster-card">
            <h4 style="color:${clusterColors[cluster] || '#64748b'}">${cluster}</h4>
            <div class="ticker-list">
                ${tickerList.map(t => `<span class="ticker-tag">${t}</span>`).join('')}
            </div>
        </div>
    `).join('');

    // Dynamics table
    const dynamicsDiv = document.getElementById('sector-dynamics-table');
    let html = '<table class="table-container"><thead><tr><th>Trigger Event</th><th>Bullish</th><th>Bearish</th></tr></thead><tbody>';
    data.dynamics.forEach(d => {
        html += `<tr><td>${d.trigger}</td><td class="positive">${d.bullish.join(', ')}</td><td class="negative">${d.bearish.join(', ') || 'None'}</td></tr>`;
    });
    html += '</tbody></table>';
    dynamicsDiv.innerHTML = html;
}

// ============================================================
// INITIALIZATION
// ============================================================

async function init() {
    try {
        await initEvalTab();
        await initLiveTab();
        await initHistoricalTab();
        await initSectorTab();
    } catch (e) {
        console.error('Init error:', e);
    }
}

init();
