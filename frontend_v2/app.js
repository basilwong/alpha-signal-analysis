// Quantum Alpha Intelligence — Frontend Application

const API_BASE = '';  // Same origin

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
    'V7d GRPO (best)': '#10b981',
    'V7b Rejection Sampling': '#22c55e',
    'V7c DPO': '#8b5cf6',
    'V4 Baseline (LoRA)': '#f59e0b',
    'Manus Teacher': '#06b6d4',
    'Qwen3-8B Base': '#ef4444',
    'Qwen3.7-Max Base': '#f97316',
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

document.getElementById('live-analyze-btn').addEventListener('click', async () => {
    const text = document.getElementById('live-input').value.trim();
    if (!text) return;

    const btn = document.getElementById('live-analyze-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Analyzing...';

    const model = document.getElementById('live-model').value;
    const source = document.getElementById('live-source').value;
    const thinking = document.getElementById('live-thinking').checked;

    try {
        const resp = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, source, model, enable_thinking: thinking }),
        });
        const data = await resp.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }

        const results = document.getElementById('live-results');
        results.classList.remove('hidden');

        // Latency
        document.getElementById('live-latency').textContent = `${data.latency_ms}ms`;

        // Signal chart
        const signal = data.signal || {};
        const sv = signal.signal_vector || {};
        createSignalChart(sv, 'live-signal-chart', 'Signal Vector');

        // Meta
        document.getElementById('live-event-type').textContent = signal.event_type || '--';
        document.getElementById('live-time-horizon').textContent = signal.time_horizon || '--';
        document.getElementById('live-signal-decay').textContent = signal.signal_decay || '--';
        document.getElementById('live-novelty').textContent = signal.information_novelty || '--';

        // Translation
        document.getElementById('live-translation').textContent = signal.technical_translation || 'N/A';

        // Thinking
        document.getElementById('live-thinking-output').textContent = data.thinking || '(Thinking disabled)';

        // JSON
        document.getElementById('live-json').textContent = JSON.stringify(signal, null, 2);

    } catch (e) {
        alert(`Request failed: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">⚡</span> Analyze';
    }
});

// ============================================================
// TAB 2: HISTORICAL PREDICTIONS
// ============================================================

let allEvents = [];

async function initHistoricalTab() {
    // Load events from the first available model
    const resp = await fetch(`${API_BASE}/api/events?model=V7d GRPO (best)`);
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
    // Get comparison across all models
    const resp = await fetch(`${API_BASE}/api/prediction_comparison?article_idx=${articleIdx}`);
    const data = await resp.json();

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

    // Price chart (get from first model's prediction)
    if (models.length > 0) {
        const firstModel = models[0];
        const predResp = await fetch(`${API_BASE}/api/prediction?model=${encodeURIComponent(firstModel)}&idx=0`);
        // Find the correct idx for this model
        const eventsResp = await fetch(`${API_BASE}/api/events?model=${encodeURIComponent(firstModel)}`);
        const eventsData = await eventsResp.json();
        const matchIdx = eventsData.events.findIndex(e => e.article_idx === articleIdx);
        if (matchIdx >= 0) {
            const predData = await (await fetch(`${API_BASE}/api/prediction?model=${encodeURIComponent(firstModel)}&idx=${matchIdx}`)).json();
            const priceData = predData.price_data || {};
            const priceTraces = Object.entries(priceData).slice(0, 5).map(([ticker, pd]) => {
                if (!pd.values || pd.values.length === 0) return null;
                const basePrice = pd.values[0];
                const returns = pd.values.map(v => ((v - basePrice) / basePrice) * 100);
                return {
                    name: ticker,
                    type: 'scatter',
                    mode: 'lines',
                    x: pd.dates,
                    y: returns,
                    line: { width: 2 },
                };
            }).filter(Boolean);

            if (priceTraces.length > 0) {
                Plotly.newPlot('hist-price-chart', priceTraces, {
                    title: { text: 'Cumulative Return (%) After Event', font: { color: '#e2e8f0', size: 14 } },
                    xaxis: { color: '#94a3b8', gridcolor: '#1e2a3a' },
                    yaxis: { title: 'Return (%)', color: '#94a3b8', gridcolor: '#1e2a3a', zerolinecolor: '#3b82f6' },
                    plot_bgcolor: '#0f1629',
                    paper_bgcolor: '#1a2035',
                    legend: { font: { color: '#e2e8f0' } },
                    margin: { l: 50, r: 20, t: 40, b: 40 },
                    height: 300,
                }, { responsive: true });
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
    Object.keys(evalData).forEach(modelName => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = modelName;
        checkbox.checked = true;
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
        legend: { font: { color: '#e2e8f0' }, x: 0.02, y: 0.98 },
        margin: { l: 60, r: 20, t: 50, b: 50 },
        height: 400,
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
        await initLiveTab();
        await initHistoricalTab();
        await initEvalTab();
        await initSectorTab();
    } catch (e) {
        console.error('Init error:', e);
    }
}

init();
