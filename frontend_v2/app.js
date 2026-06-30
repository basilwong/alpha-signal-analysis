// Alpha Signal Analysis — Memory Agent Frontend
const API_BASE = window.location.origin;

// Welcome overlay dismiss
const welcomeOverlay = document.getElementById('welcome-overlay');
const welcomeBtn = document.getElementById('welcome-enter-btn');
if (welcomeBtn) {
    welcomeBtn.addEventListener('click', () => {
        welcomeOverlay.classList.add('hidden');
    });
}

// Tab navigation
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        if (btn.dataset.tab === 'memory') loadMemory();
        if (btn.dataset.tab === 'signals') loadSignals();
    });
});

// Update memory count in header
async function updateMemoryCount() {
    try {
        const resp = await fetch(`${API_BASE}/api/health`);
        const data = await resp.json();
        const stats = data.memory_stats || {};
        document.getElementById('memory-count').textContent = `${stats.knowledge_facts || 0} memories`;
    } catch(e) {}
}
updateMemoryCount();

// ============================================================
// TAB 1: CHAT
// ============================================================

document.getElementById('chat-send').addEventListener('click', sendMessage);
document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;

    const source = document.getElementById('chat-source').value;
    const btn = document.getElementById('chat-send');

    // Add user message
    addMessage('user', text);
    input.value = '';
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Thinking...';

    // Add typing indicator
    const typingId = addMessage('agent', '<span class="loading"></span> Analyzing with memory context...');

    try {
        const resp = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, source, enable_thinking: false })
        });
        const data = await resp.json();

        // Remove typing indicator
        document.getElementById(typingId).remove();

        // Build response
        let html = '';
        const signal = data.signal || {};
        const sv = signal.signal_vector || signal;

        // Chain of thought
        if (signal.chain_of_thought) {
            html += `<p>${signal.chain_of_thought}</p>`;
        }

        // Signal vector chart
        if (typeof sv === 'object' && Object.keys(sv).length > 0) {
            html += buildSignalCard(sv);
        }

        // Memory context used
        if (data.memory_context_used) {
            html += `<div class="memory-used">${data.memory_context_used}</div>`;
        }

        // Latency
        html += `<p style="font-size:10px;color:var(--text-muted);margin-top:8px;">Latency: ${data.latency_ms}ms | Memory: ${data.memory_stats?.knowledge_facts || 0} facts</p>`;

        addMessage('agent', html);
        updateMemoryCount();

    } catch(e) {
        document.getElementById(typingId)?.remove();
        addMessage('agent', `<p style="color:var(--accent-red);">Error: ${e.message}</p>`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">⚡</span> Analyze';
    }
}

function addMessage(role, content) {
    const messages = document.getElementById('chat-messages');
    const id = `msg-${Date.now()}`;
    const avatar = role === 'agent' ? '⚛' : '→';
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${content}</div>
    `;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return id;
}

function buildSignalCard(sv) {
    const entries = Object.entries(sv)
        .map(([ticker, data]) => {
            const score = typeof data === 'number' ? data : (data?.score || 0);
            return { ticker, score };
        })
        .filter(e => Math.abs(e.score) > 0.01)
        .sort((a, b) => b.score - a.score);

    if (entries.length === 0) return '';

    const maxScore = Math.max(...entries.map(e => Math.abs(e.score)), 1);
    let html = '<div class="signal-card"><h4>Signal Vector</h4>';
    for (const { ticker, score } of entries) {
        const width = Math.abs(score) / maxScore * 100;
        const cls = score > 0 ? 'bar-positive' : 'bar-negative';
        const color = score > 0 ? 'var(--accent-green)' : 'var(--accent-red)';
        html += `<div class="signal-bar">
            <span class="ticker">${ticker}</span>
            <span class="bar ${cls}" style="width:${width}%"></span>
            <span class="score" style="color:${color}">${score > 0 ? '+' : ''}${score.toFixed(2)}</span>
        </div>`;
    }
    html += '</div>';
    return html;
}

// ============================================================
// TAB 2: MEMORY TIMELINE
// ============================================================

async function loadMemory() {
    try {
        // Load stats
        const statsResp = await fetch(`${API_BASE}/api/health`);
        const statsData = await statsResp.json();
        const stats = statsData.memory_stats || {};
        document.getElementById('stat-knowledge').textContent = stats.knowledge_facts || 0;
        document.getElementById('stat-signals').textContent = stats.signals_stored || 0;
        document.getElementById('stat-accuracy').textContent = stats.accuracy ? `${(stats.accuracy * 100).toFixed(0)}%` : 'N/A';
        document.getElementById('stat-pending').textContent = stats.pending || 0;

        // Load knowledge
        const ticker = document.getElementById('knowledge-ticker-filter').value;
        const url = ticker ? `${API_BASE}/api/memory/knowledge?ticker=${ticker}` : `${API_BASE}/api/memory/knowledge`;
        const knResp = await fetch(url);
        const knData = await knResp.json();

        const list = document.getElementById('knowledge-list');
        list.innerHTML = '';
        for (const item of (knData.knowledge || [])) {
            list.innerHTML += `<div class="memory-item">
                <span class="ticker-badge">${item.ticker}</span>
                <span style="font-size:10px;color:var(--text-muted)">${item.fact_type}</span>
                <div class="fact-content">${item.content}</div>
                <div class="fact-meta">Source: ${item.source} | Confidence: ${(item.confidence || 1).toFixed(1)} | ${item.created_at?.split('T')[0] || ''}</div>
            </div>`;
        }
    } catch(e) {
        console.error('Failed to load memory:', e);
    }
}

document.getElementById('knowledge-ticker-filter').addEventListener('change', loadMemory);
document.getElementById('btn-refresh-memory').addEventListener('click', loadMemory);
document.getElementById('btn-forget').addEventListener('click', async () => {
    const resp = await fetch(`${API_BASE}/api/memory/forget`, { method: 'POST' });
    const data = await resp.json();
    alert(`Forgetting cycle complete: ${JSON.stringify(data.forgetting_result)}`);
    loadMemory();
    updateMemoryCount();
});

// ============================================================
// TAB 3: BEFORE/AFTER COMPARISON
// ============================================================

document.getElementById('btn-compare').addEventListener('click', async () => {
    const text = document.getElementById('compare-article').value.trim();
    if (!text) return;

    const btn = document.getElementById('btn-compare');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Running comparison...';

    try {
        // With memory (normal call)
        const withResp = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, source: 'news', enable_thinking: false })
        });
        const withData = await withResp.json();

        // Without memory (we simulate by noting the memory context)
        // For a true comparison, we'd need a separate endpoint. For now, show what memory was used.
        const withSignal = withData.signal?.signal_vector || withData.signal || {};
        const memoryContext = withData.memory_context_used || 'No memory available';
        const memCount = withData.memory_stats?.knowledge_facts || 0;

        document.getElementById('compare-memory-count').textContent = memCount;

        // Render with-memory chart
        renderCompareChart(withSignal, 'compare-with-chart');
        document.getElementById('compare-with-reasoning').textContent = withData.signal?.chain_of_thought || 'No reasoning provided';

        // For without-memory, show a note
        document.getElementById('compare-without-chart').innerHTML = '<p style="color:var(--text-muted);padding:20px;text-align:center;">Without-memory comparison requires a separate API call with empty memory context. The key difference is shown in the "Key Differences" section below.</p>';
        document.getElementById('compare-without-reasoning').textContent = 'Without memory, the model would analyze this article in isolation without knowledge of previous events, roadmap targets, or competitive dynamics.';

        // Diff
        document.getElementById('compare-diff-content').innerHTML = `
            <p><strong>Memory context injected:</strong></p>
            <div class="memory-used">${memoryContext}</div>
            <p style="margin-top:8px;color:var(--text-secondary);">With memory, the model references specific facts (qubit counts, roadmap targets, past signals) that it would not have access to in a cold-start scenario. This enables more precise scoring and better-calibrated confidence.</p>
        `;

        document.getElementById('compare-results').classList.remove('hidden');
    } catch(e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">⚡</span> Compare With/Without Memory';
    }
});

function renderCompareChart(sv, containerId) {
    const entries = Object.entries(sv)
        .map(([ticker, data]) => ({ ticker, score: typeof data === 'number' ? data : (data?.score || 0) }))
        .sort((a, b) => b.score - a.score);

    const trace = {
        x: entries.map(e => e.score),
        y: entries.map(e => e.ticker),
        type: 'bar',
        orientation: 'h',
        marker: { color: entries.map(e => e.score >= 0 ? '#00ff88' : '#ff4444') }
    };
    const layout = {
        margin: { l: 50, r: 20, t: 10, b: 30 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#808080', size: 10 },
        xaxis: { gridcolor: '#2a2a2a', zerolinecolor: '#2a2a2a', range: [-2.5, 2.5] },
        yaxis: { gridcolor: '#2a2a2a' },
        height: 180
    };
    Plotly.newPlot(containerId, [trace], layout, { displayModeBar: false });
}

// ============================================================
// TAB 4: SIGNAL HISTORY
// ============================================================

async function loadSignals() {
    try {
        const resp = await fetch(`${API_BASE}/api/memory/signals`);
        const data = await resp.json();
        const list = document.getElementById('signals-list');
        list.innerHTML = '';

        for (const sig of (data.signals || [])) {
            const sv = sig.signal_vector || {};
            const chips = Object.entries(sv)
                .map(([t, d]) => {
                    const score = typeof d === 'number' ? d : (d?.score || 0);
                    if (Math.abs(score) < 0.05) return '';
                    const cls = score > 0 ? 'positive' : 'negative';
                    return `<span class="score-chip ${cls}">${t} ${score > 0 ? '+' : ''}${score.toFixed(1)}</span>`;
                })
                .filter(Boolean)
                .join('');

            list.innerHTML += `<div class="signal-item">
                <div class="signal-header">
                    <span class="signal-title">${sig.title || 'Untitled'}</span>
                    <span class="signal-date">${sig.date || ''}</span>
                </div>
                <div class="signal-scores">${chips || '<span class="score-chip neutral">No signals</span>'}</div>
            </div>`;
        }

        if (!data.signals?.length) {
            list.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px;">No signals generated yet. Use the Chat tab to analyze articles.</p>';
        }
    } catch(e) {
        console.error('Failed to load signals:', e);
    }
}
