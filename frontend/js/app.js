/**
 * Quantum Alpha Intelligence - Frontend Application
 * 
 * Connects to the Gradio Server backend via @gradio/client
 * to fetch signals, briefings, and run analysis.
 */

import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

let client = null;

// Initialize the Gradio client connection
async function initClient() {
    try {
        client = await Client.connect(window.location.origin);
        document.getElementById("last-updated").textContent = "Connected";
        console.log("Connected to Quantum Alpha backend");
        await loadDashboard();
    } catch (error) {
        console.error("Failed to connect:", error);
        document.getElementById("last-updated").textContent = "Connection failed";
    }
}

// Load all dashboard panels
async function loadDashboard() {
    await Promise.all([
        loadSectorOverview(),
        loadSignals(),
        loadBriefing(),
        renderHeatmap(),
    ]);
    updateTimestamp();
}

// Fetch and render sector overview
async function loadSectorOverview() {
    try {
        const result = await client.predict("/get_sector_overview");
        const data = result.data[0];
        renderSectorPulse(data);
    } catch (error) {
        console.error("Failed to load sector overview:", error);
    }
}

// Fetch and render live signals
async function loadSignals() {
    try {
        const result = await client.predict("/get_signals", { ticker: "all" });
        const data = result.data[0];
        renderSignalFeed(data.signals);
    } catch (error) {
        console.error("Failed to load signals:", error);
    }
}

// Fetch and render daily briefing
async function loadBriefing() {
    try {
        const result = await client.predict("/get_briefing", { date: "today" });
        const briefing = result.data[0];
        document.getElementById("briefing-content").innerHTML = 
            `<p>${briefing}</p>`;
    } catch (error) {
        console.error("Failed to load briefing:", error);
    }
}

// Render the sector sentiment gauge
function renderSectorPulse(data) {
    const container = document.getElementById("sector-sentiment");
    const sentimentColor = getSentimentColor(data.sector_sentiment);
    container.innerHTML = `
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 48px; font-weight: 700; color: ${sentimentColor};">
                ${data.sector_sentiment.toUpperCase()}
            </div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">
                ${data.signal_count_24h} signals in last 24h
            </div>
        </div>
    `;
}

// Render the signal feed
function renderSignalFeed(signals) {
    const container = document.getElementById("signal-list");
    if (!signals || signals.length === 0) {
        container.innerHTML = `
            <div class="signal-entry">
                <span class="event-type">AWAITING DATA</span>
                <p class="summary">Signal pipeline initializing. Signals will appear here as they are processed.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = signals.map(signal => `
        <div class="signal-entry ${signal.sentiment}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="ticker">${signal.ticker}</span>
                <span class="event-type">${signal.event_type}</span>
            </div>
            <p class="summary">${signal.summary}</p>
            <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                <span style="font-size: 11px; color: var(--text-muted);">
                    Confidence: ${(signal.confidence * 100).toFixed(0)}%
                </span>
                <span style="font-size: 11px; color: var(--text-muted);">
                    ${signal.timestamp}
                </span>
            </div>
        </div>
    `).join("");
}

// Render the ticker heatmap
function renderHeatmap() {
    const tickers = {
        "IONQ": "neutral", "RGTI": "neutral", "QBTS": "neutral",
        "QUBT": "neutral", "INFQ": "neutral", "IBM": "neutral",
        "GOOGL": "neutral", "MSFT": "neutral", "HON": "neutral", "NVDA": "neutral"
    };

    const container = document.getElementById("heatmap");
    container.innerHTML = Object.entries(tickers).map(([ticker, sentiment]) => `
        <div class="heatmap-cell" style="background: ${getSentimentBg(sentiment)}; color: ${getSentimentColor(sentiment)};">
            ${ticker}
        </div>
    `).join("");
}

// Analyze news text
async function analyzeNews() {
    const input = document.getElementById("news-input").value;
    if (!input.trim()) return;

    const resultDiv = document.getElementById("analysis-result");
    resultDiv.textContent = "Analyzing...";

    try {
        const result = await client.predict("/analyze_news", { 
            text: input, 
            source: "user_input" 
        });
        const data = result.data[0];
        resultDiv.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        resultDiv.textContent = `Error: ${error.message}`;
    }
}

// Utility: Get color for sentiment
function getSentimentColor(sentiment) {
    const colors = {
        "strongly_bullish": "var(--bullish)",
        "bullish": "var(--bullish)",
        "neutral": "var(--neutral)",
        "bearish": "var(--bearish)",
        "strongly_bearish": "var(--bearish)",
    };
    return colors[sentiment] || "var(--neutral)";
}

// Utility: Get background color for sentiment
function getSentimentBg(sentiment) {
    const colors = {
        "strongly_bullish": "rgba(16, 185, 129, 0.2)",
        "bullish": "rgba(16, 185, 129, 0.15)",
        "neutral": "rgba(100, 116, 139, 0.15)",
        "bearish": "rgba(239, 68, 68, 0.15)",
        "strongly_bearish": "rgba(239, 68, 68, 0.2)",
    };
    return colors[sentiment] || "rgba(100, 116, 139, 0.15)";
}

// Update timestamp
function updateTimestamp() {
    const now = new Date();
    document.getElementById("last-updated").textContent = 
        `Last updated: ${now.toLocaleTimeString()}`;
}

// Event Listeners
document.getElementById("analyze-btn").addEventListener("click", analyzeNews);

// Initialize
initClient();

// Auto-refresh every 60 seconds
setInterval(async () => {
    if (client) {
        await loadDashboard();
    }
}, 60000);
