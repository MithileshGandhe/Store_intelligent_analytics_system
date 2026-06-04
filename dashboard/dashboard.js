/* ═══════════════════════════════════════════════════════════════════
   APEX RETAIL — STORE INTELLIGENCE DASHBOARD
   Real-time dashboard client with WebSocket + REST fallback
   ═══════════════════════════════════════════════════════════════════ */

// ─── Configuration ──────────────────────────────────────────────────
const API_BASE = window.location.origin;
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_BASE = `${WS_PROTOCOL}//${window.location.host}`;
let currentStore = 'STORE_BLR_002';
let ws = null;
let wsReconnectAttempts = 0;
let wsReconnectTimer = null;
const WS_MAX_RECONNECT_DELAY = 30000;
const WS_BASE_DELAY = 1000;
let pollInterval = null;
const POLL_INTERVAL_MS = 30000;
const MAX_EVENTS = 20;

// Previous metric values for trend/animation
let prevMetrics = {
  visitors: null,
  conversion: null,
  queue: null,
  abandonment: null,
};

// Sparkline history
let sparklineHistory = {
  visitors: [],
  queue: [],
};
const SPARKLINE_MAX_POINTS = 30;

// ─── DOM References ─────────────────────────────────────────────────
const DOM = {
  storeSelector:      () => document.getElementById('store-selector'),
  statusDot:          () => document.getElementById('status-dot'),
  statusText:         () => document.getElementById('status-text'),
  currentTime:        () => document.getElementById('current-time'),
  // Metrics
  visitorsValue:      () => document.getElementById('metric-visitors-value'),
  visitorsTrend:      () => document.getElementById('metric-visitors-trend'),
  visitorsArrow:      () => document.getElementById('metric-visitors-arrow'),
  visitorsPct:        () => document.getElementById('metric-visitors-pct'),
  visitorsSub:        () => document.getElementById('metric-visitors-sub'),
  visitorsSparkline:  () => document.getElementById('metric-visitors-sparkline'),
  conversionValue:    () => document.getElementById('metric-conversion-value'),
  conversionTrend:    () => document.getElementById('metric-conversion-trend'),
  conversionArrow:    () => document.getElementById('metric-conversion-arrow'),
  conversionPct:      () => document.getElementById('metric-conversion-pct'),
  conversionBar:      () => document.getElementById('metric-conversion-bar'),
  queueValue:         () => document.getElementById('metric-queue-value'),
  queueTrend:         () => document.getElementById('metric-queue-trend'),
  queueArrow:         () => document.getElementById('metric-queue-arrow'),
  queuePct:           () => document.getElementById('metric-queue-pct'),
  queueSparkline:     () => document.getElementById('metric-queue-sparkline'),
  abandonmentValue:   () => document.getElementById('metric-abandonment-value'),
  abandonmentTrend:   () => document.getElementById('metric-abandonment-trend'),
  abandonmentArrow:   () => document.getElementById('metric-abandonment-arrow'),
  abandonmentPct:     () => document.getElementById('metric-abandonment-pct'),
  abandonmentBar:     () => document.getElementById('metric-abandonment-bar'),
  // Funnel
  funnelContainer:    () => document.getElementById('funnel-container'),
  funnelLoading:      () => document.getElementById('funnel-loading'),
  // Anomalies
  anomaliesList:      () => document.getElementById('anomalies-list'),
  anomalyCount:       () => document.getElementById('anomaly-count'),
  anomaliesEmpty:     () => document.getElementById('anomalies-empty'),
  // Heatmap
  heatmapGrid:        () => document.getElementById('heatmap-grid'),
  heatmapLoading:     () => document.getElementById('heatmap-loading'),
  // Events
  eventFeed:          () => document.getElementById('event-feed'),
  eventsEmpty:        () => document.getElementById('events-empty'),
  feedPulse:          () => document.getElementById('feed-pulse'),
  // Footer
  healthDot:          () => document.getElementById('health-dot'),
  healthStatusText:   () => document.getElementById('health-status-text'),
  lastEventTime:      () => document.getElementById('last-event-time'),
};

// ═══════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  startClock();
  await loadStores();
  const selector = DOM.storeSelector();
  selector.addEventListener('change', (e) => {
    currentStore = e.target.value;
    switchStore(currentStore);
  });
  switchStore(currentStore);
}

function startClock() {
  const tick = () => {
    const now = new Date();
    const el = DOM.currentTime();
    if (el) el.textContent = now.toLocaleTimeString('en-US', { hour12: false });
  };
  tick();
  setInterval(tick, 1000);
}

// ─── Load Stores ────────────────────────────────────────────────────
async function loadStores() {
  const selector = DOM.storeSelector();
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Attempt to find stores from health response
    let stores = [];
    if (data.stores && Array.isArray(data.stores)) {
      stores = data.stores;
    } else if (data.active_stores && Array.isArray(data.active_stores)) {
      stores = data.active_stores;
    } else if (data.store_ids && Array.isArray(data.store_ids)) {
      stores = data.store_ids;
    } else if (typeof data === 'object') {
      // Try to extract store list from any key containing 'store'
      for (const key of Object.keys(data)) {
        const val = data[key];
        if (Array.isArray(val) && val.length > 0 && typeof val[0] === 'string' && val[0].includes('STORE')) {
          stores = val;
          break;
        }
        if (typeof val === 'object' && !Array.isArray(val)) {
          const subKeys = Object.keys(val);
          if (subKeys.length > 0 && subKeys[0].includes('STORE')) {
            stores = subKeys;
            break;
          }
        }
      }
    }

    // If still empty, try a fallback
    if (stores.length === 0) {
      stores = ['STORE_BLR_002'];
    }

    selector.innerHTML = '';
    stores.forEach((storeId) => {
      const id = typeof storeId === 'string' ? storeId : storeId.store_id || storeId.id || String(storeId);
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = id.replace(/_/g, ' ');
      selector.appendChild(opt);
    });

    // Pre-select currentStore if in list
    if (stores.map(s => typeof s === 'string' ? s : s.store_id || s.id).includes(currentStore)) {
      selector.value = currentStore;
    } else {
      currentStore = selector.value;
    }

    updateHealthFooter(data);
  } catch (err) {
    console.warn('Failed to load stores from /health:', err);
    selector.innerHTML = '<option value="STORE_BLR_002">STORE BLR 002</option>';
    currentStore = 'STORE_BLR_002';
    selector.value = currentStore;
  }
}

// ─── Switch Store ───────────────────────────────────────────────────
function switchStore(storeId) {
  // Disconnect previous WS
  if (ws) {
    ws.onclose = null;
    ws.close();
    ws = null;
  }
  clearTimeout(wsReconnectTimer);
  wsReconnectAttempts = 0;
  clearInterval(pollInterval);

  // Reset UI
  resetUI();

  // Load all data
  fetchAllData(storeId);

  // Connect WebSocket
  connectWebSocket(storeId);

  // Start polling as fallback
  startPolling(storeId);
}

function resetUI() {
  ['visitors', 'conversion', 'queue', 'abandonment'].forEach(k => {
    prevMetrics[k] = null;
  });
  sparklineHistory = { visitors: [], queue: [] };
  // Clear event feed
  const feed = DOM.eventFeed();
  if (feed) feed.innerHTML = `<div class="empty-state" id="events-empty">
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="var(--text-secondary)" stroke-width="1" opacity="0.5"><rect x="6" y="8" width="28" height="24" rx="3"/><path d="M12 16h16M12 22h10M12 28h14"/></svg>
    <span>Waiting for events…</span></div>`;
}

// ═══════════════════════════════════════════════════════════════════
// DATA FETCHING (REST)
// ═══════════════════════════════════════════════════════════════════
async function fetchAllData(storeId) {
  await Promise.allSettled([
    fetchMetrics(storeId),
    fetchFunnel(storeId),
    fetchHeatmap(storeId),
    fetchAnomalies(storeId),
  ]);
}

// ─── Metrics ────────────────────────────────────────────────────────
async function fetchMetrics(storeId) {
  try {
    const res = await fetch(`${API_BASE}/stores/${storeId}/metrics`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateMetrics(data);
  } catch (err) {
    console.warn('Failed to fetch metrics:', err);
  }
}

function updateMetrics(data) {
  if (!data) return;

  // Normalize: data might be nested under data key or direct
  const m = data.data || data.metrics || data;

  // Visitors
  const visitors = m.visitors_today ?? m.visitors ?? m.total_visitors ?? m.visitor_count;
  if (visitors != null) {
    updateMetricCard('visitors', visitors, prevMetrics.visitors);
    prevMetrics.visitors = visitors;
    pushSparkline('visitors', visitors);
  }

  // Conversion
  const conversion = m.conversion_rate ?? m.conversion ?? m.conversion_pct;
  if (conversion != null) {
    const pct = conversion > 1 ? conversion : conversion * 100;
    updateMetricCard('conversion', pct, prevMetrics.conversion);
    prevMetrics.conversion = pct;
    // Update bar
    const bar = DOM.conversionBar();
    if (bar) {
      bar.style.width = `${Math.min(pct, 100)}%`;
      bar.className = 'metric-bar-fill';
      if (pct >= 30) bar.classList.add(''); // default green gradient
      else if (pct >= 15) { bar.classList.add('amber'); }
      else { bar.classList.add('danger'); }
    }
    // Color code the value
    const valEl = DOM.conversionValue();
    if (valEl) {
      valEl.classList.remove('rate-good', 'rate-warn', 'rate-bad');
      if (pct >= 30) valEl.classList.add('rate-good');
      else if (pct >= 15) valEl.classList.add('rate-warn');
      else valEl.classList.add('rate-bad');
    }
  }

  // Queue Depth
  const queue = m.avg_queue_depth ?? m.queue_depth ?? m.avg_queue ?? m.queue_length;
  if (queue != null) {
    updateMetricCard('queue', typeof queue === 'number' ? parseFloat(queue.toFixed(1)) : queue, prevMetrics.queue);
    prevMetrics.queue = queue;
    pushSparkline('queue', queue);
  }

  // Abandonment
  const abandonment = m.abandonment_rate ?? m.abandon_rate ?? m.abandonment_pct;
  if (abandonment != null) {
    const pct = abandonment > 1 ? abandonment : abandonment * 100;
    updateMetricCard('abandonment', pct, prevMetrics.abandonment);
    prevMetrics.abandonment = pct;
    const bar = DOM.abandonmentBar();
    if (bar) bar.style.width = `${Math.min(pct, 100)}%`;
  }
}

function updateMetricCard(key, value, prevValue) {
  const elMap = {
    visitors: DOM.visitorsValue,
    conversion: DOM.conversionValue,
    queue: DOM.queueValue,
    abandonment: DOM.abandonmentValue,
  };
  const trendMap = {
    visitors: { trend: DOM.visitorsTrend, arrow: DOM.visitorsArrow, pct: DOM.visitorsPct },
    conversion: { trend: DOM.conversionTrend, arrow: DOM.conversionArrow, pct: DOM.conversionPct },
    queue: { trend: DOM.queueTrend, arrow: DOM.queueArrow, pct: DOM.queuePct },
    abandonment: { trend: DOM.abandonmentTrend, arrow: DOM.abandonmentArrow, pct: DOM.abandonmentPct },
  };

  const el = elMap[key]();
  if (!el) return;

  // Format value
  const isPercentage = key === 'conversion' || key === 'abandonment';
  const displayValue = isPercentage ? formatPercentage(value) : formatNumber(value);

  // Animate if changed
  if (prevValue != null && prevValue !== value) {
    animateValueChange(el, prevValue, value, isPercentage);

    // Trend
    const refs = trendMap[key];
    if (refs) {
      const trendEl = refs.trend();
      const arrowEl = refs.arrow();
      const pctEl = refs.pct();
      if (trendEl && arrowEl && pctEl) {
        const diff = value - prevValue;
        const pctChange = prevValue !== 0 ? Math.abs((diff / prevValue) * 100) : 0;
        // For queue and abandonment, up is bad
        const isInverse = key === 'queue' || key === 'abandonment';
        if (diff > 0) {
          trendEl.className = `metric-trend ${isInverse ? 'down' : 'up'}`;
          arrowEl.textContent = '↑';
          pctEl.textContent = `${pctChange.toFixed(1)}%`;
        } else if (diff < 0) {
          trendEl.className = `metric-trend ${isInverse ? 'up' : 'down'}`;
          arrowEl.textContent = '↓';
          pctEl.textContent = `${pctChange.toFixed(1)}%`;
        } else {
          trendEl.className = 'metric-trend neutral';
          arrowEl.textContent = '→';
          pctEl.textContent = '0%';
        }
      }
    }
  } else {
    el.textContent = displayValue;
  }
}

function animateValueChange(el, from, to, isPercentage) {
  const duration = 600;
  const startTime = performance.now();

  el.classList.add('changed');
  setTimeout(() => el.classList.remove('changed'), duration);

  function step(timestamp) {
    const elapsed = timestamp - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (to - from) * eased;
    el.textContent = isPercentage ? formatPercentage(current) : formatNumber(current);
    if (progress < 1) {
      requestAnimationFrame(step);
    }
  }
  requestAnimationFrame(step);
}

// ─── Sparklines ─────────────────────────────────────────────────────
function pushSparkline(key, value) {
  sparklineHistory[key].push(value);
  if (sparklineHistory[key].length > SPARKLINE_MAX_POINTS) {
    sparklineHistory[key].shift();
  }
  renderSparkline(key);
}

function renderSparkline(key) {
  const containerMap = {
    visitors: DOM.visitorsSparkline,
    queue: DOM.queueSparkline,
  };
  const container = containerMap[key]?.();
  if (!container) return;

  const data = sparklineHistory[key];
  if (data.length < 2) return;

  // Ensure canvas exists
  let canvas = container.querySelector('canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    container.innerHTML = '';
    container.appendChild(canvas);
  }

  const rect = container.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const w = rect.width;
  const h = rect.height;
  const min = Math.min(...data) * 0.9;
  const max = Math.max(...data) * 1.1 || 1;
  const range = max - min || 1;

  const stepX = w / (data.length - 1);

  // Gradient fill
  const gradient = ctx.createLinearGradient(0, 0, 0, h);
  if (key === 'visitors') {
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.25)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
  } else {
    gradient.addColorStop(0, 'rgba(139, 92, 246, 0.25)');
    gradient.addColorStop(1, 'rgba(139, 92, 246, 0)');
  }

  // Area
  ctx.beginPath();
  ctx.moveTo(0, h);
  data.forEach((v, i) => {
    const x = i * stepX;
    const y = h - ((v - min) / range) * h * 0.85 - h * 0.05;
    if (i === 0) ctx.lineTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Line
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = i * stepX;
    const y = h - ((v - min) / range) * h * 0.85 - h * 0.05;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = key === 'visitors' ? '#3b82f6' : '#8b5cf6';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Last point dot
  const lastX = (data.length - 1) * stepX;
  const lastY = h - ((data[data.length - 1] - min) / range) * h * 0.85 - h * 0.05;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = key === 'visitors' ? '#60a5fa' : '#a78bfa';
  ctx.fill();
}

// ─── Funnel ─────────────────────────────────────────────────────────
async function fetchFunnel(storeId) {
  try {
    const res = await fetch(`${API_BASE}/stores/${storeId}/funnel`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderFunnel(data);
  } catch (err) {
    console.warn('Failed to fetch funnel:', err);
    const loading = DOM.funnelLoading();
    if (loading) loading.innerHTML = '<span style="color:var(--text-tertiary)">Funnel data unavailable</span>';
  }
}

function renderFunnel(data) {
  const container = DOM.funnelContainer();
  if (!container) return;

  // Normalize: extract stages from various possible structures
  let stages = [];
  const raw = data.data || data.funnel || data.stages || data;

  if (Array.isArray(raw)) {
    stages = raw;
  } else if (typeof raw === 'object') {
    // Might be { entry: N, zone_visit: N, billing: N, purchase: N }
    const orderedKeys = ['entry', 'zone_visit', 'billing', 'purchase',
                         'entries', 'zone_visits', 'billings', 'purchases',
                         'Entry', 'Zone Visit', 'Billing', 'Purchase'];
    const stageLabels = ['Entry', 'Zone Visit', 'Billing', 'Purchase'];
    const foundKeys = [];
    for (const key of Object.keys(raw)) {
      if (typeof raw[key] === 'number') {
        foundKeys.push({ key, value: raw[key] });
      }
    }
    if (foundKeys.length > 0) {
      stages = foundKeys.map((item, i) => ({
        label: stageLabels[i] || item.key.replace(/_/g, ' '),
        count: item.value,
      }));
    }
  }

  if (stages.length === 0) {
    // Fallback: show placeholder
    container.innerHTML = '<div class="funnel-loading"><span style="color:var(--text-tertiary)">No funnel data</span></div>';
    return;
  }

  // Determine max for bar width
  const maxCount = Math.max(...stages.map(s => s.count || s.value || 0)) || 1;

  container.innerHTML = '';
  stages.forEach((stage, i) => {
    const count = stage.count || stage.value || 0;
    const label = stage.label || stage.stage || stage.name || `Stage ${i + 1}`;
    const widthPct = (count / maxCount) * 100;

    // Drop-off percentage
    let dropoff = '';
    if (i > 0) {
      const prevCount = stages[i - 1].count || stages[i - 1].value || 0;
      if (prevCount > 0) {
        const dropPct = ((prevCount - count) / prevCount * 100).toFixed(1);
        dropoff = `−${dropPct}%`;
      }
    }

    const stageEl = document.createElement('div');
    stageEl.className = 'funnel-stage';
    stageEl.style.animationDelay = `${i * 0.1}s`;
    stageEl.innerHTML = `
      <span class="funnel-label">${escapeHtml(label)}</span>
      <div class="funnel-bar-container">
        <div class="funnel-bar stage-${i % 4}" style="width: 0%">
          <span class="funnel-bar-value">${formatNumber(count)}</span>
        </div>
      </div>
      <span class="funnel-dropoff">${dropoff}</span>
    `;
    container.appendChild(stageEl);

    // Animate bar width after DOM insertion
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const bar = stageEl.querySelector('.funnel-bar');
        if (bar) bar.style.width = `${widthPct}%`;
      });
    });
  });
}

// ─── Heatmap ────────────────────────────────────────────────────────
async function fetchHeatmap(storeId) {
  try {
    const res = await fetch(`${API_BASE}/stores/${storeId}/heatmap`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderHeatmap(data);
  } catch (err) {
    console.warn('Failed to fetch heatmap:', err);
    const loading = DOM.heatmapLoading();
    if (loading) loading.innerHTML = '<span style="color:var(--text-tertiary)">Heatmap data unavailable</span>';
  }
}

function renderHeatmap(data) {
  const grid = DOM.heatmapGrid();
  if (!grid) return;

  // Normalize zones
  let zones = [];
  const raw = data.data || data.zones || data.heatmap || data;
  if (Array.isArray(raw)) {
    zones = raw;
  } else if (typeof raw === 'object') {
    // Might be { zone_id: { score, visits, dwell } }
    zones = Object.entries(raw).map(([key, val]) => ({
      zone_id: key,
      ...(typeof val === 'object' ? val : { normalized_score: val }),
    }));
  }

  if (zones.length === 0) {
    grid.innerHTML = '<div class="heatmap-loading"><span style="color:var(--text-tertiary)">No zone data</span></div>';
    return;
  }

  grid.innerHTML = '';
  zones.forEach((zone, i) => {
    const score = zone.normalized_score ?? zone.score ?? zone.heat ?? 50;
    const visits = zone.visit_count ?? zone.visits ?? zone.visitor_count ?? '—';
    const dwell = zone.avg_dwell_time ?? zone.dwell_time ?? zone.avg_dwell ?? '—';
    const name = zone.zone_id ?? zone.zone_name ?? zone.name ?? `Zone ${i + 1}`;
    const confidence = zone.confidence ?? zone.confidence_level ?? 'high';

    const bgColor = getHeatColor(score);
    const isLowConf = confidence === 'low' || confidence < 0.5;

    const zoneEl = document.createElement('div');
    zoneEl.className = `heatmap-zone${isLowConf ? ' low-confidence' : ''}`;
    zoneEl.style.background = bgColor;
    zoneEl.style.animationDelay = `${i * 0.05}s`;
    zoneEl.setAttribute('title', `${name} — Score: ${score}`);

    const dwellFormatted = typeof dwell === 'number' ? `${dwell.toFixed(0)}s` : dwell;

    zoneEl.innerHTML = `
      <span class="zone-score">${Math.round(score)}</span>
      <div class="zone-name">${escapeHtml(name.replace(/_/g, ' '))}</div>
      <div class="zone-stats">
        <div class="zone-stat">
          <span class="zone-stat-label">Visits</span>
          <span>${formatNumber(visits)}</span>
        </div>
        <div class="zone-stat">
          <span class="zone-stat-label">Dwell</span>
          <span>${dwellFormatted}</span>
        </div>
      </div>
    `;
    grid.appendChild(zoneEl);
  });
}

function getHeatColor(score) {
  // Map 0-100 to green → yellow → red with transparency
  const s = Math.max(0, Math.min(100, score));
  let r, g, b;
  if (s <= 50) {
    // Green to Yellow
    const t = s / 50;
    r = Math.round(16 + (245 - 16) * t);
    g = Math.round(185 + (158 - 185) * t);
    b = Math.round(129 + (11 - 129) * t);
  } else {
    // Yellow to Red
    const t = (s - 50) / 50;
    r = Math.round(245 + (239 - 245) * t);
    g = Math.round(158 + (68 - 158) * t);
    b = Math.round(11 + (68 - 11) * t);
  }
  return `rgba(${r}, ${g}, ${b}, 0.25)`;
}

// ─── Anomalies ──────────────────────────────────────────────────────
async function fetchAnomalies(storeId) {
  try {
    const res = await fetch(`${API_BASE}/stores/${storeId}/anomalies`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderAnomalies(data);
  } catch (err) {
    console.warn('Failed to fetch anomalies:', err);
  }
}

function renderAnomalies(data) {
  const list = DOM.anomaliesList();
  const countEl = DOM.anomalyCount();
  if (!list) return;

  // Normalize
  let anomalies = [];
  const raw = data.data || data.anomalies || data.alerts || data;
  if (Array.isArray(raw)) {
    anomalies = raw;
  }

  if (countEl) countEl.textContent = anomalies.length;

  if (anomalies.length === 0) {
    list.innerHTML = `<div class="empty-state" id="anomalies-empty">
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="var(--text-secondary)" stroke-width="1" opacity="0.5">
        <circle cx="20" cy="20" r="16"/><path d="M14 20l4 4 8-8"/>
      </svg>
      <span>No anomalies detected</span></div>`;
    return;
  }

  // Sort by severity: CRITICAL > WARN > INFO
  const severityOrder = { CRITICAL: 0, WARN: 1, WARNING: 1, INFO: 2 };
  anomalies.sort((a, b) => {
    const sa = severityOrder[(a.severity || a.level || 'INFO').toUpperCase()] ?? 3;
    const sb = severityOrder[(b.severity || b.level || 'INFO').toUpperCase()] ?? 3;
    return sa - sb;
  });

  list.innerHTML = '';
  anomalies.forEach((anomaly) => {
    const severity = (anomaly.severity || anomaly.level || 'INFO').toUpperCase();
    const message = anomaly.message || anomaly.description || anomaly.text || 'Unknown anomaly';
    const action = anomaly.suggested_action || anomaly.action || anomaly.recommendation || '';
    const timestamp = anomaly.timestamp || anomaly.detected_at || anomaly.time || '';

    let sevClass = 'info';
    if (severity === 'CRITICAL') sevClass = 'critical';
    else if (severity === 'WARN' || severity === 'WARNING') sevClass = 'warn';

    const item = document.createElement('div');
    item.className = 'anomaly-item';
    item.innerHTML = `
      <span class="anomaly-severity ${sevClass}">${severity}</span>
      <div class="anomaly-content">
        <div class="anomaly-message">${escapeHtml(message)}</div>
        ${action ? `<div class="anomaly-action">${escapeHtml(action)}</div>` : ''}
      </div>
      ${timestamp ? `<span class="anomaly-time">${getTimeSince(timestamp)}</span>` : ''}
    `;
    list.appendChild(item);
  });
}

// ═══════════════════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════════════════
function connectWebSocket(storeId) {
  if (ws) {
    ws.onclose = null;
    ws.close();
  }

  const url = `${WS_BASE}/ws/dashboard/${storeId}`;
  showConnectionStatus('reconnecting');

  try {
    ws = new WebSocket(url);
  } catch (err) {
    console.warn('WebSocket creation failed:', err);
    showConnectionStatus('disconnected');
    scheduleReconnect(storeId);
    return;
  }

  ws.onopen = () => {
    console.log(`[WS] Connected to ${storeId}`);
    wsReconnectAttempts = 0;
    showConnectionStatus('connected');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      onWebSocketMessage(msg);
    } catch (err) {
      console.warn('[WS] Failed to parse message:', err);
    }
  };

  ws.onerror = (err) => {
    console.warn('[WS] Error:', err);
  };

  ws.onclose = (event) => {
    console.log(`[WS] Disconnected (code ${event.code})`);
    showConnectionStatus('disconnected');
    scheduleReconnect(storeId);
  };
}

function scheduleReconnect(storeId) {
  clearTimeout(wsReconnectTimer);
  const delay = Math.min(WS_BASE_DELAY * Math.pow(2, wsReconnectAttempts), WS_MAX_RECONNECT_DELAY);
  wsReconnectAttempts++;
  console.log(`[WS] Reconnecting in ${delay}ms (attempt ${wsReconnectAttempts})`);
  wsReconnectTimer = setTimeout(() => {
    connectWebSocket(storeId);
  }, delay);
}

function onWebSocketMessage(msg) {
  const type = msg.type || msg.event_type || msg.kind;
  const data = msg.data || msg.payload || msg;

  switch (type) {
    case 'metrics_update':
      updateMetrics(data);
      break;

    case 'funnel_update':
      renderFunnel(data);
      break;

    case 'anomaly_update':
      renderAnomalies(data);
      break;

    case 'new_event':
      addEventToFeed(data);
      break;

    case 'health_update':
      updateHealthFooter(data);
      break;

    case 'heatmap_update':
      renderHeatmap(data);
      break;

    default:
      // Might be a raw event
      if (data.event_type || data.visitor_id) {
        addEventToFeed(data);
      } else {
        console.log('[WS] Unknown message type:', type, data);
      }
  }
}

function showConnectionStatus(status) {
  const dot = DOM.statusDot();
  const text = DOM.statusText();
  if (!dot || !text) return;

  dot.className = 'status-dot';

  switch (status) {
    case 'connected':
      dot.classList.add('connected');
      text.textContent = 'Connected';
      text.style.color = 'var(--accent-green)';
      break;
    case 'disconnected':
      dot.classList.add('disconnected');
      text.textContent = 'Disconnected';
      text.style.color = 'var(--accent-red)';
      break;
    case 'reconnecting':
      dot.classList.add('reconnecting');
      text.textContent = 'Connecting…';
      text.style.color = 'var(--accent-amber)';
      break;
  }
}

// ═══════════════════════════════════════════════════════════════════
// LIVE EVENT FEED
// ═══════════════════════════════════════════════════════════════════
function addEventToFeed(event) {
  const feed = DOM.eventFeed();
  if (!feed) return;

  // Remove empty state
  const empty = feed.querySelector('.empty-state');
  if (empty) empty.remove();

  const eventType = event.event_type || event.type || 'UNKNOWN';
  const visitorId = event.visitor_id || event.visitor || '—';
  const zone = event.zone || event.zone_id || '';
  const timestamp = event.timestamp || event.time || new Date().toISOString();

  // Create event item
  const item = document.createElement('div');
  item.className = 'event-item new';
  const badgeClass = eventType.toLowerCase().replace(/ /g, '_');
  item.innerHTML = `
    <span class="event-type-badge ${badgeClass}">${eventType.replace(/_/g, ' ')}</span>
    <span class="event-details">
      <span class="visitor-id">${escapeHtml(String(visitorId).slice(-8))}</span>
      ${zone ? `<span class="zone-info">@ ${escapeHtml(zone)}</span>` : ''}
    </span>
    <span class="event-time">${getTimeSince(timestamp)}</span>
  `;

  // Prepend
  feed.insertBefore(item, feed.firstChild);

  // Limit to MAX_EVENTS
  while (feed.children.length > MAX_EVENTS) {
    feed.removeChild(feed.lastChild);
  }

  // Remove "new" class after animation
  setTimeout(() => item.classList.remove('new'), 1000);

  // Update footer last event time
  const lastTimeEl = DOM.lastEventTime();
  if (lastTimeEl) lastTimeEl.textContent = `Last event: ${getTimeSince(timestamp)}`;

  // Pulse the feed dot
  const pulse = DOM.feedPulse();
  if (pulse) {
    pulse.style.animation = 'none';
    void pulse.offsetHeight; // trigger reflow
    pulse.style.animation = 'pulse-green 0.5s ease-out 2';
  }
}

// ═══════════════════════════════════════════════════════════════════
// HEALTH / FOOTER
// ═══════════════════════════════════════════════════════════════════
function updateHealthFooter(data) {
  const dot = DOM.healthDot();
  const text = DOM.healthStatusText();
  if (!dot || !text) return;

  const status = data.status || data.health || data.state || 'unknown';
  const statusLower = status.toLowerCase();

  dot.className = 'health-dot';
  if (statusLower === 'healthy' || statusLower === 'ok' || statusLower === 'up') {
    dot.classList.add('healthy');
    text.textContent = `System: Healthy`;
  } else if (statusLower === 'degraded' || statusLower === 'warning') {
    dot.classList.add('degraded');
    text.textContent = `System: Degraded`;
  } else {
    dot.classList.add('unhealthy');
    text.textContent = `System: ${status}`;
  }

  // Append uptime or latency info if available
  if (data.uptime) text.textContent += ` • Uptime: ${data.uptime}`;
  if (data.latency_ms) text.textContent += ` • ${data.latency_ms}ms`;
}

async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateHealthFooter(data);
  } catch (err) {
    const dot = DOM.healthDot();
    const text = DOM.healthStatusText();
    if (dot) { dot.className = 'health-dot unhealthy'; }
    if (text) text.textContent = 'System: Unreachable';
  }
}

// ═══════════════════════════════════════════════════════════════════
// POLLING FALLBACK
// ═══════════════════════════════════════════════════════════════════
function startPolling(storeId) {
  clearInterval(pollInterval);
  pollInterval = setInterval(() => {
    // Only poll if WS is not connected
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.log('[Poll] WebSocket down, polling REST endpoints');
      fetchAllData(storeId);
      fetchHealth();
    }
  }, POLL_INTERVAL_MS);
}

// ═══════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════
function formatNumber(n) {
  if (n == null || n === '—') return '—';
  const num = Number(n);
  if (isNaN(num)) return String(n);
  if (Number.isInteger(num)) {
    return num.toLocaleString('en-US');
  }
  return num.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

function formatPercentage(n) {
  if (n == null || n === '—') return '—';
  const num = Number(n);
  if (isNaN(num)) return String(n);
  return num.toFixed(1);
}

function getTimeSince(timestamp) {
  if (!timestamp) return '—';
  try {
    const then = new Date(timestamp);
    const now = new Date();
    const diffMs = now - then;
    const diffSec = Math.floor(diffMs / 1000);

    if (diffSec < 0) return 'just now';
    if (diffSec < 5) return 'just now';
    if (diffSec < 60) return `${diffSec}s ago`;

    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;

    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}h ago`;

    const diffDay = Math.floor(diffHour / 24);
    return `${diffDay}d ago`;
  } catch {
    return '—';
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
