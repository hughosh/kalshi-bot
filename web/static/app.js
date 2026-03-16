"use strict";

// ── Reconnecting WebSocket (vanilla, no npm) ──────────────────────────────
class ReconnectingWebSocket {
  constructor(url, opts = {}) {
    this.url = url;
    this.reconnectDelay = opts.reconnectDelay ?? 2000;
    this.maxReconnectDelay = opts.maxReconnectDelay ?? 30000;
    this._delay = this.reconnectDelay;
    this._ws = null;
    this._dead = false;
    this.onmessage = null;
    this.onopen = null;
    this.onclose = null;
    this._connect();
  }
  _connect() {
    if (this._dead) return;
    this._ws = new WebSocket(this.url);
    this._ws.onopen = (e) => {
      this._delay = this.reconnectDelay;
      setConnected(true);
      if (this.onopen) this.onopen(e);
    };
    this._ws.onmessage = (e) => { if (this.onmessage) this.onmessage(e); };
    this._ws.onclose = this._ws.onerror = () => {
      setConnected(false);
      if (!this._dead) setTimeout(() => this._connect(), this._delay);
      this._delay = Math.min(this._delay * 1.5, this.maxReconnectDelay);
    };
  }
  close() { this._dead = true; this._ws && this._ws.close(); }
}

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  config: {},
  matches: {},   // match_id → match_state row
  tradesOffset: 0,
  tradesTotal: 0,
};

// ── Helpers ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmtTime = ts => ts ? new Date(ts * 1000).toLocaleTimeString() : '—';
const fmtAgo = ts => {
  const s = Math.round(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  return `${Math.floor(s / 60)}m ago`;
};
const fmtCents = c => c != null ? `${(c / 100).toFixed(2)}` : '—';
const fmtDollars = v => v != null ? `$${parseFloat(v).toFixed(2)}` : '—';
const pnlClass = c => c == null ? '' : (c >= 0 ? 'pnl-pos' : 'pnl-neg');

function setConnected(ok) {
  const el = $('conn-status');
  el.className = 'badge ' + (ok ? 'green' : 'red');
  el.innerHTML = `<span class="dot${ok ? ' pulse' : ''}"></span> ${ok ? 'Connected' : 'Reconnecting…'}`;
}

// ── Toast notifications ───────────────────────────────────────────────────
function showToast(data) {
  const el = document.createElement('div');
  el.className = `toast ${data.level || 'info'}`;
  el.textContent = data.message;
  $('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 6000);
}

// ── Config / controls panel ───────────────────────────────────────────────
function syncControls(cfg) {
  state.config = cfg;
  const fields = [
    'min_edge_pct', 'min_confidence', 'min_points_observed',
    'max_contracts_per_match', 'max_total_exposure',
    'match_stop_loss_dollars', 'portfolio_stop_loss_dollars', 'profit_target_dollars',
  ];
  fields.forEach(k => {
    const el = $('cfg-' + k);
    if (el && document.activeElement !== el) el.value = cfg[k] ?? '';
  });
  // Trading toggle button
  const btn = $('btn-toggle-trading');
  if (btn) {
    btn.textContent = cfg.trading_enabled ? '⏸ Pause Trading' : '▶ Resume Trading';
    btn.className = cfg.trading_enabled ? 'btn-warning' : 'btn-success';
  }
  // Mode badge
  const modeBadge = $('mode-badge');
  if (modeBadge) {
    modeBadge.textContent = cfg.paper_mode ? 'PAPER' : 'LIVE';
    modeBadge.className = 'badge ' + (cfg.paper_mode ? 'yellow' : 'red');
  }
}

async function applyConfig(key) {
  const el = $('cfg-' + key);
  if (!el) return;
  const value = el.value.trim();
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value }),
  });
  if (!res.ok) showToast({ level: 'error', message: `Failed to set ${key}` });
}

async function toggleTrading() {
  await fetch('/api/trading/toggle', { method: 'POST' });
}

async function setMode(mode) {
  await fetch('/api/trading/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  });
}

// ── Risk panel ────────────────────────────────────────────────────────────
function renderRisk(data) {
  // Portfolio stats
  const expEl = $('stat-exposure');
  const pnlEl = $('stat-pnl');
  const matchesEl = $('stat-matches');
  const haltedBanner = $('halted-banner');

  if (expEl) {
    const exp = data.total_exposure ?? data.total_cost ?? 0;
    const maxExp = state.config.max_total_exposure ?? 500;
    expEl.textContent = `$${parseFloat(exp).toFixed(2)} / $${parseFloat(maxExp).toFixed(2)}`;
  }
  if (pnlEl) {
    const pnl = (data.total_unrealized_pnl_cents ?? 0) / 100;
    pnlEl.textContent = `$${pnl.toFixed(2)}`;
    pnlEl.className = 'value ' + (pnl >= 0 ? 'pos' : 'neg');
  }
  if (matchesEl) matchesEl.textContent = data.active_matches ?? 0;

  if (haltedBanner) {
    haltedBanner.style.display = data.portfolio_halted ? 'block' : 'none';
  }

  // Per-match rows
  const tbody = $('risk-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  const matchData = data.matches ?? {};
  for (const [mid, m] of Object.entries(matchData)) {
    const pnl = (m.unrealized_pnl_cents ?? 0) / 100;
    const flags = [];
    if (m.stop_loss_triggered) flags.push('<span class="text-red">SL</span>');
    if (m.profit_target_triggered) flags.push('<span class="text-green">PT</span>');
    tbody.insertAdjacentHTML('beforeend', `
      <tr>
        <td title="${mid}">${m.player1 ?? ''} vs ${m.player2 ?? ''}</td>
        <td>${m.yes_contracts ?? 0}Y / ${m.no_contracts ?? 0}N</td>
        <td class="${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${pnl.toFixed(2)}</td>
        <td>${flags.join(' ') || '—'}</td>
      </tr>`);
  }
}

// ── Match signal cards ────────────────────────────────────────────────────
function renderMatchCard(match) {
  state.matches[match.match_id] = match;

  let card = $('card-' + match.match_id);
  if (!card) {
    card = document.createElement('div');
    card.className = 'match-card';
    card.id = 'card-' + match.match_id;
    $('cards-grid').appendChild(card);
  }

  const prob1 = match.win_prob_p1 != null ? (match.win_prob_p1 * 100).toFixed(1) : '—';
  const prob2 = match.win_prob_p1 != null ? ((1 - match.win_prob_p1) * 100).toFixed(1) : '—';
  const barPct = match.win_prob_p1 != null ? (match.win_prob_p1 * 100).toFixed(1) : 50;

  const confStr = match.confidence != null ? match.confidence.toFixed(2) : '—';
  const ciStr = (match.ci_lower != null && match.ci_upper != null)
    ? `[${(match.ci_lower*100).toFixed(0)}–${(match.ci_upper*100).toFixed(0)}%]` : '';

  const askStr = match.best_ask_cents != null ? match.best_ask_cents + '¢' : '—';
  const bidStr = match.best_bid_cents != null ? match.best_bid_cents + '¢' : '—';

  const edgeStr = match.edge != null
    ? `<span class="${match.edge >= 0 ? 'text-green' : 'text-red'}">${match.edge_side?.toUpperCase() ?? ''} ${(match.edge * 100).toFixed(1)}¢</span>`
    : '<span class="text-muted">—</span>';

  const score = `${match.sets_p1}-${match.sets_p2}  ${match.games_p1}-${match.games_p2}  ${match.points_p1}-${match.points_p2}`;
  const serverName = match.server === 0 ? match.player1 : match.player2;

  let sigHtml = '';
  if (match.last_signal) {
    const ls = typeof match.last_signal === 'string' ? JSON.parse(match.last_signal) : match.last_signal;
    const cls = ls.decision === 'traded' ? 'traded' : (ls.decision === 'skipped' ? 'skipped' : 'blocked');
    const detail = ls.decision === 'traded'
      ? `TRADE ${ls.side?.toUpperCase()} ${ls.contracts}@${ls.price_cents}¢`
      : (ls.skip_reason ?? ls.decision);
    sigHtml = `<div class="last-signal ${cls}">${detail} · ${fmtAgo(ls.timestamp)}</div>`;
  }

  const isHalted = match.stop_loss_triggered || match.profit_target_triggered;
  card.className = 'match-card' + (isHalted ? ' stopped' : '');

  card.innerHTML = `
    <div class="card-header">
      <div class="card-title">${match.player1} vs ${match.player2}</div>
      <button class="btn-danger" style="font-size:11px;padding:2px 8px" onclick="exitMatch('${match.match_id}')">EXIT</button>
    </div>
    <div class="card-score">${score}${match.is_tiebreak ? ' · TB' : ''}  <span class="text-muted">(${serverName} serving)</span></div>
    <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:${barPct}%"></div></div>
    <div class="prob-labels"><span class="p1">${match.player1.split(' ').pop()} ${prob1}%</span><span class="p2">${prob2}% ${match.player2.split(' ').pop()}</span></div>
    <div class="card-meta">
      <span>Conf <strong>${confStr}</strong> ${ciStr}</span>
      <span>Ask <strong>${askStr}</strong> Bid <strong>${bidStr}</strong></span>
      <span>Edge ${edgeStr}</span>
    </div>
    ${sigHtml}`;
}

async function exitMatch(matchId) {
  if (!confirm(`Close all positions for match ${matchId}?`)) return;
  const res = await fetch(`/api/matches/${matchId}/exit`, { method: 'POST' });
  if (!res.ok) showToast({ level: 'error', message: 'Exit request failed' });
}

// Remove inactive cards
function removeMatchCard(matchId) {
  const card = $('card-' + matchId);
  if (card) card.remove();
  delete state.matches[matchId];
}

// ── Trade history table ───────────────────────────────────────────────────
function prependTradeRow(trade) {
  const tbody = $('trades-tbody');
  if (!tbody) return;
  state.tradesTotal++;
  const tr = tradeRow(trade);
  tbody.insertAdjacentHTML('afterbegin', tr);
}

function tradeRow(t) {
  const sideClass = t.side === 'yes' ? 'side-yes' : 'side-no';
  const pnl = t.pnl_cents != null ? `<span class="${pnlClass(t.pnl_cents)}">$${(t.pnl_cents/100).toFixed(2)}</span>` : '<span class="text-muted">—</span>';
  const modeClass = t.mode === 'live' ? 'mode-live' : 'mode-paper';
  return `<tr>
    <td>${fmtTime(t.timestamp)}</td>
    <td title="${t.player1} vs ${t.player2}">${(t.player1||'').split(' ').pop()} / ${(t.player2||'').split(' ').pop()}</td>
    <td title="${t.ticker}" style="font-size:10px">${t.ticker ? t.ticker.slice(-12) : '—'}</td>
    <td class="${sideClass}">${(t.side||'').toUpperCase()}</td>
    <td>${t.contracts}</td>
    <td>${t.price_cents}¢</td>
    <td>${t.edge != null ? (t.edge*100).toFixed(1)+'¢' : '—'}</td>
    <td>${(t.model_prob*100||0).toFixed(0)}%</td>
    <td>${(t.confidence||0).toFixed(2)}</td>
    <td>${pnl}</td>
    <td class="${modeClass}">${(t.mode||'').toUpperCase()}</td>
  </tr>`;
}

async function loadTrades(reset = false) {
  if (reset) { state.tradesOffset = 0; $('trades-tbody').innerHTML = ''; }
  const res = await fetch(`/api/trades?limit=50&offset=${state.tradesOffset}`);
  const { trades, total } = await res.json();
  state.tradesTotal = total;
  state.tradesOffset += trades.length;
  const tbody = $('trades-tbody');
  trades.forEach(t => tbody.insertAdjacentHTML('beforeend', tradeRow(t)));
  $('load-more-btn').style.display = state.tradesOffset < total ? 'block' : 'none';
  $('trades-count').textContent = `${state.tradesTotal} total`;
}

// ── Initialise ────────────────────────────────────────────────────────────
async function init() {
  // Load initial data
  const [riskRes, cfgRes] = await Promise.all([
    fetch('/api/risk'),
    fetch('/api/config'),
  ]);
  renderRisk(await riskRes.json());
  syncControls(await cfgRes.json());
  await loadTrades(true);

  // Connect WebSocket
  const ws = new ReconnectingWebSocket(`ws://${location.host}/ws`);
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    switch (msg.type) {
      case 'init':
        syncControls(msg.data.config);
        (msg.data.matches || []).forEach(m => renderMatchCard(m));
        break;
      case 'match_update':
        renderMatchCard(msg.data);
        break;
      case 'trade':
        prependTradeRow(msg.data);
        break;
      case 'risk_update':
        renderRisk(msg.data);
        break;
      case 'config_update':
        syncControls(msg.data);
        break;
      case 'alert':
        showToast(msg.data);
        break;
    }
  };
}

document.addEventListener('DOMContentLoaded', init);
