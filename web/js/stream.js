// Talks to the experiment server: status, start, and the event stream.
//
// /api/stream replays everything of the current (or last) run, then goes live, then
// closes when the run ends. A refresh or a dropped connection simply attaches again;
// events already handled for the same run are skipped, so nothing is shown twice.

const TRANSIENT = new Set(['live', 'preview', 'progress']);
const CATCH_UP_QUIET_MS = 250; // a replayed log arrives in one burst; live events are seconds apart

export async function getStatus() {
  const r = await fetch('/api/status', { cache: 'no-store' });
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

/** The drawing settings as query parameters (only the ones that are set). */
function query(params) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') q.set(k, v);
  return q.toString();
}

/** Why a request failed, in the server's own words when it gave any. */
async function failure(r, what) {
  let text = '';
  try { const j = await r.json(); text = typeof j.detail === 'string' ? j.detail : ''; } catch { /* not JSON */ }
  const e = new Error(text || `${what} ${r.status}`);
  e.status = r.status;
  return e;
}

/** params: {style, trials, model, palette, yaw, pitch}. Answers {started, running, run}; started is false when a job was already going. */
export async function startRun(params) {
  const r = await fetch(`/api/start?${query({ trials: 16, ...params })}`, { method: 'POST' });
  if (!r.ok) throw await failure(r, 'start');
  return r.json();
}

/** Draw a saved run again: params {run, style, model, palette, yaw, pitch}. Same answer shape as startRun, plus index. */
export async function redrawRun(params) {
  const r = await fetch(`/api/draw?${query(params)}`, { method: 'POST' });
  if (!r.ok) throw await failure(r, 'redraw');
  return r.json();
}

/**
 * onEvent(msg, {catchingUp}) for every event; onCaughtUp() once a replayed burst ends;
 * onConnection('ok' | 'retrying' | 'closed').
 */
export function createStream({ onEvent, onCaughtUp = () => {}, onConnection = () => {} }) {
  let es = null, runId, handled = 0, finished = false, attachedRunning = false;
  let failures = 0, retryTimer = 0, quietTimer = 0, catchingUp = false, leaving = false;

  addEventListener('pagehide', () => { leaving = true; close(); });

  function close() {
    if (es) { es.onmessage = es.onerror = null; es.close(); es = null; }
    clearTimeout(retryTimer);
  }

  function attach() {
    close();
    let count = 0; // non-transient events seen on this connection
    catchingUp = true;
    es = new EventSource('/api/stream');
    es.onopen = () => { failures = 0; onConnection('ok'); };
    es.onmessage = e => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      clearTimeout(quietTimer);
      quietTimer = setTimeout(() => { if (catchingUp) { catchingUp = false; onCaughtUp(); } }, CATCH_UP_QUIET_MS);
      if (msg.kind === 'attached') {
        attachedRunning = msg.running;
        if (msg.run !== runId) { runId = msg.run; handled = 0; finished = false; onEvent(msg, { catchingUp, fresh: true }); }
        return;
      }
      if (!TRANSIENT.has(msg.kind)) {
        count += 1;
        if (count <= handled) return; // already shown before the reconnect
        handled = count;
      }
      if (msg.kind === 'done' || msg.kind === 'error') finished = true;
      onEvent(msg, { catchingUp });
    };
    es.onerror = () => {
      // The server closes the stream when a run ends; EventSource would otherwise reconnect and replay it forever.
      close();
      if (leaving) return;
      if (finished || !attachedRunning) { onConnection('closed'); return; }
      retryLater();
    };
  }

  // Wait until the server answers again, then attach; only say so after a few failed tries.
  function retryLater() {
    failures += 1;
    if (failures >= 3) onConnection('retrying');
    retryTimer = setTimeout(async () => {
      try { await getStatus(); attach(); } catch { retryLater(); }
    }, Math.min(800 * failures, 5000));
  }

  return {
    attach,
    close,
    get runId() { return runId; },
    forget() { runId = undefined; handled = 0; finished = false; },
  };
}
