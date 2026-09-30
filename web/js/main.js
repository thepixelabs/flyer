// Flyer: wires the stream to the stage, the docks and the HUD.

import { $, $$, esc, sgn, num, int, clamp, runDate, lower } from './util.js';
import { learn, info, blendKey, CLASSES } from './names.js';
import { getStatus, startRun, redrawRun, createStream } from './stream.js';
import { loadAnatomy } from './anatomy.js';
import { Wall } from './wall.js';
import { Drawing } from './drawing.js';
import { renderResult, renderChosen, resetResults } from './results.js';
import { initAbout } from './explain.js';
import { initHud, setPhase, setTelemetry, Spark, ORDER } from './hud.js';
import { initTips, setTip } from './tips.js';

const app = $('#app');
const LIVE_STAGES = new Set(['screen', 'retest', 'combine', 'confirm']);

const state = {
  mode: 'idle',      // idle | live | drawing (a redraw) | done | saved | error
  running: false,
  runId: null,
  style: null,
  started: null,     // epoch seconds
  finishedAt: null,
  stage: 'idle',
  batch: null,       // {stage, conditions, trials}
  blend: [],         // blend so far during the combine rounds
  retest: [], rounds: [],
  result: null,
  spikes: 0,
  lastFrame: 0, interval: 1.5,
  selectedIdx: -1,
  follow: true,
  manualView: false,
  channels: null,    // /api/channels, when the server has it
  job: null,         // 'experiment' | 'redraw': what the attached stream is doing
};

let brain = null, anatomy = null, styles = {}, models = [];
const spark = new Spark($('#spark'));
const about = initAbout();
initHud();
initTips();

// ---------------------------------------------------------------------------------------
// the stage views and the dock panels

function setView(v, { auto = false } = {}) {
  if (auto && state.manualView) return;
  if (!auto) state.manualView = true;
  app.dataset.view = v;
  $$('.stage-bar [data-view]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.view === v)));
  $$(`.stage-bar [data-view="${v}"] .badge`).forEach(b => (b.hidden = true));
  brain?.setActive(v !== 'art');
  wall.setActive(v === 'wall');
  updateStageNote();
}
$$('.stage-bar [data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
$('#expandBrain').addEventListener('click', () => setView('brain'));

const RIGHT = ['choice', 'sure', 'picture', 'past'];
let rightPanel = 'choice';
function setPanel(p) {
  app.dataset.panel = p;
  if (RIGHT.includes(p)) rightPanel = p;
  RIGHT.forEach(k => {
    const on = k === rightPanel;
    $(`#panel${k[0].toUpperCase()}${k.slice(1)}`).hidden = !on;
    $(`.dock-tabs [data-panel="${k}"]`).setAttribute('aria-selected', String(on));
  });
  $$('.mobile-tabs [data-panel]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.panel === p)));
  $$(`button[data-panel="${p}"] .badge`).forEach(b => (b.hidden = true));
}
$$('[data-panel]').forEach(b => b.tagName === 'BUTTON' && b.addEventListener('click', () => setPanel(b.dataset.panel)));

/** A dot on a tab whose content changed while it wasn't showing. */
function badge(p) {
  const shown = (p === rightPanel && !isPhone()) || app.dataset.panel === p;
  if (!shown) $$(`button[data-panel="${p}"] .badge`).forEach(b => (b.hidden = false));
}
function badgeView(v) { if (app.dataset.view !== v) $$(`.stage-bar [data-view="${v}"] .badge`).forEach(b => (b.hidden = false)); }
const isPhone = () => matchMedia('(max-width: 1099px)').matches;

document.addEventListener('click', e => { if (e.target.closest('[data-open-about]')) about.open(); });

// ---------------------------------------------------------------------------------------
// the wall, the drawing

const wall = new Wall($('#wall'), $('#wallScroll'), { onSelect: onSelectTest });
const drawing = new Drawing({ onView: v => setView(v), describe, onRecord: rec => drawing.showStyle(styles[rec.style], rec.style) });

// ---------------------------------------------------------------------------------------
// drawing settings: style, image model, colours

const MODEL_NAMES = [[/juggernaut/i, 'Juggernaut XL'], [/illustrious/i, 'Illustrious XL'], [/realvis/i, 'RealVisXL'],
                     [/epicrealism/i, 'epiCRealism XL'], [/pony/i, 'Pony Diffusion XL']];
const DEFAULT_MODEL = /^Juggernaut-?XL_v9/i;
function modelName(file) {
  if (!file) return 'Juggernaut XL';
  const hit = MODEL_NAMES.find(([re]) => re.test(file));
  return hit ? hit[1] : file.replace(/\.safetensors$/i, '').replace(/[_]+/g, ' ');
}
const palette = () => $('input[name="palette"]:checked')?.value || 'ink';
const angleWords = a => (!a || (!Math.round(a.yaw || 0) && !Math.round(a.pitch || 0)) ? 'from the front'
  : `turned ${deg(a.yaw)}, tilted ${deg(a.pitch)}`);
/** Whole degrees with a real minus sign. */
function deg(v) { const d = Math.round(v || 0); return `${d < 0 ? '\u2212' : ''}${Math.abs(d)}\u00b0`; }
function settings() {
  return { style: $('#styleSelect').value || 'cajal', model: $('#modelSelect').value || undefined, palette: palette() };
}
/** Plain words for one drawing's settings: a short gallery label and a full sentence. */
function describe(rec) {
  const style = styles[rec.style]?.label || rec.style || 'Drawing';
  const colours = rec.palette === 'data' ? 'colours from the data' : 'ink colours';
  const angle = angleWords(rec);
  return {
    short: [style.split(' ')[0], rec.model && !DEFAULT_MODEL.test(rec.model) ? modelName(rec.model) : '',
            rec.palette === 'data' ? 'data colours' : '', angle === 'from the front' ? '' : deg(rec.yaw)].filter(Boolean).join(', '),
    long: `${style}, ${modelName(rec.model)}, ${colours}, ${angle}`,
  };
}
/** Show the settings a running job uses (after a refresh the panel should match what is being drawn). */
function applySettings(m) {
  if (m.style && $('#styleSelect').querySelector(`option[value="${m.style}"]`)) { $('#styleSelect').value = m.style; noteStyle(); }
  if (m.model && $('#modelSelect').querySelector(`option[value="${m.model}"]`)) $('#modelSelect').value = m.model;
  const radio = m.palette && $(`input[name="palette"][value="${m.palette}"]`);
  if (radio) radio.checked = true;
}

/** Only models that passed the fidelity test are offered. Retries once; the default always works. */
async function loadModels(attempt = 1) {
  const sel = $('#modelSelect'), note = $('#modelNoteBody');
  try {
    const r = await fetch('/api/models', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status === 502 ? "the image program isn't answering" : `the server answered ${r.status}`);
    models = await r.json();
  } catch (e) {
    if (attempt < 2) { setTimeout(() => loadModels(attempt + 1), 1500); return; }
    sel.innerHTML = '<option value="">Juggernaut XL (default)</option>';
    note.innerHTML = `<p>Couldn't list the image models (${esc(e.message)}), so only the default is offered.</p>`;
    return;
  }
  const ok = models.filter(m => m.qualified).sort((a, b) => (DEFAULT_MODEL.test(b.name) - DEFAULT_MODEL.test(a.name)) || modelName(a.name).localeCompare(modelName(b.name)));
  const keep = sel.value;
  sel.innerHTML = ok.map(m => `<option value="${esc(m.name)}">${esc(modelName(m.name))}${DEFAULT_MODEL.test(m.name) ? ' (default)' : ''}</option>`).join('')
    || '<option value="">Juggernaut XL (default)</option>';
  if (keep && ok.some(m => m.name === keep)) sel.value = keep;
  const pct = v => `${Math.round(v * 100)}%`;
  const tested = models.filter(m => !m.qualified && m.candidate && m.fidelity);
  note.innerHTML = `<p><b>Only models that stay faithful to the data are offered.</b> Each drew the same saved
      experiment; we measured the share of its strongest lines on the data, and how much of the data it traced.</p>
    <ul>${ok.map(m => `<li><b>${esc(modelName(m.name))}</b>: ${m.fidelity
      ? `${pct(m.fidelity.on_data)} on the data, traces ${pct(m.fidelity.coverage)}`
      : 'the reference'}</li>`).join('')}</ul>
    ${tested.length ? `<p>Tested and left out: ${tested.map(m => `${esc(modelName(m.name))} (${pct(m.fidelity.on_data)} on the data)`).join(', ')}.</p>` : ''}
    <p>Other models can't be used: ControlNet, which holds the drawing to the line map, works with one family only.</p>`;
}

function onSelectTest(key, { follow }) {
  const b = state.batch;
  state.follow = follow;
  state.selectedIdx = b ? b.conditions.findIndex(c => blendKey(c) === key) : -1;
  const cond = b?.conditions[state.selectedIdx] || key.split('|').filter(Boolean);
  updateMarks(cond);
  updateBrainHud();
  spark.draw(state.selectedIdx);
}

function updateMarks(cond = []) {
  if (!brain) return;
  const ch = state.channels;
  const inputs = ch ? [...(cond.length || state.mode === 'live' ? ch.sugar : []), ...cond.flatMap(id => ch.byId.get(id) || [])] : [];
  brain.setMarks({ inputs, mn9: ch ? ch.mn9 : [] });
}

function updateBrainHud() {
  const title = $('#viewTitle'), sub = $('#viewSub'), cap = $('#viewCaption');
  const live = state.mode === 'live' && state.batch && LIVE_STAGES.has(state.stage);
  $('#mn9Hud').hidden = !live;
  if (brain?.circuit) {
    title.textContent = 'The circuit of the answer';
    sub.textContent = `${int(brain.circuit.local.size)} neurons that fired and the strongest connections between them`;
    cap.textContent = 'Lines appear in the order the signal reached them. Then one recorded second replays, 10 times slower.';
  } else if (live && state.selectedIdx >= 0) {
    const name = wall.nameOf(blendKey(state.batch.conditions[state.selectedIdx])) || 'this test';
    title.textContent = `Showing: ${name}`;
    sub.textContent = `Live spikes of copy 1 of ${state.batch.trials}. Amber rings: the sensor cells switched on. White: MN9`;
    cap.textContent = state.follow ? 'Following the test with the most feeding. Pick any test to show it here instead.'
      : 'You picked this test. Pick it again to follow the leader.';
  } else {
    title.textContent = "A fruit fly's brain";
    sub.textContent = anatomy ? `All ${int(anatomy.n)} neurons, each at its real position` : 'Every dot is one neuron';
    cap.textContent = 'Drag to turn it. Scroll or pinch to zoom.' + (state.channels ? ' The white rings are MN9, the feeding neuron.' : '');
  }
}

function updateStageNote() {
  const v = app.dataset.view, b = state.batch;
  let t = '';
  if (v === 'wall') t = state.mode === 'live' && b ? `${b.conditions.length} tests, ${b.trials} copies of each: ${int(b.conditions.length * b.trials)} brains running at once`
    : wall.size ? 'The first screen of this experiment, best first' : '';
  else if (v === 'brain') t = brain?.circuit ? 'The neurons that fired for the favourite blend' : 'Pick a test in Tests to see its spikes here';
  else t = 'From the brain\'s response to a drawing';
  $('#stageNote').textContent = t;
}

function wallCaption() {
  const s = state.stage, b = state.batch, r = state.rounds.at(-1);
  let t;
  if (state.mode === 'saved' || state.mode === 'done') t = 'The first screen of this experiment, best first. Green tests clearly raised the feeding neuron; faded ones did not. Pick one to see where its sensor cells enter the brain.';
  else if (s === 'screen') t = 'Each small brain tastes sugar plus one other thing. Flashes are its real spikes. The bar is the feeding neuron and the white tick is sugar alone.';
  else if (s === 'retest') t = state.retest.length ? 'Green ones held up when checked again. The faded ones were probably luck.' : `Checking again: the shortlist, ${b?.trials || 64} copies each, next to sugar alone.`;
  else if (s === 'combine') t = r && r.round === rounds
    ? (r.added ? `Round ${r.round}: ${info(r.added).short.toLowerCase()} joins the blend.` : `Round ${r.round}: nothing clearly adds to the blend, so the search stops.`)
    : `Round ${rounds}: the blend so far (first) next to the blend plus each candidate.`;
  else if (s === 'confirm') t = 'The final check: sugar alone next to the final blend, from scratch, 32 copies each.';
  else t = wall.size ? 'The last batch of tests.' : 'No tests yet. They appear here when an experiment runs.';
  $('#wallCap').textContent = t;
  $('#wallEmpty').hidden = wall.size > 0;
}
let rounds = 0; // combine rounds started in this run

// ---------------------------------------------------------------------------------------
// the event stream

const stream = createStream({ onEvent, onCaughtUp, onConnection });

function onConnection(s) {
  if (s === 'retrying') setTop('Reconnecting to the server…');
  if (s === 'ok' && state.mode === 'live') setTop();
}

function beginLive(m) {
  resetRun();
  state.mode = m.running ? 'live' : 'done';
  state.running = m.running;
  state.runId = m.run;
  state.style = m.style;
  state.started = m.started;
  if (m.running) applySettings(m);
  drawing.showStyle(styles[m.style], m.style);
  setMode();
}

function onEvent(m, o) { if (gate) gate.push([m, o]); else handle(m, o); }

function handle(m, { catchingUp }) {
  const instant = catchingUp;
  switch (m.kind) {
    case 'attached':
      if (m.redraw && m.run) beginRedraw(m);
      else if (m.run) { beginLive(m); state.job = 'experiment'; state.redrawable = true; }
      break;
    case 'stage':
      state.stage = m.stage;
      if (m.stage === 'combine') rounds += 1;
      setPhase(m.stage, { text: esc(m.text), banner: !instant });
      if (LIVE_STAGES.has(m.stage)) setView('wall', { auto: true });
      if (m.stage === 'portrait') setView('brain', { auto: true });
      if (m.stage === 'draw') { setView('art', { auto: true }); setTelemetry({ progressLabel: 'Drawing progress' }); if (state.job === 'redraw') drawing.mapsReady(); }
      else setTelemetry({ progressLabel: 'Progress of this step', progress: 0 });
      wallCaption(); updateBrainHud(); updateStageNote();
      break;
    case 'progress':
      if (m.stage === 'draw') setTelemetry({ progress: m.value, sim: null });
      else setTelemetry({ progress: m.value });
      break;
    case 'live_init':
      state.batch = { stage: m.stage, conditions: m.conditions, trials: m.trials };
      if (m.stage === 'combine') state.blend = m.conditions[0];
      state.spikes = 0;
      spark.reset();
      wall.begin(m.conditions, { instant });
      setTelemetry({ copies: [m.conditions.length, m.trials], spikes: 0, sim: 0, best: null });
      wallCaption(); updateStageNote();
      break;
    case 'live': onLive(m, instant); break;
    case 'screen':
      learn(m.channels);
      wall.screen(m, { instant });
      setTelemetry({ base: m.baseline });
      wallCaption();
      break;
    case 'retest':
      state.retest = m.channels;
      wall.retest(m.channels, { instant });
      renderChosen(state.retest, state.rounds);
      badge('sure'); wallCaption();
      break;
    case 'round':
      state.rounds.push(m);
      wall.round(m, state.blend, { instant });
      renderChosen(state.retest, state.rounds);
      badge('sure'); wallCaption();
      break;
    case 'result':
      state.result = m;
      renderResult(m);
      wall.confirm(m.confirm, (m.blend || []).map(b => b.id));
      badge('choice');
      if (!instant && !isPhone()) setPanel('choice');
      break;
    case 'portrait':
      drawing.portrait(m, { instant });
      showCircuit(m, !instant);
      badge('picture'); badgeView('art');
      if (!instant) setView('art', { auto: true });
      break;
    case 'preview':
      drawing.preview(m, { instant });
      setTelemetry({ progress: m.total ? m.step / m.total : 0 });
      break;
    case 'drawing':
      drawing.drawing(m, { instant });
      badge('picture');
      break;
    case 'done':
      state.running = false; state.finishedAt = Date.now() / 1000;
      if (state.job === 'redraw') {
        state.mode = 'saved'; state.stage = 'saved';
        setPhase('saved', { date: runDate(state.runId) });
      } else {
        state.mode = 'done'; state.stage = 'done';
        setPhase('done');
      }
      setMode();
      loadHistory();
      break;
    case 'error':
      state.running = false;
      if (state.job === 'redraw') {
        state.mode = 'saved';
        setPhase('saved', { date: runDate(state.runId) });
        setMode();
        notice(`The drawing stopped: ${m.text}`, true);
        break;
      }
      state.mode = 'error';
      setPhase('error', { text: esc(m.text) });
      setMode();
      break;
  }
}

function onLive(m, instant) {
  const mn9 = m.mn9 || [], frames = m.frames || [];
  let leader = mn9.length > 1 ? 1 : 0;
  for (let i = 1; i < mn9.length; i++) if (mn9[i] > mn9[leader]) leader = i;
  // glow time matched to how often frames arrive, so each flash fades as the next one lands
  const now = performance.now() / 1000;
  if (state.lastFrame && !instant) state.interval = clamp(0.7 * state.interval + 0.3 * (now - state.lastFrame), 0.2, 4);
  state.lastFrame = now;
  const tau = clamp(state.interval * 0.55, 0.25, 1.6);
  wall.setTau(tau); brain?.setTau(tau);
  wall.live(m, leader);
  for (const f of frames) state.spikes += f.length;
  spark.push(m.t, mn9);
  if (state.selectedIdx >= 0 && frames[state.selectedIdx]) brain?.flash(frames[state.selectedIdx]);
  spark.draw(state.selectedIdx);
  const b = state.batch, leadName = b ? wall.nameOf(blendKey(b.conditions[leader])) : '';
  setTelemetry({ sim: m.t, spikes: state.spikes, base: mn9[0], best: mn9.length > 1 ? [mn9[leader], leadName] : null });
  const sel = mn9[state.selectedIdx];
  $('#mn9Now').textContent = sel != null ? num(sel) : '';
  $('#mn9Ref').textContent = state.selectedIdx > 0 ? `vs ${num(mn9[0])} for ${state.batch?.conditions[0].length ? 'the blend so far' : 'sugar alone'}` : '';
}

function onCaughtUp() { if (gate) gate.push(null); else caughtUp(); }

function caughtUp() {
  if (state.job === 'redraw') { updateBrainHud(); updateStageNote(); return; }
  // a finished run replays its whole log: show its full first screen rather than the last small batch
  if (!state.running && state.result) {
    const c = state.result.confirm;
    setTelemetry({ sim: null, copies: null, spikes: state.spikes, elapsed: null, progress: 1, base: c.baseline, best: [c.blend, 'final blend'] });
    wall.showSaved(state.result);
    state.batch = null;
    wallCaption();
    setView(state.result && drawing.p ? 'art' : 'wall', { auto: true });
  }
  if (state.running) wallCaption();
  updateBrainHud(); updateStageNote();
}

function showCircuit(p, animate) {
  state.portrait = p;
  if (!brain || !p.index?.length) return;
  brain.showCircuit(p, { animate });
  brain.setMarks({ inputs: [], mn9: state.channels?.mn9 || [] });
  updateBrainHud();
}

// ---------------------------------------------------------------------------------------
// modes: live, done, saved

function resetRun() {
  state.batch = null; state.blend = []; state.retest = []; state.rounds = []; state.result = null; state.portrait = null;
  state.spikes = 0; state.selectedIdx = -1; state.follow = true; state.manualView = false; state.finishedAt = null;
  rounds = 0;
  wall.clear(); resetResults(); drawing.reset(); spark.reset();
  brain?.clearCircuit();
  updateMarks([]);
  setTelemetry({ sim: null, copies: null, spikes: 0, base: null, best: null, progress: 0 });
  $$('.badge').forEach(b => (b.hidden = true));
  $$('.hist').forEach(h => h.removeAttribute('aria-current'));
}

function setMode() {
  const chip = $('#modeChip'), btn = $('#startBtn');
  const m = state.mode;
  chip.className = 'chip' + (m === 'live' || m === 'drawing' ? ' live' : m === 'saved' || m === 'done' ? ' saved' : m === 'error' ? ' error' : '');
  chip.textContent = { live: 'Live', drawing: 'Drawing', done: 'Finished', saved: 'Saved run', error: 'Stopped', idle: 'Idle' }[m];
  const busy = m === 'live' || m === 'drawing';
  btn.setAttribute('aria-disabled', String(busy));
  btn.textContent = m === 'live' ? 'Running\u2026' : m === 'drawing' ? 'Drawing\u2026' : m === 'idle' ? 'Ask the fly' : 'Ask again';
  $('#drawSettings').disabled = busy;
  $('#runNote').className = 'run-note';
  $('#runNote').textContent = m === 'live' ? 'Safe to refresh or close: it keeps running.'
    : m === 'drawing' ? 'Drawing again, nothing is simulated. Safe to refresh.'
    : 'Takes a few minutes. The drawing comes last.';
  $('#pastNote').hidden = true;
  $$('.js-redraw').forEach(b => { b.disabled = busy; });
  updateRedrawControls();
  setTop();
  updateBrainHud(); updateStageNote(); wallCaption();
}

function setTop(text) {
  const el = $('#topStatus');
  if (text) { el.textContent = text; return; }
  if (state.mode === 'live' || state.mode === 'drawing') el.innerHTML = `<span class="live">\u25cf ${state.mode === 'live' ? 'Live' : 'Drawing'}</span> ${elapsed()}`;
  else if (state.mode === 'saved' || state.mode === 'done') el.textContent = state.runId ? `Run of ${runDate(state.runId)}` : '';
  else el.textContent = '';
}
const elapsed = () => {
  if (!state.started) return '';
  const s = (state.finishedAt || Date.now() / 1000) - state.started;
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};
setInterval(() => {
  if (state.mode === 'live' || state.mode === 'drawing') { setTop(); setTelemetry({ elapsed: (Date.now() / 1000) - state.started }); }
}, 1000);

$('#startBtn').addEventListener('click', async () => {
  if (state.mode === 'live' || state.mode === 'drawing') return;
  const btn = $('#startBtn');
  btn.setAttribute('aria-disabled', 'true');
  btn.textContent = 'Starting…';
  try {
    const r = await startRun({ ...settings(), yaw: 0, pitch: 0 });
    stream.forget();
    stream.attach();
    if (!r.started) { $('#runNote').textContent = 'An experiment was already running, so you are watching that one.'; }
    setPanel(isPhone() ? 'now' : rightPanel);
  } catch (e) {
    state.mode = state.mode === 'live' ? 'live' : 'idle';
    setMode();
    $('#runNote').className = 'run-note error';
    $('#runNote').textContent = `The server didn't start the experiment (${e.message}). Is it still running?`;
  }
});

// ---------------------------------------------------------------------------------------
// drawing a saved run again

/** A short message where the visitor is looking: the Now dock, the picture caption and Past runs. */
function notice(text, error = false) {
  for (const el of [$('#runNote'), $('#pastNote')]) { el.className = (el.id === 'runNote' ? 'run-note' : 'fine') + (error ? ' error' : ''); el.textContent = text; el.hidden = false; }
  if (app.dataset.view === 'art') $('#artCap').textContent = text;
}

const canRedraw = () => !!state.runId && (state.mode === 'saved' || state.mode === 'done') && state.redrawable !== false;

/** Redraw run `id` (default: the one on screen) with the chosen settings, at `angle` (default: the front). */
async function redraw(id = state.runId, angle = { yaw: 0, pitch: 0 }) {
  if (!id || state.mode === 'live' || state.mode === 'drawing') return;
  $$('.js-redraw').forEach(b => b.setAttribute('aria-disabled', 'true'));
  try {
    const r = await redrawRun({ run: id, ...settings(), ...angle });
    if (!r.started) notice('Something else is running right now, so you are watching that instead. Try again when it has finished.');
    stream.forget();
    stream.attach();
  } catch (e) {
    notice(e.status === 404
      ? 'This experiment was saved before redrawing was possible, so it lacks the full circuit needed to draw it again. Run a new experiment to draw it in any style or from any angle.'
      : `The drawing didn't start: ${e.message}.`, true);
  } finally {
    $$('.js-redraw').forEach(b => b.removeAttribute('aria-disabled'));
  }
}

/** A redraw job attached (fresh, or after a refresh): make sure its saved run is on screen, then play the drawing events. */
let gate = null;
async function beginRedraw(m) {
  gate = [];
  try {
    if (state.runId !== m.run || !(state.mode === 'saved' || state.mode === 'done')) await openRun(m.run, { keepStream: true });
  } finally {
    state.job = 'redraw';
    state.mode = m.running ? 'drawing' : 'saved';
    state.running = m.running;
    state.started = m.started;
    state.finishedAt = null;
    state.manualView = false;
    drawing.beginRedraw({ run: m.run, index: m.index ?? drawing.drawings.length });
    if (m.running) applySettings(m);
    drawing.showStyle(styles[m.style], m.style);
    setView('art', { auto: true });
    setMode();
    const queued = gate;
    gate = null;
    for (const x of queued) x ? handle(...x) : caughtUp();
  }
}

$('#drawViewBtn').addEventListener('click', () => brain && redraw(state.runId, brain.angles()));
$('#redrawBtn').addEventListener('click', () => redraw(state.runId, drawing.current ? { yaw: drawing.current.yaw || 0, pitch: drawing.current.pitch || 0 } : undefined));

function updateRedrawControls(angles = brain?.angles()) {
  const can = canRedraw();
  const dv = $('#drawViewBtn');
  dv.hidden = !can || !brain;
  if (!dv.hidden) {
    const t = `Draw this view \u00b7 ${angleWords(angles)}`;
    if (dv.textContent !== t) dv.textContent = t;
  }
  const rb = $('#redrawBtn');
  rb.hidden = !can;
  rb.title = `Draw this experiment again, ${angleWords(drawing.current || {})}, with the style, model and colours chosen in How to draw it`;
}

// ---------------------------------------------------------------------------------------
// saved runs

async function openRun(id, { keepStream = false } = {}) {
  const res = await fetch(`/api/run/${encodeURIComponent(id)}`);
  if (!res.ok) { notice(`Couldn't open that experiment (${res.status}).`, true); return; }
  const r = await res.json();
  if (!keepStream) { stream.close(); stream.forget(); }
  resetRun();
  Object.assign(state, { mode: 'saved', running: false, runId: id, style: r.style, started: null, stage: 'saved', result: r.result });
  learn(r.result.screen); learn(r.result.blend);
  wall.showSaved(r.result);
  renderResult(r.result);
  setTelemetry({ elapsed: null, progress: 1, base: r.result.confirm.baseline, best: [r.result.confirm.blend, 'final blend'] });
  const base = { data: `/runs/${id}/data.png`, control: `/runs/${id}/control.png`, n_neurons: r.n_neurons, n_edges: r.n_edges };
  const listed = history.find(h => h.id === id);
  state.redrawable = listed?.redrawable; // undefined on an older server: then Redraw is offered and the server decides
  let circuit = null;
  if (listed?.portrait !== false) {
    try { const pr = await fetch(`/runs/${id}/portrait.json`); if (pr.ok) circuit = await pr.json(); } catch { /* older runs have none */ }
  }
  // newer runs save every connection; the 3D view and the draw-in keep the strongest 1,500, like the live event
  if (circuit?.edges?.length > 1500) circuit = { ...circuit, edges: [...circuit.edges].filter(e => e[2] >= 8).sort((a, b) => b[2] - a[2]).slice(0, 1500) };
  drawing.portrait({ neurons: [], index: [], spikes: [], edges: [], ...circuit, ...base }, { instant: true });
  if (circuit) showCircuit(circuit, true);
  drawing.showStyle(styles[r.style], r.style);
  if (r.drawings?.length) {
    drawing.setDrawings(r.drawings, id);
    drawing.showRecord(r.drawings.reduce((a, b) => (b.index > a.index ? b : a)), { view: 'strict' });
  } else {
    drawing.drawing({ strict: `/runs/${id}/strict.png`, free: `/runs/${id}/free.png`, fidelity: r.fidelity,
                      drawing: { index: 0, style: r.style, model: r.model, palette: r.palette || 'ink', yaw: 0, pitch: 0,
                                 strict: `/runs/${id}/strict.png`, free: `/runs/${id}/free.png`, control: `/runs/${id}/control.png`,
                                 data: `/runs/${id}/data.png`, fidelity: r.fidelity } }, { instant: true });
  }
  setPhase('saved', { date: runDate(id) });
  setMode();
  $$('.hist').forEach(h => h.toggleAttribute('aria-current', h.dataset.id === id));
  state.job = null;
  state.manualView = false;
  setView('art', { auto: true });
  if (!isPhone()) setPanel('choice');
}

let history = [];
async function loadHistory() {
  try { history = await (await fetch('/api/history')).json(); } catch { history = []; }
  const list = $('#historyList');
  if (!history.length) { list.innerHTML = '<p class="empty">No saved experiments yet.</p>'; return history; }
  list.innerHTML = history.map(r => {
    const names = r.ids ? r.ids.map(i => lower(info(i).short)) : r.blend;
    return `<div class="hist" data-id="${esc(r.id)}"><button class="hist-open" type="button"><img src="${esc(r.thumb)}" loading="lazy" alt="">
      <span>sugar + ${esc(names.join(' + ') || 'nothing')}, <b>${sgn(r.gain)}</b> a second
      <small>${esc(styles[r.style]?.label || r.style || '')}, ${esc(runDate(r.id))}</small></span></button>
      ${r.redrawable === false ? '<span class="hist-old tip-u" data-tip="Saved before redrawing existed, so it lacks the full circuit needed to draw it again.">No redraw</span>'
        : '<button class="btn small js-redraw" type="button" title="Draw this experiment again from the front, with the settings in How to draw it">Redraw</button>'}</div>`;
  }).join('');
  $$('.hist', list).forEach(h => {
    h.querySelector('.hist-open').addEventListener('click', () => { openRun(h.dataset.id); if (isPhone()) setPanel('picture'); });
    h.querySelector('.js-redraw')?.addEventListener('click', () => redraw(h.dataset.id));
  });
  $$('.js-redraw', list).forEach(b => { b.disabled = state.mode === 'live' || state.mode === 'drawing'; });
  return history;
}

// ---------------------------------------------------------------------------------------
// the 3D brain

async function initBrain() {
  const msg = $('#brainMsg');
  try {
    anatomy = await loadAnatomy(f => { msg.textContent = `Loading the neurons… ${Math.round(f * 100)}%`; });
  } catch (e) {
    msg.textContent = `Couldn't load the neurons (${e.message}). Everything else still works.`;
    return;
  }
  wall.setAnatomy(anatomy);
  renderLegend();
  msg.textContent = `Building ${int(anatomy.n)} neurons…`;
  try {
    const { Brain3D } = await import('./brain3d.js');
    brain = new Brain3D($('#brainCanvas'), anatomy);
    brain.setActive(app.dataset.view !== 'art');
    msg.hidden = true;
    brain.on((kind, a) => {
      if (kind === 'interact') $('#viewCaption').textContent = 'Arrow keys turn it too. Press 0 for the front view.';
      if (kind === 'view') updateRedrawControls(a);
    });
    updateRedrawControls();
    updateMarks(state.batch?.conditions[state.selectedIdx] || []);
    if (state.portrait) showCircuit(state.portrait, false);
    updateBrainHud();
  } catch (e) {
    msg.textContent = `The 3D view couldn't start (${e.message}). It needs WebGL and the three.js library from the internet. The tests and the picture still work.`;
  }
}
$$('[data-cam]').forEach(b => b.addEventListener('click', () => brain?.setView(b.dataset.cam)));

function renderLegend() {
  const counts = new Map();
  for (const c of anatomy.cls) counts.set(c, (counts.get(c) || 0) + 1);
  const keys = ['', ...anatomy.classes];
  $('#legendList').innerHTML = [...counts].sort((a, b) => b[1] - a[1]).map(([code, n]) => {
    const [label, colour] = CLASSES[keys[code]] || CLASSES[''];
    return `<li><i style="background:${colour}"></i><span>${esc(label)}</span><span>${int(n)}</span></li>`;
  }).join('');
}
$('#legendBtn').addEventListener('click', () => {
  const p = $('#legendPanel'), open = p.hidden;
  p.hidden = !open;
  $('#legendBtn').setAttribute('aria-expanded', String(open));
});

async function loadChannels() {
  try {
    const r = await fetch('/api/channels');
    if (!r.ok) return;
    const c = await r.json();
    learn(c.channels);
    state.channels = { sugar: c.sugar, mn9: c.mn9, byId: new Map(c.channels.map(x => [x.id, x.neurons])) };
    wall.setInputs(state.channels.byId);
    $('#legendMarks').textContent = 'Rings: amber marks the sensor cells switched on for the test you picked; white marks MN9, the feeding neuron.';
    updateMarks([]);
    updateBrainHud();
  } catch { /* an older server has no /api/channels; names fill in when the screen arrives */ }
}

// ---------------------------------------------------------------------------------------
// start up

/** The drawing styles, with their notes. Retries a slow or failed fetch, and says so rather than sitting empty. */
async function loadStyles(attempt = 1) {
  const sel = $('#styleSelect');
  if (attempt === 1) sel.innerHTML = '<option value="">Loading styles\u2026</option>';
  try {
    const r = await fetch('/api/styles', { cache: 'no-store' });
    if (!r.ok) throw new Error(`the server answered ${r.status}`);
    const raw = await r.json();
    // older servers answer {key: label}; newer ones {key: {label, description, why, prompt}}
    styles = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, typeof v === 'string' ? { label: v } : v]));
    if (!Object.keys(styles).length) throw new Error('the list was empty');
  } catch (e) {
    if (attempt < 4) { setTimeout(() => loadStyles(attempt + 1), 600 * attempt); return; }
    sel.innerHTML = '<option value="">Styles unavailable</option>';
    $('#runNote').className = 'run-note error';
    $('#runNote').innerHTML = `The drawing styles didn't load (${esc(e.message)}). <button class="btn small" type="button" id="retryStyles">Try again</button>`;
    $('#retryStyles').addEventListener('click', () => loadStyles(1));
    return;
  }
  const keep = sel.value || state.style;
  sel.innerHTML = Object.entries(styles).map(([k, v]) => `<option value="${esc(k)}">${esc(v.label || k)}</option>`).join('');
  if (keep && styles[keep]) sel.value = keep;
  noteStyle();
  if (state.style) drawing.showStyle(styles[state.style], state.style);
  loadHistory();
}
function noteStyle() {
  const s = styles[$('#styleSelect').value];
  setTip($('#styleTip'), s?.description ? `${s.label}: ${s.description} Why this style: ${s.why}` : 'The art style of the drawing.');
}
$('#styleSelect').addEventListener('change', noteStyle);

/** While nothing is attached, notice a run started from another tab or device. */
async function watchForRuns() {
  if (state.mode === 'live' || state.mode === 'drawing') return;
  try {
    const s = await getStatus();
    if (s.running && s.run !== state.runId) {
      $('#topStatus').innerHTML = '<button class="btn small primary" type="button" id="watchNew">A new experiment is running: watch it</button>';
      $('#watchNew').addEventListener('click', () => { stream.forget(); stream.attach(); });
    }
  } catch { /* the server is down; try again later */ }
}

(async function init() {
  setPanel('now');
  setView('brain', { auto: true });
  setPhase('idle');
  setMode();
  about.firstVisit();
  loadStyles();
  loadModels();
  initBrain();
  loadChannels();
  setInterval(watchForRuns, 8000);
  const [hist, status] = await Promise.all([loadHistory(), getStatus().catch(() => null)]);
  setPanel(isPhone() ? 'now' : 'choice');
  if (status?.running) { stream.attach(); return; }
  // not running: the last run the server remembers replays in full; otherwise open the newest saved one
  if (status?.run) {
    stream.attach();
    setTimeout(() => { if (!state.result && hist[0]) openRun(hist[0].id); }, 1500);
  } else if (hist[0]) openRun(hist[0].id);
})();

// for poking at the page from the browser console
window.fly = { state, wall, drawing, get brain() { return brain; } };
