// Where am I, and what are the live numbers: the phase rail, the phase card, telemetry, the MN9 sparkline.

import { $, $$, int, num, clock, motionOK } from './util.js';

export const ORDER = ['screen', 'retest', 'combine', 'confirm', 'portrait', 'draw'];
const NAMES = { screen: 'Try everything', retest: 'Check again', combine: 'Build a blend', confirm: 'Confirm', portrait: 'Map the response', draw: 'Draw it' };

// The four steps: one line each, with the detail on hover, focus or tap.
const HOW4 = `<ol class="how4">
  <li><span data-tip="Scientists mapped every nerve cell in one fruit fly's brain and how they connect. We make each cell fire when enough signals reach it.">A real brain, copied cell by cell</span></li>
  <li><span data-tip="Sugar is always being tasted. On top of it we switch on one other taste or smell at a time, 63 in all, and watch MN9, the cell that makes the fly reach out to eat.">Sugar, plus one more thing</span></li>
  <li><span data-tip="Anything that raises MN9 is tested again many times to rule out luck, then combined one by one. Each addition has to earn its place.">Keep what clearly helps</span></li>
  <li><span data-tip="The cells that fired for the winner, and the wires between them, become a line map. An image model draws over it in an old art style. It shows the brain's response, not food.">Draw the response</span></li>
</ol><button class="link-btn" type="button" data-open-about>Read the whole story</button>`;

const PHASES = {
  screen: ['Trying everything', `<p>Each small brain tastes sugar plus one other thing, all at once. The bar under it is
    MN9, the feeding neuron; the white tick is sugar alone.</p>`],
  retest: ['Checking again', `<p>The few that beat sugar alone run again with many more copies, so luck can't win.</p>`],
  combine: ['Building a blend', `<p>The best joins the blend, then the rest are tried on top of it. Each must clearly
    add something, or the search stops.</p>`],
  confirm: ['Confirming', `<p>Plain sugar and the final blend run again from scratch. This is the answer we report.</p>`],
  portrait: ['Mapping the response', `<p>The neurons that fired for the blend, where they sit and how they connect. The 3D
    brain shows that circuit.</p>`],
  draw: ['Drawing it', `<p>The image model gets a line map, a <span data-tip="All 138,639 neurons at their real 3D positions, nearer ones brighter, so the shading follows the real shape of the brain.">depth map</span>
    and a style. Watch the picture condense out of speckle.</p>`],
};

export function initHud() {
  $('#phaseBody').innerHTML = HOW4;
}

/** Mark the rail and fill the phase card. stage: one of ORDER, or idle | done | saved | error. */
export function setPhase(stage, { text = '', date = '', banner = false } = {}) {
  const k = ORDER.indexOf(stage), finished = stage === 'done' || stage === 'saved';
  $$('#rail li').forEach(li => {
    const j = ORDER.indexOf(li.dataset.stage);
    li.className = finished || (k >= 0 && j < k) ? 'done' : j === k ? 'on' : '';
    li.toggleAttribute('aria-current', j === k);
  });
  $('#railCompact').textContent = k >= 0 ? `Step ${k + 1} of 6: ${NAMES[stage]}` : finished ? 'All six steps done' : 'Ready';
  document.getElementById('app').dataset.stage = stage;

  const card = { step: $('#phaseStep'), title: $('#phaseTitle'), body: $('#phaseBody') };
  if (k >= 0) {
    card.step.textContent = `Now: step ${k + 1} of 6`;
    card.title.textContent = PHASES[stage][0];
    card.body.innerHTML = PHASES[stage][1] + (text ? `<p class="mono">${text.replace(/\u2026$/, '')}</p>` : '');
    if (banner && motionOK()) showBanner(`Step ${k + 1} of 6`, PHASES[stage][0]);
  } else if (stage === 'done') {
    card.step.textContent = 'Finished';
    card.title.textContent = 'The fly has chosen';
    card.body.innerHTML = `<p>See <b>What it chose</b> and <b>How sure</b> on the right. Each run starts from fresh
      noise, so ask again to see if the answer holds.</p>`;
  } else if (stage === 'saved') {
    card.step.textContent = 'A past experiment';
    card.title.textContent = date ? `From ${date}` : 'A saved experiment';
    card.body.innerHTML = `<p>A finished experiment: tests, answer and picture. <b>Ask the fly</b> runs a new one live.</p>` + HOW4;
  } else if (stage === 'error') {
    card.step.textContent = 'Something went wrong';
    card.title.textContent = 'The experiment stopped';
    card.body.innerHTML = `<p class="no">${text}</p><p>Press <b>Ask the fly</b> to try again.</p>`;
  } else {
    card.step.textContent = 'How it works';
    card.title.textContent = 'Four steps, in plain words';
    card.body.innerHTML = HOW4;
  }
}

function showBanner(n, title) {
  const b = $('#phaseBanner');
  b.querySelector('.pb-n').textContent = n;
  b.querySelector('.pb-t').textContent = title;
  b.classList.remove('show');
  void b.offsetWidth;
  b.classList.add('show');
}

export function setTelemetry(t) {
  const set = (id, v, hot) => { const e = $(id); if (v !== undefined) e.textContent = v; if (hot !== undefined) e.classList.toggle('hot', hot); };
  if ('sim' in t) set('#tmSim', t.sim == null ? 'idle' : `${int(t.sim)} ms`, t.sim != null);
  if ('copies' in t) {
    set('#tmCopies', t.copies ? int(t.copies[0] * t.copies[1]) : 'none');
    $('#tmCopies').title = t.copies ? `${t.copies[0]} tests, ${t.copies[1]} copies of each` : '';
  }
  if ('spikes' in t) set('#tmSpikes', int(t.spikes));
  if ('elapsed' in t) set('#tmElapsed', t.elapsed == null ? 'n/a' : clock(t.elapsed));
  if ('base' in t) set('#tmBase', t.base == null ? 'waiting' : `${num(t.base)} a second`);
  if ('best' in t) { set('#tmBest', t.best == null ? 'waiting' : `${num(t.best[0])}, ${t.best[1]}`); $('#tmBest').title = $('#tmBest').textContent; }
  if ('progress' in t) $('#tmProg').style.width = `${Math.round((t.progress || 0) * 100)}%`;
  if ('progressLabel' in t) $('#tmProgLabel').textContent = t.progressLabel;
}

/** MN9 over simulated time for the shown test (bright) and the batch's reference (dim). */
export class Spark {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.rows = []; // {t, mn9: [...]}
  }
  reset() { this.rows = []; this.draw(-1); }
  push(t, mn9) { this.rows.push({ t, mn9 }); }
  draw(sel) {
    const dpr = Math.min(devicePixelRatio || 1, 2), w = this.c.clientWidth, h = this.c.clientHeight;
    if (!w || !h) return;
    if (this.c.width !== Math.round(w * dpr)) { this.c.width = Math.round(w * dpr); this.c.height = Math.round(h * dpr); }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.rows.length || sel < 0) return;
    const max = Math.max(20, ...this.rows.flatMap(r => [r.mn9[0] || 0, r.mn9[sel] || 0])) * 1.1;
    const line = (k, colour, width) => {
      ctx.beginPath();
      this.rows.forEach((r, i) => { const x = (r.t / 1000) * w, y = h - ((r.mn9[k] || 0) / max) * (h - 2) - 1; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.stroke();
    };
    line(0, 'rgba(207,214,230,0.55)', 1);
    if (sel > 0) line(sel, '#7fe3ff', 1.6);
  }
}
