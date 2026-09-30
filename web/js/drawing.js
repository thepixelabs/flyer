// The picture: the circuit, the line map, the depth map, the image model's live previews, the drawings.
//
// Live, it plays as one sequence: the circuit's connections draw themselves in (in the order
// the signal reached them), the exact line map the model receives fades in over them, then
// each preview from the sampler condenses onto the lines, and the finished drawing lands.
// A run can hold several drawings (other styles, models, colours or angles); the gallery
// under the picture lists them and any one can be shown.

import { $, $$, esc, int, motionOK, ART_W as W, ART_H as H } from './util.js';
import { setTip } from './tips.js';

const RAMP = [[255, 196, 90], [255, 110, 150], [120, 170, 255], [120, 255, 210]];
function rgb(ms) {
  const f = Math.min(1, Math.max(0, ms / 60)), k = Math.min(2, Math.floor(f * 3)), u = f * 3 - k;
  return RAMP[k].map((v, j) => Math.round(v * (1 - u) + RAMP[k + 1][j] * u)).join(',');
}
const proj = (x, y) => [x * W, (y - 0.5) * W + H / 2];
const isFront = r => !r || (!Math.round(r.yaw || 0) && !Math.round(r.pitch || 0));
const LAYERS = ['data', 'replay', 'control', 'depth', 'lines', 'preA', 'preB', 'strict', 'free'];

export class Drawing {
  /** describe(record) -> {short, long}: plain words for a drawing's settings (main knows the names). */
  constructor({ onView, onRecord, describe = () => ({ short: '', long: '' }) }) {
    this.onView = onView;
    this.onRecord = onRecord;
    this.describe = describe;
    this.el = {
      data: $('#imgData'), control: $('#imgControl'), depth: $('#imgDepth'), strict: $('#imgStrict'), free: $('#imgFree'),
      overlay: $('#overlay'), replay: $('#replayCanvas'), lines: $('#linesCanvas'), preA: $('#preA'), preB: $('#preB'),
      cap: $('#artCap'), legend: $('#stageLegend'), step: $('#artStep'), stepN: $('#stepN'), stepT: $('#stepT'),
      ring: $('#ringFill'), empty: $('#artEmpty'), replayBtn: $('#replayDrawBtn'), ov: $('#ovBtn'),
      gallery: $('#gallery'), galleryEmpty: $('#galleryEmpty'),
    };
    for (const c of [this.el.replay, this.el.lines]) { c.width = W; c.height = H; }
    this.rctx = this.el.replay.getContext('2d');
    this.lctx = this.el.lines.getContext('2d');
    this.view = 'data';
    $$('.art-bar button[data-v]').forEach(b => b.addEventListener('click', () => this.setView(b.dataset.v)));
    this.el.ov.addEventListener('click', () => this.setOverlay(this.el.ov.getAttribute('aria-pressed') !== 'true'));
    this.el.replayBtn.addEventListener('click', () => this.replayForming());
    $$('.thumb[data-go]').forEach(b => b.addEventListener('click', () => { this.setView(b.dataset.go); this.onView?.('art'); }));
    this.el.gallery.addEventListener('click', e => {
      const b = e.target.closest('[data-index]');
      if (b) this.showRecord(this.drawings.find(d => d.index === +b.dataset.index));
    });
    this.tick = this.tick.bind(this);
    requestAnimationFrame(this.tick);
    this.reset();
  }

  reset() {
    this.p = null; this.fid = null; this.frames = []; this.phase = 'none';
    this.replay = null; this.grow = null; this.formingReplay = null;
    this.drawings = []; this.current = null; this.runId = null;
    this.rctx.clearRect(0, 0, W, H); this.lctx.clearRect(0, 0, W, H);
    for (const k of ['data', 'control', 'depth', 'strict', 'free']) this.el[k].removeAttribute('src');
    this.el.overlay.removeAttribute('src');
    this.setImages({ data: '', control: '', depth: '', free: '', strict: '' });
    this.el.replayBtn.hidden = true;
    this.el.empty.hidden = false;
    this.setOverlay(false);
    this.renderFid(null);
    this.renderGallery();
    this.show([]);
    this.el.step.hidden = true;
    this.el.cap.textContent = '';
    this.el.legend.textContent = 'The picture appears at the end of an experiment.';
    this.enableTabs(false);
  }

  /** Which layers are visible. */
  show(layers) {
    for (const k of LAYERS) this.el[k].hidden = !layers.includes(k);
  }
  enableTabs(on) {
    $$('.art-bar button[data-v]').forEach(b => {
      const v = b.dataset.v;
      b.disabled = v === 'depth' ? !this.el.depth.getAttribute('src') : !on && v !== 'data';
    });
    this.el.ov.disabled = !this.el.overlay.getAttribute('src');
  }

  setImages({ data, control, depth, free, strict }) {
    const put = (sel, url) => $$(sel).forEach(im => (url ? (im.src = url) : im.removeAttribute('src')));
    if (data !== undefined) { put('.tData', data); if (data) this.el.data.src = data; }
    if (control !== undefined) { put('.tControl', control); if (control) { this.el.control.src = control; this.el.overlay.src = control; } }
    if (depth !== undefined) { put('.tDepth', depth); if (depth) this.el.depth.src = depth; else this.el.depth.removeAttribute('src'); }
    if (free !== undefined) { put('.tFree', free); if (free) this.el.free.src = free; }
    if (strict !== undefined) { put('.tStrict', strict); if (strict) this.el.strict.src = strict; }
  }

  setCounts(nNeur, nEdge) {
    $$('.nNeur').forEach(e => (e.textContent = int(nNeur)));
    $$('.nEdge').forEach(e => (e.textContent = int(nEdge)));
    this.counts = [nNeur, nEdge];
  }

  // ---- the live sequence ------------------------------------------------------------------

  /** The portrait event (or a saved portrait.json): the circuit of the answer. */
  portrait(p, { instant = false } = {}) {
    this.p = p;
    this.el.empty.hidden = true;
    this.setImages({ data: p.data, control: p.control });
    this.setCounts(p.n_neurons ?? p.index.length, p.n_edges ?? p.edges.length);
    const pos = new Map(p.index.map((gi, k) => [gi, proj(p.neurons[k][0], p.neurons[k][1])]));
    this.replay = { pos, spikes: p.spikes || [], idx: 0, last: 0 };
    this.enableTabs(false);
    if (instant || !motionOK()) { this.phase = 'final'; this.setView('data'); return; }
    // draw the circuit in, ordered by when each connection's sending neuron first fired
    const first = new Map(p.index.map((gi, k) => [gi, p.neurons[k][2]]));
    const edges = p.edges.filter(e => pos.has(e[0]) && pos.has(e[1])).sort((a, b) => first.get(a[0]) - first.get(b[0]));
    this.grow = { edges, pos, first, t0: performance.now(), ms: 3400, done: 0 };
    this.lctx.clearRect(0, 0, W, H);
    this.phase = 'circuit';
    this.show(['lines']);
    this.el.control.classList.add('fade');
    this.caption('<b>The circuit draws itself.</b> Each connection appears when the neuron sending it first fired, so you see the signal spread from the mouth.',
      'The connections between the neurons that fired for the favourite blend, drawn in the order the signal reached them. Colour is the time a neuron first fired, from orange (the mouth, 0 ms) through pink and blue to green (60 ms and later).');
    this.select('forming');
  }

  /** A redraw of a saved run is starting: show the new view's line map, then let the previews condense on it. */
  beginRedraw({ run, index }) {
    this.pending = `/runs/${run}/drawings/${index}`;
    this.frames = [];
    this.phase = 'forming';
    this.grow = null;
    this.formingReplay = null;
    this.el.empty.hidden = true;
    this.el.replayBtn.hidden = true;
    this.show(['data']);
    this.select('forming');
    this.renderGallery(index);
    this.caption('<b>Drawing it again.</b> First the line map and the depth map are made for this view, from the saved circuit.',
      'A new drawing of the same experiment. Nothing is simulated again: the neurons, their positions and their connections are the saved ones, seen from the angle and with the settings you chose.');
  }

  /** The server saved the new view's maps (it says so with the "draw" stage): show the line map the model will follow. */
  mapsReady() {
    if (!this.pending || this.phase !== 'forming') return;
    const base = this.pending;
    this.pending = null;
    this.setImages({ control: `${base}/control.png`, depth: `${base}/depth.png` });
    this.el.control.style.opacity = '1';
    this.show(['control']);
    this.caption('<b>The line map for this view.</b> The image model starts from here in a moment.', '');
  }

  /** ComfyUI's live preview: the drawing forming out of noise. */
  preview(m, { instant = false } = {}) {
    if (!m.image) return;
    this.frames.push(m);
    this.el.empty.hidden = true;
    if (this.phase === 'final') return;
    this.phase = 'forming';
    this.grow = null;
    this.showFrame(m, instant);
  }

  showFrame(m, instant) {
    const f = m.total ? m.step / m.total : 0;
    const [a, b] = this.flip ? [this.el.preB, this.el.preA] : [this.el.preA, this.el.preB];
    this.flip = !this.flip;
    a.onload = () => { a.style.opacity = String(0.5 + 0.5 * f); b.style.opacity = '0'; };
    a.style.opacity = '0';
    a.src = m.image;
    this.show(['control', 'preA', 'preB']);
    this.el.control.style.opacity = String(Math.max(0.12, 1 - f));
    this.el.step.hidden = false;
    this.el.stepN.textContent = m.step;
    this.el.stepT.textContent = m.total;
    this.el.ring.style.strokeDashoffset = String(100 - f * 100);
    this.caption(`<b>The image model at work.</b> Step ${m.step} of ${m.total}: speckle turns into ink, held to the line map underneath and to the brain's real shape.`,
      'A live preview from the image model while it draws. At every step it removes a little speckle, while ControlNet holds its strokes to the line map (the white lines fading out underneath) and its shading to the depth map. The words it gets only name a style.');
    if (!instant) this.select('forming');
  }

  /** The finished drawing (from the stream): its images, scores and, when the server sends it, its settings. */
  drawing(m, { instant = false } = {}) {
    const rec = m.drawing || { index: this.drawings.length, strict: m.strict, free: m.free, fidelity: m.fidelity };
    this.addRecord(rec);
    this.el.step.hidden = true;
    this.grow = null;
    this.el.replayBtn.hidden = this.frames.length < 3;
    if (instant || !motionOK()) { this.phase = 'final'; this.showRecord(rec, { view: 'strict' }); return; }
    // the moment: the free drawing lands on the last preview, then strict keeps only what's on the data
    this.current = rec;
    this.onRecord?.(rec);
    this.fid = rec.fidelity;
    this.setImages({ free: rec.free, strict: rec.strict, depth: rec.depth });
    this.renderFid(rec.fidelity);
    this.renderGallery(rec.index);
    this.phase = 'final';
    this.enableTabs(true);
    this.el.free.classList.add('fade');
    this.el.free.style.opacity = '0';
    this.show(['control', 'preA', 'preB', 'free']);
    this.el.free.onload = () => requestAnimationFrame(() => { this.el.free.style.opacity = '1'; });
    this.caption('<b>Finished.</b> This is everything the model drew. In a moment, the strict version keeps only the ink that sits on real data.', '');
    this.select('free', false);
    clearTimeout(this.toStrict);
    this.toStrict = setTimeout(() => { this.el.free.style.opacity = ''; this.showRecord(rec, { view: 'strict' }); }, 2600);
  }

  // ---- the gallery ------------------------------------------------------------------------

  /** Every saved drawing of a run (from /api/run/{id}). Older runs have none; they still show their top-level images. */
  setDrawings(list, runId) {
    this.runId = runId;
    this.drawings = [...(list || [])].sort((a, b) => a.index - b.index);
    this.renderGallery(this.current?.index);
  }
  addRecord(rec) {
    this.drawings = [...this.drawings.filter(d => d.index !== rec.index), rec].sort((a, b) => a.index - b.index);
  }

  /** Show one drawing: its line map, depth map, circuit at that angle, strict and free, and its scores. */
  showRecord(rec, { view } = {}) {
    if (!rec) return;
    clearTimeout(this.toStrict);
    this.current = rec;
    this.fid = rec.fidelity;
    this.phase = 'final';
    this.el.empty.hidden = true;
    this.setImages({ control: rec.control, depth: rec.depth || '', free: rec.free, strict: rec.strict,
                     data: isFront(rec) ? (this.p?.data || rec.data) : rec.data });
    this.renderFid(rec.fidelity);
    this.renderGallery(rec.index);
    this.enableTabs(true);
    let v = view || this.view;
    if (v === 'depth' && !rec.depth) v = 'strict';
    this.setView(v);
    this.onRecord?.(rec);
  }

  renderGallery(active) {
    const g = this.el.gallery, list = this.drawings;
    this.el.galleryEmpty.hidden = list.length > 0;
    $$('.g-item', g).forEach(e => e.remove());
    g.insertAdjacentHTML('beforeend', list.map(d => {
      const { short, long } = this.describe(d);
      const on = d.index === (active ?? this.current?.index);
      return `<button class="g-item" type="button" data-index="${d.index}" aria-pressed="${on}" title="${esc(long)}" aria-label="Drawing ${d.index + 1}: ${esc(long)}">
        ${d.strict ? `<img src="${esc(d.strict)}" alt="" loading="lazy">` : '<span class="g-wait">drawing…</span>'}
        <span class="g-lbl">${esc(short)}</span></button>`;
    }).join('') + (active != null && !list.some(d => d.index === active)
      ? `<button class="g-item" type="button" disabled aria-pressed="true"><span class="g-wait">drawing…</span><span class="g-lbl">new</span></button>` : ''));
    this.scrollGallery();
  }
  scrollGallery() {
    // scroll only the strip itself; scrollIntoView would also move the page
    const g = this.el.gallery, b = g.querySelector('[aria-pressed="true"]');
    if (b) g.scrollLeft = Math.max(0, b.offsetLeft - g.clientWidth / 2 + b.offsetWidth / 2);
  }

  replayForming() {
    if (this.frames.length < 3) return;
    this.formingReplay = { i: 0, next: 0 };
    this.phase = 'forming';
  }

  // ---- views ------------------------------------------------------------------------------

  setView(v) {
    this.view = v;
    this.formingReplay = null;
    if (this.phase !== 'final' && v !== 'data' && v !== 'control' && v !== 'depth') return;
    this.el.free.style.opacity = '';
    this.el.control.style.opacity = '';
    const front = isFront(this.current);
    const map = { data: front ? ['data', 'replay'] : ['data'], control: ['control'], depth: ['depth'], strict: ['strict'], free: ['free'] };
    this.show(map[v] || []);
    this.select(v);
    const f = this.fid?.[v];
    const score = f ? ` ${Math.round(f.on_data * 100)}% of its strongest lines are on the data (random lines would score ${Math.round(f.chance * 100)}%), and it traces ${Math.round(f.coverage * 100)}% of the data.` : '';
    const n = this.counts ? int(this.counts[0]) : 'The';
    const settings = this.current ? this.describe(this.current).long : '';
    const text = {
      data: [front
        ? `<b>The circuit.</b> ${n} neurons that fired, at their real positions, seen from the front. The flashes replay one second of one trial, 10 times slower<span id="capT"></span>.`
        : `<b>The circuit</b> from this drawing's angle: ${n} neurons that fired, at their real positions.`,
        `Each dot is a neuron that fired while the fly tasted its favourite blend, at its real position. A bigger dot fired more often. Each line is a real connection of 8 or more synapses. Colour is when a neuron first fired: <span class="ramp" role="img" aria-label="orange through pink and blue to green"></span> 0 to 60 ms and later, spreading from the taste cells at the mouth. <span class="swatch"></span>Blue violet lines are connections that calm the next cell down. The faint shape behind is all 138,639 neurons.`],
      control: ['<b>The line map.</b> The structure the image model has to follow: the brain outline, the strongest connections and a circle for each neuron.',
        'White on black, structure only: the outline of the brain, the 900 strongest connections (15 or more synapses) and a circle for each neuron that fired. The image model gets this, the depth map and a few words naming a style, nothing else.'],
      depth: ['<b>Depth map:</b> all 138,639 neurons at their real 3D positions, nearer ones brighter. It gives the drawing the real shape of the brain.',
        'Every neuron of the brain, not only the ones that fired, placed at its real 3D position and seen from the drawing\'s angle. Nearer cells are brighter. The image model uses it to shade the drawing like the real, rounded brain.'],
      strict: [`<b>Strict drawing.</b>${score}`, 'The model\'s ink is kept only near real data lines, and everything else fades to the colour of the paper. Where things are, and what connects to what, is data. The medium is the model\'s.'],
      free: [`<b>Free drawing.</b>${score}`, 'Everything the model drew over the line map, including strokes that aren\'t in the data. Turn on <b>Data lines</b> to see where the data is.'],
    }[v];
    if (text) this.caption(text[0], text[1] + (settings && v !== 'data' ? `<br><br><b>This drawing:</b> ${esc(settings)}.` : ''));
  }
  select(v, announce = true) {
    $$('.art-bar button[data-v]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === v)));
    if (announce && v !== 'forming') this.onView?.('art');
  }
  setOverlay(on) {
    this.el.overlay.style.opacity = on ? '0.75' : '0';
    this.el.ov.setAttribute('aria-pressed', String(on));
  }
  caption(short, long) {
    this.el.cap.innerHTML = short;
    if (long) this.el.legend.innerHTML = long;
  }

  // ---- fidelity and style ----------------------------------------------------------------

  /** Only the line scores. The server also reports an "anatomy" score, which isn't validated yet, so it stays out. */
  renderFid(f) {
    const box = $('#fid');
    if (!f?.strict || !f?.free) { box.innerHTML = '<p class="fine">The scores appear when the drawing is finished.</p>'; $('#fidMini').textContent = ''; return; }
    const row = k => {
      const v = f[k], on = Math.round(v.on_data * 100), ch = Math.round(v.chance * 100), cov = Math.round(v.coverage * 100);
      const lift = v.chance ? v.on_data / v.chance : 0;
      return `<div class="fid-row"><div class="hd"><span>${k === 'strict' ? 'Strict' : 'Free'}</span><b class="${lift >= 1.5 ? 'ok' : ''}">${on}% on data</b></div>
        <div class="meter-f" role="img" aria-label="${on}% on data; chance level ${ch}%"><div class="fill" style="width:${on}%"></div><div class="chance" style="left:${ch}%"></div></div>
        <div class="sub2">chance ${ch}%, ${lift.toFixed(1)} times chance, covers ${cov}% of the data</div></div>`;
    };
    const fr = f.free, frOn = Math.round(fr.on_data * 100), ch = Math.round(fr.chance * 100);
    const reading = fr.on_data < 0.5
      ? `In the free drawing only ${frOn}% of the strongest lines are on the data (chance ${ch}%), so most of what stands out there is the model's own ink. Strict removes it.`
      : `In the free drawing ${frOn}% of the strongest lines are on the data (chance ${ch}%), so it mostly follows the data.`;
    box.innerHTML = row('strict') + row('free') + `<p class="fine">${reading}</p>`;
    const pc = v => Math.round(v * 100) + '%';
    $('#fidMini').innerHTML = `This drawing: strict <b>${pc(f.strict.on_data)}</b> on data, free <b>${pc(fr.on_data)}</b>, chance <b>${pc(fr.chance)}</b>`;
  }

  showStyle(s, key) {
    $('#scLabel').textContent = s ? s.label : (key || 'Unknown style');
    setTip($('#scTip'), s?.description ? `${s.description} Why this style: ${s.why}` : 'The art style of the drawing.');
    $('#scPrompt').textContent = s?.prompt ? `“${s.prompt}”` : 'Restart the server to see the prompt for this style.';
  }

  // ---- animation ---------------------------------------------------------------------------

  tick(now) {
    requestAnimationFrame(this.tick);
    if (document.hidden) return;
    if (this.grow) this.growStep(now);
    if (this.formingReplay) {
      const r = this.formingReplay;
      if (now >= r.next) {
        const m = this.frames[r.i++];
        if (!m) { this.formingReplay = null; this.phase = 'final'; this.setView(this.view === 'free' ? 'free' : 'strict'); return; }
        this.showFrame(m, true);
        r.next = now + 220;
      }
    }
    if (this.replay && !this.el.replay.hidden) this.replayStep(now);
  }

  growStep(now) {
    const g = this.grow, ctx = this.lctx;
    const f = Math.min(1, (now - g.t0) / g.ms), upto = Math.floor(f * f * g.edges.length);
    ctx.lineWidth = 1.3;
    for (; g.done < upto; g.done++) {
      const [a, b, syn, sign] = g.edges[g.done];
      const [x0, y0] = g.pos.get(a), [x1, y1] = g.pos.get(b);
      const mx = (x0 + x1) / 2 - (y1 - y0) * 0.18, my = (y0 + y1) / 2 + (x1 - x0) * 0.18;
      const alpha = Math.min(0.85, 0.25 + Math.log1p(syn) / 7);
      ctx.strokeStyle = sign > 0 ? `rgba(${rgb(g.first.get(a))},${alpha})` : `rgba(140,150,255,${alpha})`;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.quadraticCurveTo(mx, my, x1, y1); ctx.stroke();
      ctx.fillStyle = `rgb(${rgb(g.first.get(a))})`;
      ctx.beginPath(); ctx.arc(x0, y0, 2.4, 0, 7); ctx.fill();
    }
    if (f >= 1) {
      this.grow = null;
      // then the exact line map the model receives settles over the drawn circuit
      this.el.control.style.opacity = '0';
      this.show(['lines', 'control']);
      requestAnimationFrame(() => { this.el.control.style.opacity = '0.9'; this.el.lines.style.opacity = '0.35'; });
      this.caption('<b>The line map.</b> The circuit reduced to white lines: the structure the image model has to follow.',
        'Structure only: the outline of the brain, the 900 strongest connections and a circle for each neuron that fired. The image model gets this, a depth map of the whole brain and a few words naming a style, nothing else.');
    }
  }

  replayStep(now) {
    const r = this.replay, ctx = this.rctx;
    const t = (now / 10) % 1000;
    if (t < r.last) r.idx = 0;
    r.last = t;
    ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,0.12)'; ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = 'lighter';
    const s = r.spikes;
    while (r.idx < s.length / 2 && s[2 * r.idx] <= t) {
      const p = r.pos.get(s[2 * r.idx + 1]);
      if (p) {
        ctx.fillStyle = 'rgba(255,245,220,0.9)'; ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 7); ctx.fill();
        ctx.fillStyle = 'rgba(255,200,120,0.15)'; ctx.beginPath(); ctx.arc(p[0], p[1], 10, 0, 7); ctx.fill();
      }
      r.idx++;
    }
    ctx.globalCompositeOperation = 'source-over';
    const cap = document.getElementById('capT');
    if (cap) cap.textContent = `: t = ${Math.round(t)} ms`;
  }
}
