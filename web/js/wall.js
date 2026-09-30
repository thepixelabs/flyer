// The wall: one small brain per test in the current simulation batch.
//
// Tiles are keyed by the blend they test, so when a new batch starts, a test that carries
// over (sugar alone, or the blend so far) glides to its new place, the rest fade out and
// new ones fade in. Each tile draws the real spikes of its first copy as they arrive.

import { esc, num, sgn, motionOK, dur, EASE, frontY } from './util.js';
import { blendKey, testLabel, info } from './names.js';

const TILE_ASPECT = 1.3;   // width / height of a tile, labels included
const MIN_TILE = 78;       // below this the wall scrolls inside the stage instead of shrinking
const SIL_W = 350, SIL_H = 200;

export class Wall {
  constructor(root, scroller, { onSelect }) {
    this.root = root;
    this.scroller = scroller;
    this.onSelect = onSelect;
    this.tiles = new Map();  // key -> tile
    this.keys = [];          // display order
    this.batchKeys = [];     // condition order of the current batch (live frames use it)
    this.selected = null;
    this.follow = true;      // follow the leader until the visitor picks a tile
    this.tau = 0.9;
    this.scale = 80;         // Hz at the right end of every meter
    this.ref = null;         // reference firing rate (the batch's first test)
    this.active = true;
    root.style.position = 'relative';
    new ResizeObserver(() => this.layout()).observe(scroller);
    this.loop = this.loop.bind(this);
    this.last = performance.now();
    requestAnimationFrame(this.loop);
  }

  /** Positions of every neuron in the front view, and the faint brain silhouette for the tiles. */
  setAnatomy(a) {
    this.px = a.x;
    this.py = a.y.map(frontY);
    const dens = new Float32Array(SIL_W * SIL_H);
    for (let i = 0; i < a.n; i++) {
      const x = Math.floor(this.px[i] * SIL_W), y = Math.floor(this.py[i] * SIL_H);
      if (x >= 0 && x < SIL_W && y >= 0 && y < SIL_H) dens[y * SIL_W + x] += 1;
    }
    const c = document.createElement('canvas');
    c.width = SIL_W; c.height = SIL_H;
    const ctx = c.getContext('2d'), img = ctx.createImageData(SIL_W, SIL_H);
    for (let k = 0; k < dens.length; k++) {
      const v = Math.min(1, Math.sqrt(dens[k] / 10));
      img.data.set([70, 110, 180, Math.round(v * 120)], 4 * k);
    }
    ctx.putImageData(img, 0, 0);
    this.root.style.setProperty('--sil', `url(${c.toDataURL()})`);
    for (const t of this.tiles.values()) this.drawMarks(t);
  }

  /** Where each channel's sensor cells sit, to mark on idle tiles (from /api/channels). */
  setInputs(byId) { this.inputs = byId; for (const t of this.tiles.values()) this.drawMarks(t); }

  setTau(s) { this.tau = s; }
  setActive(on) { this.active = on; if (on) this.layout(); }

  get size() { return this.keys.length; }

  // ---- building the wall ----------------------------------------------------------------

  /** A new simulation batch: conditions is a list of blends; the first is the reference. */
  begin(conditions, { instant = false } = {}) {
    const ref = conditions[0] || [];
    const keys = conditions.map(blendKey);
    this.batchKeys = keys;
    this.ref = null;
    this.transition(() => {
      for (const [k, t] of this.tiles) if (!keys.includes(k)) this.exit(t, instant);
      conditions.forEach((cond, i) => {
        const t = this.tiles.get(keys[i]) || this.create(keys[i], cond);
        this.label(t, testLabel(cond, ref));
        t.mn9 = 0; t.spikes = null; t.result = false; t.el.classList.remove('helps', 'hurts', 'dimmed', 'joined', 'lead');
        this.tag(t, i === 0 ? 'yardstick' : '');
        t.val.textContent = '';
        t.val.className = 'val';
        this.setMeter(t, 0);
        t.ctx?.clearRect(0, 0, t.canvas.width, t.canvas.height);
      });
      this.keys = keys;
      for (const k of keys) this.root.appendChild(this.tiles.get(k).el);
    }, instant);
    if (this.selected && !keys.includes(this.selected)) this.follow = true;
    this.markSelected();
  }

  create(key, cond) {
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'tile';
    el.setAttribute('aria-pressed', 'false');
    el.innerHTML = '<span class="mini"><canvas></canvas></span><span class="nm"></span>' +
      '<span class="row"><span class="meter"><i></i><b hidden></b></span><span class="val"></span></span><span class="tag" hidden></span>';
    const t = { key, cond, el, canvas: el.querySelector('canvas'), mini: el.querySelector('.mini'), nm: el.querySelector('.nm'),
                bar: el.querySelector('.meter i'), refTick: el.querySelector('.meter b'), val: el.querySelector('.val'),
                tagEl: el.querySelector('.tag'), mn9: 0, spikes: null, lastSpike: -1e9, fade: 0 };
    t.ctx = t.canvas.getContext('2d');
    el.addEventListener('click', () => this.pick(key));
    this.tiles.set(key, t);
    if (motionOK()) el.animate([{ opacity: 0, transform: 'scale(0.9)' }, { opacity: 1, transform: 'none' }],
      { duration: dur('--dur-scene'), easing: EASE.in, delay: 120 + Math.random() * 180, fill: 'backwards' });
    return t;
  }

  exit(t, instant) {
    this.tiles.delete(t.key);
    if (instant || !motionOK()) { t.el.remove(); return; }
    // take it out of the grid where it stands, then fade it away
    const { offsetLeft: l, offsetTop: tp, offsetWidth: w, offsetHeight: h } = t.el;
    Object.assign(t.el.style, { position: 'absolute', left: l + 'px', top: tp + 'px', width: w + 'px', height: h + 'px', pointerEvents: 'none' });
    t.el.animate([{ opacity: 1 }, { opacity: 0, transform: 'scale(0.85)' }], { duration: dur('--dur-panel'), easing: EASE.out, fill: 'forwards' })
      .finished.then(() => t.el.remove(), () => t.el.remove());
  }

  label(t, { name, sub, kind }) {
    t.name = name;
    t.kind = kind;
    t.nm.innerHTML = `${esc(name)} <small>${esc(sub)}</small>`;
    t.el.classList.remove('taste', 'smell', 'base');
    t.el.classList.add(kind);
    t.el.setAttribute('aria-label', name);
    this.drawMarks(t);
  }
  tag(t, text) { t.tagEl.textContent = text; t.tagEl.hidden = !text; }

  /** Reorder with a FLIP animation: measure, change, then play each tile from where it was. */
  transition(mutate, instant) {
    if (instant || !motionOK()) { mutate(); this.layout(); return; }
    const before = new Map([...this.tiles.values()].map(t => [t.el, t.el.getBoundingClientRect()]));
    mutate();
    this.layout();
    let i = 0;
    for (const t of this.tiles.values()) {
      const a = before.get(t.el);
      if (!a || !t.el.isConnected) continue;
      const b = t.el.getBoundingClientRect();
      const dx = a.left - b.left, dy = a.top - b.top, sx = a.width / b.width, sy = a.height / b.height;
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1 && Math.abs(sx - 1) < 0.01) continue;
      t.el.animate([{ transformOrigin: '0 0', transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})` },
                    { transformOrigin: '0 0', transform: 'none' }],
                   { duration: dur('--dur-scene') * 1.3, easing: EASE.move, delay: Math.min(i++ * 10, 260) });
    }
  }

  /** Size the tiles so the whole batch fits the stage; if that gets too small, scroll instead. */
  layout() {
    const n = this.keys.length;
    if (!n || !this.active) return;
    const W = this.scroller.clientWidth - 24, H = this.scroller.clientHeight - 24;
    if (W <= 0 || H <= 0) return;
    const gap = n > 24 ? 6 : 10;
    let best = { c: 1, w: 0 };
    for (let c = 1; c <= n; c++) {
      const r = Math.ceil(n / c);
      const w = Math.min((W - gap * (c - 1)) / c, ((H - gap * (r - 1)) / r) * TILE_ASPECT);
      if (w > best.w) best = { c, w };
    }
    let { c, w } = best;
    w = Math.min(w, n <= 2 ? 560 : 460); // the silhouette is 350 px wide; much larger only blurs it
    if (w < MIN_TILE) { c = Math.max(1, Math.floor((W + gap) / (MIN_TILE + gap))); w = (W - gap * (c - 1)) / c; }
    const s = this.root.style;
    s.setProperty('--cols', c);
    s.setProperty('--tw', `${Math.floor(w)}px`);
    s.setProperty('--th', `${Math.floor(w / TILE_ASPECT)}px`);
    s.setProperty('--tg', `${gap}px`);
    this.root.classList.toggle('tiny', w < 110);
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    for (const t of this.tiles.values()) {
      const cw = Math.round(t.mini.clientWidth * dpr), ch = Math.round(t.mini.clientHeight * dpr);
      if (cw && ch && (cw !== t.canvas.width || ch !== t.canvas.height)) { t.canvas.width = cw; t.canvas.height = ch; this.drawMarks(t); }
    }
  }

  // ---- live data ------------------------------------------------------------------------

  /** One 25 ms slice: the spikes of each test's first copy and each test's running MN9 rate. */
  live({ frames = [], mn9 = [] }, leader) {
    this.ref = mn9[0] ?? this.ref;
    this.scale = Math.max(this.scale, ...mn9.map(v => v * 1.15));
    this.batchKeys.forEach((k, i) => {
      const t = this.tiles.get(k);
      if (!t) return;
      if (frames[i]?.length) { t.spikes = frames[i]; t.lastSpike = performance.now(); }
      t.mn9 = mn9[i] ?? t.mn9;
      this.setMeter(t, t.mn9);
      if (!t.result) t.val.textContent = num(t.mn9);
      t.el.classList.toggle('lead', i === leader && i > 0);
    });
    if (this.follow && leader != null && this.batchKeys[leader]) this.select(this.batchKeys[leader], true);
  }

  setMeter(t, hz) {
    t.bar.style.width = `${Math.min(100, (hz / this.scale) * 100)}%`;
    if (this.ref != null && t.key !== this.batchKeys[0]) {
      t.refTick.hidden = false;
      t.refTick.style.left = `${Math.min(100, (this.ref / this.scale) * 100)}%`;
    }
  }

  // ---- results --------------------------------------------------------------------------

  /** Show a measured gain on a tile: delta over the batch's reference, with its noise. */
  annotate(key, { delta, sem, mn9, tag, cls }) {
    const t = this.tiles.get(key);
    if (!t) return;
    t.result = true;
    if (mn9 != null) { t.mn9 = mn9; this.setMeter(t, mn9); }
    t.val.textContent = sgn(delta);
    t.val.className = 'val' + (delta > 2 * sem ? ' up' : delta < -2 * sem ? ' down' : '');
    t.el.classList.toggle('helps', delta > 2 * sem);
    t.el.classList.toggle('hurts', delta < -2 * sem);
    if (cls) t.el.classList.add(cls);
    if (tag !== undefined) this.tag(t, tag);
    t.el.setAttribute('aria-label', `${t.name}: ${sgn(delta)} times a second compared with the yardstick, plus or minus ${num(sem)}`);
  }

  /** Put the reference first, then the rest by gain, best first. */
  sortBy(gainOf, instant) {
    const [first, ...rest] = this.keys;
    rest.sort((a, b) => (gainOf(b) ?? -1e9) - (gainOf(a) ?? -1e9));
    this.transition(() => {
      this.keys = [first, ...rest];
      for (const k of this.keys) this.root.appendChild(this.tiles.get(k).el);
    }, instant);
  }

  /** The screen of all 63 channels, from the stream or a saved run. */
  screen(msg, { instant = false } = {}) {
    const base = this.tiles.get('');
    if (base) { base.result = true; base.val.textContent = num(msg.baseline); this.tag(base, 'yardstick'); }
    this.ref = msg.baseline;
    const shortlist = new Set(msg.channels.filter(r => r.delta > r.sem).slice(0, 10).map(r => r.id));
    for (const r of msg.channels) {
      this.annotate(r.id, { delta: r.delta, sem: r.sem, mn9: r.mn9, tag: shortlist.has(r.id) ? 'shortlist' : '' });
      this.tiles.get(r.id)?.el.classList.toggle('dimmed', !shortlist.has(r.id) && r.delta <= 2 * r.sem);
    }
    const byId = new Map(msg.channels.map(r => [r.id, r.delta]));
    this.sortBy(k => byId.get(k), instant);
  }

  retest(channels, { instant = false } = {}) {
    for (const r of channels) {
      const pass = r.delta > 2 * r.sem;
      this.annotate(r.id, { delta: r.delta, sem: r.sem, tag: pass ? 'holds up' : 'luck', cls: pass ? '' : 'dimmed' });
    }
    const byId = new Map(channels.map(r => [r.id, r.delta]));
    this.sortBy(k => byId.get(k), instant);
  }

  round(msg, blend, { instant = false } = {}) {
    const baseHz = msg.mn9 - msg.gain;
    const byKey = new Map();
    for (const t of msg.tried) {
      const key = blendKey([...blend, t.add]);
      byKey.set(key, t.mn9 - baseHz);
      this.annotate(key, { delta: t.mn9 - baseHz, sem: msg.noise, mn9: t.mn9, tag: t.add === msg.added ? 'joins' : '',
                           cls: t.add === msg.added ? 'joined' : '' });
    }
    this.sortBy(k => byKey.get(k), instant);
  }

  confirm(c, blend) {
    const b = this.tiles.get(''), f = this.tiles.get(blendKey(blend));
    if (b) { b.result = true; b.val.textContent = num(c.baseline); }
    if (f && blend.length) this.annotate(f.key, { delta: c.gain, sem: c.noise, mn9: c.blend, tag: 'the answer', cls: 'joined' });
  }

  /** A saved run: the screen tiles without live spikes. */
  showSaved(result) {
    this.begin([[], ...result.screen.map(r => [r.id])], { instant: true });
    this.live({ mn9: [result.baseline, ...result.screen.map(r => r.mn9)] }, null);
    this.screen({ baseline: result.baseline, channels: result.screen }, { instant: true });
    this.follow = false;
  }

  clear() {
    for (const t of this.tiles.values()) t.el.remove();
    this.tiles.clear();
    this.keys = []; this.batchKeys = [];
  }

  // ---- selection ------------------------------------------------------------------------

  pick(key) {
    if (this.selected === key && !this.follow) { this.follow = true; this.onSelect(key, { follow: true }); this.markSelected(); return; }
    this.follow = false;
    this.select(key, false);
  }
  select(key, auto) {
    if (this.selected === key) return;
    this.selected = key;
    this.markSelected();
    const t = this.tiles.get(key);
    this.onSelect(key, { follow: auto, name: t?.name, cond: t?.cond, index: this.batchKeys.indexOf(key) });
  }
  markSelected() {
    for (const t of this.tiles.values()) t.el.setAttribute('aria-pressed', String(t.key === this.selected));
  }
  nameOf(key) { return this.tiles.get(key)?.name; }

  // ---- drawing --------------------------------------------------------------------------

  /** The front-view box of the brain inside a tile canvas (matches background-size: contain). */
  box(t) {
    const cw = t.canvas.width, ch = t.canvas.height, w = Math.min(cw, ch * 1.75), h = w / 1.75;
    return [(cw - w) / 2, (ch - h) / 2, w, h];
  }

  /** On a tile with no live spikes, show where its input's sensor cells sit. */
  drawMarks(t) {
    if (!this.px || !this.inputs || t.spikes || !t.ctx) return;
    const ctx = t.ctx, [ox, oy, w, h] = this.box(t);
    ctx.clearRect(0, 0, t.canvas.width, t.canvas.height);
    ctx.fillStyle = t.kind === 'smell' ? 'rgba(146,180,255,0.9)' : 'rgba(255,181,71,0.9)';
    for (const id of t.cond) for (const i of this.inputs.get(id) || []) ctx.fillRect(ox + this.px[i] * w - 1, oy + this.py[i] * h - 1, 2, 2);
  }

  loop(now) {
    requestAnimationFrame(this.loop);
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (!this.active || document.hidden || !this.px) return;
    const fade = 1 - Math.exp(-dt / this.tau);
    for (const t of this.tiles.values()) {
      const ctx = t.ctx;
      if (!ctx || !t.canvas.width) continue;
      if (t.spikes) {
        const [ox, oy, w, h] = this.box(t), s = Math.max(1.5, w / 110);
        ctx.globalCompositeOperation = 'lighter';
        ctx.fillStyle = 'rgba(170, 235, 255, 0.85)';
        for (const i of t.spikes) ctx.fillRect(ox + this.px[i] * w - s / 2, oy + this.py[i] * h - s / 2, s, s);
        ctx.globalCompositeOperation = 'source-over';
        t.spikes = null;
        t.fading = true;
      } else if (t.fading) {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = `rgba(0,0,0,${fade})`;
        ctx.fillRect(0, 0, t.canvas.width, t.canvas.height);
        ctx.globalCompositeOperation = 'source-over';
        if (now - t.lastSpike > this.tau * 5000) t.fading = false;
      }
    }
  }
}
