// Small shared helpers.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export const esc = s => String(s ?? '').replace(/[&<>"']/g, ch =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));

/** Signed number with a real minus sign, e.g. +4.2 or −1.0. */
export const sgn = (v, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
export const num = (v, d = 1) => Number(v).toFixed(d);
export const int = n => Math.round(n).toLocaleString('en-US');
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const lower = s => s.charAt(0).toLowerCase() + s.slice(1);

const reduce = matchMedia('(prefers-reduced-motion: reduce)');
export const motionOK = () => !reduce.matches;

/** A duration token from styles.css in ms (0 under reduced motion). */
export function dur(name) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return motionOK() ? parseFloat(v) || 0 : 0;
}
export const EASE = { in: 'cubic-bezier(0, 0, 0.2, 1)', out: 'cubic-bezier(0.4, 0, 1, 1)', move: 'cubic-bezier(0.4, 0, 0.2, 1)' };

/** localStorage that never throws (private windows, blocked storage). */
export const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};

/** "20260926-221444" -> "26 Sep, 22:14" */
export function runDate(id) {
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(id || '');
  if (!m) return id || '';
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) + ', ' + m[4] + ':' + m[5];
}
export const clock = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

/** Calls fn(visible) when the element enters or leaves the viewport. */
export function watchVisible(el, fn) {
  const io = new IntersectionObserver(es => fn(es[es.length - 1].isIntersecting), { threshold: 0.01 });
  io.observe(el);
  return io;
}

/** Front view of the brain, as in portrait._proj: x fills the width, y is centred (canvas 1344 x 768). */
export const ART_W = 1344, ART_H = 768;
export const frontX = x => x;
export const frontY = y => (y - 0.5) * (ART_W / ART_H) + 0.5;
