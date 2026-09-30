// Tooltips: secondary explanations shown on hover, keyboard focus or tap.
//
// Mark any element with data-tip="plain text" (or data-tip-src="id" to reuse a hidden element's
// HTML). Each trigger gets a permanent aria-describedby pointing at a hidden copy of its text,
// so screen readers hear it on focus; the visible bubble is one floating layer above the docks,
// which never gets clipped by their scroll areas. Escape closes it; a tap toggles it on touch.

const layer = document.createElement('div');
layer.className = 'tip-layer';
layer.setAttribute('aria-hidden', 'true'); // the text is already announced through aria-describedby
layer.hidden = true;
const store = document.createElement('div');
store.hidden = true;
let owner = null, pinned = false, suppressed = null, seq = 0;

const SEL = '[data-tip], [data-tip-src]';
const trigger = el => el?.closest?.(SEL);

/** Give every trigger a hidden description it can point at (runs again whenever the page changes). */
function wire(root = document) {
  for (const t of root.querySelectorAll(SEL)) {
    if (t.dataset.tipSrc) { t.setAttribute('aria-describedby', t.dataset.tipSrc); continue; }
    let d = t.dataset.tipId && document.getElementById(t.dataset.tipId);
    if (!d) {
      d = document.createElement('span');
      d.id = t.dataset.tipId = `tip-${++seq}`;
      store.append(d);
    }
    if (d.textContent !== t.dataset.tip) d.textContent = t.dataset.tip;
    t.setAttribute('aria-describedby', d.id);
    if (!t.matches('button, a, input, select, textarea, label, [tabindex]')) t.tabIndex = 0;
  }
}

function show(t, pin = false) {
  const html = t.dataset.tipSrc ? document.getElementById(t.dataset.tipSrc)?.innerHTML : null;
  if (html == null && !t.dataset.tip) return;
  owner = t;
  pinned = pin;
  if (html != null) layer.innerHTML = html; else layer.textContent = t.dataset.tip;
  layer.hidden = false;
  place(t);
}
function hide() { owner = null; pinned = false; layer.hidden = true; }

/** Below the trigger if it fits, otherwise above; always inside the window. */
function place(t) {
  const r = t.getBoundingClientRect(), pad = 8;
  layer.style.maxWidth = `${Math.min(320, innerWidth - 2 * pad)}px`;
  layer.style.left = '0px'; layer.style.top = '0px';
  const w = layer.offsetWidth, h = layer.offsetHeight;
  const left = Math.max(pad, Math.min(innerWidth - w - pad, r.left + r.width / 2 - w / 2));
  const below = r.bottom + 6, above = r.top - h - 6;
  const top = below + h <= innerHeight - pad || above < pad ? below : above;
  layer.style.left = `${Math.round(left)}px`;
  layer.style.top = `${Math.round(Math.max(pad, Math.min(innerHeight - h - pad, top)))}px`;
}

/** Change a trigger's text later (for example when the chosen style changes). */
export function setTip(el, text) {
  if (!el) return;
  el.dataset.tip = text;
  wire(el.parentElement || document);
  if (owner === el) show(el, pinned);
}

export function initTips() {
  document.body.append(layer, store);
  wire();
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; wire(); });
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-tip', 'data-tip-src'] });

  document.addEventListener('pointerover', e => {
    if (e.pointerType !== 'mouse') return;
    const t = trigger(e.target);
    if (t && t !== owner && t !== suppressed && !pinned) show(t);
  });
  document.addEventListener('pointerout', e => {
    if (e.pointerType !== 'mouse') return;
    const t = trigger(e.target);
    if (!t || t.contains(e.relatedTarget)) return;
    if (t === suppressed) suppressed = null;
    if (t === owner && !pinned && document.activeElement !== t && !t.contains(document.activeElement)) hide();
  });
  document.addEventListener('focusin', e => {
    const t = trigger(e.target);
    if (t && t !== suppressed) show(t, pinned && t === owner);
  });
  document.addEventListener('focusout', e => {
    const t = trigger(e.target);
    if (t && t === owner && !t.contains(e.relatedTarget)) hide();
    if (t === suppressed) suppressed = null;
  });
  document.addEventListener('click', e => {
    if (e.target.tagName === 'INPUT') return; // the second, synthetic click a <label> sends to its radio
    const t = trigger(e.target);
    if (!t) { if (pinned && !layer.contains(e.target)) hide(); return; }
    if (t.matches('select')) return;
    if (owner === t && pinned) hide(); else show(t, true);
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && owner) { suppressed = owner; hide(); }
  });
  addEventListener('resize', () => owner && place(owner));
  document.addEventListener('scroll', () => owner && place(owner), true);
}
