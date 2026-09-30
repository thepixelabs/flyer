// "What is this?": the welcome story, one short chapter at a time. Opens by itself on a first visit.

import { $, $$, esc, clamp, store } from './util.js';

const SEEN = 'askthefly.seenStory';

export function initAbout() {
  const dlg = $('#about'), pages = $$('#aboutPages > section'), nav = $('#aboutNav');
  const prev = $('#aboutPrev'), next = $('#aboutNext'), count = $('#aboutCount');
  let i = 0, opener = null;

  nav.innerHTML = pages.map((s, k) => `<button type="button" data-k="${k}">${esc(s.dataset.title)}</button>`).join('');
  const tabs = $$('button', nav);

  function go(k) {
    i = clamp(k, 0, pages.length - 1);
    pages.forEach((s, j) => { s.hidden = j !== i; });
    pages[i].classList.remove('enter');
    void pages[i].offsetWidth; // restart the entrance animation
    pages[i].classList.add('enter');
    tabs.forEach((b, j) => (j === i ? b.setAttribute('aria-current', 'step') : b.removeAttribute('aria-current')));
    nav.scrollLeft = Math.max(0, tabs[i].offsetLeft - nav.clientWidth / 2 + tabs[i].offsetWidth / 2); // never scrollIntoView: it would scroll the page too
    count.textContent = `${i + 1} of ${pages.length}`;
    prev.disabled = i === 0;
    next.textContent = i === pages.length - 1 ? 'Start exploring' : 'Next';
    $('#aboutPages').scrollTop = 0;
  }

  function open(at = 0) {
    opener = document.activeElement;
    go(at);
    dlg.showModal();
    next.focus();
    store.set(SEEN, '1');
  }
  function close() { dlg.close(); }

  dlg.addEventListener('close', () => opener?.focus?.());
  dlg.addEventListener('click', e => { if (e.target === dlg) close(); }); // a click on the backdrop
  dlg.addEventListener('keydown', e => {
    if (e.target.closest('summary')) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); go(i + 1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); go(i - 1); }
  });
  nav.addEventListener('click', e => { const b = e.target.closest('button'); if (b) go(+b.dataset.k); });
  prev.addEventListener('click', () => go(i - 1));
  next.addEventListener('click', () => (i === pages.length - 1 ? close() : go(i + 1)));
  $('#aboutClose').addEventListener('click', close);
  $('#aboutBtn').addEventListener('click', () => open());

  return { open, firstVisit() { if (!store.get(SEEN)) open(); } };
}
