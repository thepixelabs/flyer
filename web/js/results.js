// The answer and how sure we are: the right dock's first two tabs.

import { $, $$, esc, sgn, num, lower } from './util.js';

// the one explanation of the ± numbers, shown on hover or focus wherever one appears
const NOISE = 'The noise: roughly how far this number would move if everything were run again (one standard error). A result clearly helps when it is more than twice its noise.';
const pm = v => `<span class="tip-u" data-tip="${NOISE}">±${num(v)}</span>`;
import { info, learn, blendText } from './names.js';

export function resetResults() {
  $('#answerBox').hidden = true;
  $('#answerEmpty').hidden = false;
  $('#retestBox').hidden = true;
  $('#roundsBox').hidden = true;
  $('#rounds').innerHTML = '';
  $('#sureSummary').innerHTML = '<p class="empty">This fills in as the experiment runs: first the check of the promising tastes, then each round of building the blend.</p>';
}

/** The full result (from the stream or a saved run). */
export function renderResult(r) {
  learn(r.screen); learn(r.blend);
  const c = r.confirm, blend = r.blend || [], rounds = r.rounds || [], retest = r.retest || [];
  const held = c.gain > 2 * c.noise;
  const steps = rounds.filter(x => x.added);
  const stepOf = new Map(steps.map(x => [x.added, x]));
  $('#answerEmpty').hidden = true;
  $('#answerBox').hidden = false;
  $$('[data-fill]').forEach(e => { const v = r[e.dataset.fill]; if (v != null) e.textContent = v; });

  $('#answerTitle').innerHTML = blend.length
    ? 'Sugar' + blend.map(b => ` <span class="plus">+</span> <span class="ing">${esc(lower(info(b.id).short))}</span>`).join('')
    : 'Plain sugar';
  const smellsIn = blend.filter(b => b.kind === 'smell').length;
  const smellHelped = retest.some(q => q.id.startsWith('smell:') && q.delta > 2 * q.sem);
  $('#answerSub').textContent = !blend.length
    ? 'Nothing else raised the urge to eat by more than twice the noise, so plain sugar wins.'
    : smellsIn ? `Sugar plus ${blend.length} more, including ${smellsIn} smell${smellsIn > 1 ? 's' : ''}.`
    : smellHelped ? 'No smell made the cut. Some helped on their own, but none added enough on top of the tastes.'
    : 'No smell made it hungrier: none held up when checked again, so its favourite has no smell.';

  const ratio = c.noise > 0 ? c.gain / c.noise : 0;
  const pct = c.baseline > 0 ? (c.gain / c.baseline) * 100 : 0;
  $('#gain').innerHTML = `
    <div class="big"><span class="num ${!blend.length ? '' : held ? 'ok' : 'no'}">${sgn(c.gain)}</span>
      <span class="unit">times a second (${sgn(pct, 0)}%)</span>
      ${blend.length ? `<span class="pill ${held ? 'ok' : 'no'}" data-tip="${held ? 'The gain is more than twice its noise. See How sure.' : 'The gain is not more than twice its noise, so it could be luck. See How sure.'}">${held ? '✓ Clearly real' : '✗ Could be chance'}</span>` : ''}</div>
    <p>MN9, the feeding neuron: <b>${num(c.baseline)}</b> a second with sugar alone, <b>${num(c.blend)}</b> with the blend.
      <span class="tip-u" data-tip="Both were measured again from scratch for this number, with 32 fresh tries each.">Measured afresh</span>.</p>`;

  $('#ingredients').innerHTML =
    `<li class="ing-card base"><span class="stripe"></span><div class="nm"><span class="tip-u" data-tip="Sugar sensors are on in every test: the yardstick for everything else.">Sugar</span></div>
       <div class="add">${num(c.baseline)}<small>on its own</small></div></li>` +
    blend.map(b => {
      const i = info(b.id), x = stepOf.get(b.id);
      return `<li class="ing-card ${i.kind}"><span class="stripe"></span><div class="nm"><span class="tip-u" data-tip="${esc(i.what[0].toUpperCase() + i.what.slice(1))}.">${esc(i.short)}</span></div>
        <div class="add">${x ? sgn(x.gain) : ''}<small>${x ? `${pm(x.noise)}, round ${x.round}` : ''}</small></div></li>`;
    }).join('');

  renderWaterfall(c, steps);
  renderSure(r);
  $('#mBlend').textContent = blendText(blend.map(b => b.id));
}

function renderWaterfall(c, steps) {
  let cum = c.baseline;
  const rows = [{ k: 'Sugar alone', from: 0, to: c.baseline, cls: 'sugar', v: `<b>${num(c.baseline)}</b>` }];
  for (const x of steps) {
    rows.push({ k: '+ ' + lower(info(x.added).short), from: cum, to: cum + x.gain, cls: 'step', err: x.noise,
                v: `<span class="ok">${sgn(x.gain)}</span> ${pm(x.noise)}` });
    cum += x.gain;
  }
  if (steps.length) rows.push({ k: 'Final blend', total: true, from: 0, to: c.blend, cls: 'total', err: c.blend_sem,
                                v: `<b>${num(c.blend)}</b> ${pm(c.blend_sem || 0)}` });
  // start the axis a little below sugar alone so the steps are readable; the notch on the sugar bar shows the break
  const hiV = Math.max(cum, c.blend + (c.blend_sem || 0));
  const lo = steps.length ? Math.max(0, Math.floor((c.baseline - Math.max(5, hiV - c.baseline) * 0.5) / 10) * 10) : 0;
  const top = lo + Math.ceil((hiV - lo) / 10) * 10 || 10;
  const px = v => Math.max(0, Math.min(100, ((v - lo) / (top - lo)) * 100));
  $('#waterfall').innerHTML = rows.map(w => `
    <div class="k ${w.total ? 'total' : ''}">${esc(w.k)}</div>
    <div class="t">${w.cls === 'step' ? `<div class="guide" style="left:${px(w.from)}%"></div>` : ''}
      <div class="seg ${w.cls}${lo > 0 && w.from === 0 ? ' broken' : ''}" style="left:${px(w.from)}%;width:${px(w.to) - px(w.from)}%"></div>
      ${w.err ? `<div class="whisk" style="left:${px(w.to - w.err)}%;width:${px(w.to + w.err) - px(w.to - w.err)}%"></div>` : ''}</div>
    <div class="v">${w.v}</div>`).join('') +
    `<div class="axis" aria-hidden="true"><span>${lo}</span><span>${(lo + top) / 2}</span><span>${top} a second</span></div>`;
  $('#wfNote').innerHTML = steps.length
    ? `Amber: what each addition added in its round (<b>${sgn(cum - c.baseline)}</b> in all). Green: the final check
       (<b>${sgn(c.gain)}</b>). <span class="tip-u" data-tip="Each round measured its addition side by side with the blend so far; the final check used fresh tries, so the two needn't match exactly. White ticks show the noise.${lo > 0 ? ` The scale starts at ${lo} so the steps are visible.` : ''}">Why they differ</span>`
    : 'Nothing was added: no taste or smell beat twice the noise on top of sugar.';
}

/** "How sure are we?": the verdict first, then the checks that led to it. */
function renderSure(r) {
  const c = r.confirm, held = c.gain > 2 * c.noise, ratio = c.noise > 0 ? c.gain / c.noise : 0;
  $('#sureSummary').innerHTML = `<div class="gain">
      <div class="big"><span class="num ${held ? 'ok' : 'no'}">${ratio.toFixed(0)}×</span><span class="unit">the noise</span>
        <span class="pill ${held ? 'ok' : 'no'}">${held ? '✓ Clearly real' : '✗ Could be chance'}</span></div>
      <p>The gain, <b>${sgn(c.gain)}</b> a second, against its noise, <b>${pm(c.noise)}</b>. Real means more than twice the noise.
        ${held ? `This is ${ratio.toFixed(0)} times, so ${ratio >= 4 ? 'very unlikely' : 'unlikely'} to be luck.` : 'This is not, so treat the blend as no better than sugar alone.'}</p>
    </div>`;
  renderChosen(r.retest || [], r.rounds || [], new Set((r.blend || []).map(b => b.id)));
}

export function renderChosen(retest, rounds, inBlend = new Set()) {
  const rt = [...retest].sort((a, b) => b.delta - a.delta);
  $('#retestBox').hidden = !rt.length;
  if (rt.length) {
    const passed = rt.filter(q => q.delta > 2 * q.sem).length;
    $('#retestLede').innerHTML = `The ${rt.length} that looked helpful, tested again with many more tries: <b>${passed}</b>
      clearly helped (✓) and went on to the rounds.`;
    const lo = Math.min(0, ...rt.map(q => q.delta - q.sem)), hi = Math.max(1, ...rt.map(q => q.delta + q.sem));
    const x = v => ((v - lo) / (hi - lo)) * 100;
    $('#retest').innerHTML = rt.map(q => {
      const i = info(q.id), pass = q.delta > 2 * q.sem, a = x(Math.min(0, q.delta)), b = x(Math.max(0, q.delta));
      return `<div class="k ${inBlend.has(q.id) ? 'inblend' : ''}"><span class="tip-u" data-tip="${esc(i.what[0].toUpperCase() + i.what.slice(1))}.">${esc(i.short)}</span></div>
        <div class="t"><div class="zero" style="left:${x(0)}%"></div>
          <div class="seg ${pass ? 'pass' : 'fail'}" style="left:${a}%;width:${b - a}%"></div>
          <div class="whisk" style="left:${x(q.delta - q.sem)}%;width:${x(q.delta + q.sem) - x(q.delta - q.sem)}%"></div></div>
        <div class="v ${pass ? 'ok' : ''}"><b>${sgn(q.delta)}</b> ${pm(q.sem)} ${pass ? '✓' : ''}</div>`;
    }).join('') + (inBlend.size ? '<p class="fine" style="grid-column:1/-1">Amber names ended up in the blend. Gains are a second, over sugar alone.</p>' : '');
  }
  $('#roundsBox').hidden = !rounds.length;
  $('#rounds').innerHTML = rounds.map(x => {
    const base = x.mn9 - x.gain;
    const tried = [...x.tried].sort((a, b) => b.mn9 - a.mn9);
    const best = tried[0] ? tried[0].add : x.added;
    const head = x.added
      ? `added <b class="ok">${esc(info(x.added).short)}, ${sgn(x.gain)}</b> (±${num(x.noise)})`
      : `stopped: the best, ${esc(info(best).short)}, added ${sgn(x.gain)}, not more than twice the noise (±${num(x.noise)})`;
    const list = tried.map(t => {
      const s = `${esc(info(t.add).short)} ${sgn(t.mn9 - base)}`;
      return `<li>${t.add === x.added ? `<b>${s}</b>` : s}</li>`;
    }).join('');
    return `<li class="${x.added ? 'added' : 'stop'}"><details><summary><span class="rn">ROUND ${x.round}</span>: tried ${x.tried.length}, ${head}</summary>
      <ul class="tried"><li>what each one added on top of the blend so far:</li>${list}</ul></details></li>`;
  }).join('');
}
