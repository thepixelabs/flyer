// Plain names for every input channel, and for the parts of the brain.

const TASTE = {
  water: ['Water', 'water sensors on the tip of the proboscis'],
  'low-salt': ['Low salt', 'sensors for a little salt, which flies find appetising'],
  bitter: ['Bitter', 'bitter sensors on the tip of the proboscis'],
  'taste peg': ['Taste pegs', 'taste pegs inside the mouth'],
  pharyngeal_nerve_sensory_group1: ['Throat taste 1', 'taste cells in the throat, group 1'],
  pharyngeal_nerve_sensory_group2: ['Throat taste 2', "taste cells in the throat, group 2 (what they detect isn't known yet)"],
  pharyngeal_nerve_sensory_group3: ['Throat taste 3', "taste cells in the throat, group 3 (what they detect isn't known yet)"],
  accessory_pharyngeal_nerve_sensory_group1: ['Back of throat 1', 'taste cells on a second throat nerve, group 1'],
  accessory_pharyngeal_nerve_sensory_group2: ['Back of throat 2', 'taste cells on a second throat nerve, group 2'],
  SA_VTV_pro_meso_meta: ['Legs and body', 'taste cells on the legs and body'],
};

// FlyWire "super class" of each neuron, in plain words, with its colour in the 3D view.
export const CLASSES = {
  optic: ['Optic lobes (vision)', '#3f78c8'],
  visual_projection: ['Vision relay cells', '#5f95e0'],
  visual_centrifugal: ['Feedback to vision', '#7d86e6'],
  central: ['Central brain', '#a7b2d6'],
  sensory: ['Sensory cells (taste, smell, touch)', '#f0a64a'],
  sensory_ascending: ['Sensory messages from the body', '#d99545'],
  ascending: ['Messages up from the body', '#4fc2a2'],
  descending: ['Commands down to the body', '#cf7fc4'],
  motor: ['Motor neurons (move muscles)', '#ff7f62'],
  endocrine: ['Hormone cells', '#e0cf6c'],
  '': ['Not labelled', '#56607a'],
};

const known = new Map(); // id -> {kind, name, receptor, detects, n, neurons}
export function learn(list) {
  for (const c of list || []) if (c && c.id) known.set(c.id, { ...known.get(c.id), ...c });
}
export const channel = id => known.get(id);

const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

/** Everything the page says about one channel id, e.g. "taste:water" or "smell:DA1". */
export function info(id) {
  const c = known.get(id) || {};
  const [kind, key = id] = String(id).split(':');
  const k = c.kind || kind;
  if (k === 'taste') {
    const t = TASTE[key];
    return { id, kind: 'taste', short: t ? t[0] : (c.name || key), what: t ? t[1] : (c.name || key), tag: 'taste' };
  }
  const detects = c.detects || '';
  const rec = c.receptor ? ` (receptor ${c.receptor})` : '';
  return {
    id, kind: 'smell', tag: `smell ${key}`,
    short: detects ? cap(detects.split(',')[0]) : `Smell ${key}`,
    what: `smell cells of the ${key} group${rec}` + (detects ? `, which respond to ${detects}` : ", whose smell isn't well known"),
  };
}

export const blendKey = list => (list || []).join('|');

/** Label for one test in a batch, relative to the batch's first test (the reference). */
export function testLabel(cond, ref) {
  if (!cond.length) return { name: 'Sugar alone', sub: 'the yardstick', kind: 'base' };
  const extra = cond.filter(c => !ref.includes(c));
  if (!extra.length) return { name: 'Blend so far', sub: 'sugar + ' + cond.map(c => info(c).short.toLowerCase()).join(' + '), kind: 'base' };
  const i = info(extra[0]);
  return ref.length
    ? { name: '+ ' + i.short, sub: 'on top of the blend', kind: i.kind }
    : { name: i.short, sub: i.kind === 'smell' ? i.tag : 'taste', kind: i.kind };
}

export const blendText = ids => 'sugar' + ids.map(id => ' + ' + info(id).short.toLowerCase()).join('');
