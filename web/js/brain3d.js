// The rotatable 3D brain: every neuron at its real soma position, lit by real spikes.
//
// One Points cloud holds all 138,639 neurons. Each has an "hot" time: the moment it last
// spiked. The shader turns (now - hot) into a glow that fades, so only spike arrival
// needs uploading, never per-frame animation data.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CLASSES } from './names.js';
import { motionOK } from './util.js';

const WORLD = 10;            // the brain's widest axis spans about 10 units
const MAX_DPR = 1.5;
const RAMP = [[255, 196, 90], [255, 110, 150], [120, 170, 255], [120, 255, 210]].map(c => c.map(v => v / 255));

/** Same colours as portrait._time_colour: warm at the mouth (0 ms), cool as the signal spreads (60+ ms). */
export function timeColour(ms) {
  const f = Math.min(1, Math.max(0, ms / 60)), k = Math.min(2, Math.floor(f * 3)), u = f * 3 - k;
  return RAMP[k].map((v, j) => v * (1 - u) + RAMP[k + 1][j] * u);
}

const VERT = /* glsl */`
  attribute vec3 aColor;
  attribute float aHot;
  attribute float aSize;
  uniform float uTime, uTau, uSize, uBase, uDim, uGlow, uGrow;
  varying vec3 vColor;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float age = uTime - aHot;
    float glow = age >= 0.0 ? exp(-age / uTau) : 0.0;
    vColor = aColor * uBase * uDim + vec3(0.72, 0.94, 1.0) * glow * uGlow;
    gl_PointSize = uSize * aSize * (1.0 + glow * uGrow) / -mv.z;
    gl_Position = projectionMatrix * mv;
  }`;
const FRAG = /* glsl */`
  uniform float uRing;
  varying vec3 vColor;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = dot(c, c) * 4.0;
    if (d > 1.0) discard;
    float a = mix(1.0 - d, smoothstep(0.55, 0.8, d) * (1.0 - smoothstep(0.85, 1.0, d)) * 2.0, uRing);
    gl_FragColor = vec4(vColor * a, 1.0);
  }`;

function pointsMaterial(u) {
  return new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uTau: { value: 0.8 }, uSize: { value: 1 }, uBase: { value: 1 }, uDim: { value: 1 }, uRing: { value: 0 }, uGlow: { value: 1.5 }, uGrow: { value: 2.2 }, ...u },
    vertexShader: VERT, fragmentShader: FRAG,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
}

export class Brain3D {
  /** anatomy: {n, x, y, z: Float32Array (0..1), cls: Uint8Array, classes: string[]} */
  constructor(host, anatomy) {
    this.host = host;
    this.a = anatomy;
    this.active = true;
    this.clock = new THREE.Clock();
    this.listeners = new Set();

    const r = this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance' });
    r.setPixelRatio(Math.min(devicePixelRatio || 1, MAX_DPR));
    r.setClearColor(0x000000, 0);
    host.appendChild(r.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 400);
    this.pos = this.positions();
    this.cloud = this.makeCloud();
    this.scene.add(this.cloud);
    this.marks = null;
    this.circuit = null;

    const c = this.controls = new OrbitControls(this.camera, r.domElement);
    c.enableDamping = motionOK();
    c.dampingFactor = 0.08;
    c.enablePan = false;
    c.rotateSpeed = 0.7;
    c.minDistance = 6;
    c.maxDistance = 90;
    c.autoRotate = motionOK();
    c.autoRotateSpeed = 0.35;
    c.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_ROTATE };
    let idle = 0;
    c.addEventListener('start', () => { c.autoRotate = false; clearTimeout(idle); this.userMoved = true; this.emit('interact'); });
    c.addEventListener('end', () => { idle = setTimeout(() => { c.autoRotate = motionOK(); }, 9000); });
    c.addEventListener('change', () => this.emit('view', this.angles()));

    host.addEventListener('keydown', e => this.onKey(e));
    new ResizeObserver(() => this.resize()).observe(host);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) this.clock.getDelta(); });
    this.resize();
    this.setView('front', false);
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  on(fn) { this.listeners.add(fn); }
  emit(kind, data) { for (const fn of this.listeners) fn(kind, data); }

  /** Real positions, centred, in world units (x right, y up, z towards the viewer). */
  positions() {
    const { n, x, y, z } = this.a;
    // centre of the bounding box, so the brain turns about its middle and frames evenly
    const mid = a => { let lo = Infinity, hi = -Infinity; for (let i = 0; i < n; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; } return (lo + hi) / 2; };
    const cx = mid(x), cy = mid(y), cz = mid(z);
    const p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      p[3 * i] = (x[i] - cx) * WORLD;
      p[3 * i + 1] = -(y[i] - cy) * WORLD;
      p[3 * i + 2] = -(z[i] - cz) * WORLD;
    }
    return p;
  }
  at(i) { return [this.pos[3 * i], this.pos[3 * i + 1], this.pos[3 * i + 2]]; }

  makeCloud() {
    const { n, cls, classes } = this.a;
    const palette = [CLASSES[''][1], ...classes.map(k => (CLASSES[k] || CLASSES[''])[1])].map(h => new THREE.Color(h));
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const c = palette[cls[i]] || palette[0]; col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b; }
    this.hot = new Float32Array(n).fill(-1e4);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aHot', new THREE.BufferAttribute(this.hot, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    this.cloudMat = pointsMaterial({ uBase: { value: 0.16 } });
    return new THREE.Points(g, this.cloudMat);
  }

  /** Light up these neurons now; they fade over the current time constant. */
  flash(indices) {
    const t = this.clock.elapsedTime;
    for (const i of indices) this.hot[i] = t;
    this.cloud.geometry.attributes.aHot.needsUpdate = true;
  }
  /** Seconds for a spike's glow to fade to a third; matched to how often frames arrive. */
  setTau(s) { this.cloudMat.uniforms.uTau.value = s; }

  /** Rings where the inputs enter (amber) and where MN9, the feeding neuron, sits (white). */
  setMarks({ inputs = [], mn9 = [] }) {
    if (this.marks) { this.scene.remove(this.marks); this.marks.geometry.dispose(); }
    const all = [...inputs.map(i => [i, 0]), ...mn9.map(i => [i, 1])];
    if (!all.length) { this.marks = null; return; }
    const p = new Float32Array(all.length * 3), col = new Float32Array(all.length * 3), size = new Float32Array(all.length);
    const amber = new THREE.Color('#ffb547'), white = new THREE.Color('#ffffff');
    all.forEach(([i, kind], k) => {
      p.set(this.at(i), 3 * k);
      const c = kind ? white : amber;
      col.set([c.r, c.g, c.b], 3 * k);
      size[k] = kind ? 9 : 4;
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aHot', new THREE.BufferAttribute(new Float32Array(all.length).fill(-1e4), 1));
    g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    this.marksMat ??= pointsMaterial({ uRing: { value: 1 }, uBase: { value: 1 } });
    this.marks = new THREE.Points(g, this.marksMat);
    this.scene.add(this.marks);
  }

  /**
   * The portrait's circuit in 3D: the neurons that fired and the strongest connections between them.
   * Connections appear in the order their sending neuron first fired, so the build-up follows the signal.
   */
  showCircuit(p, { animate = true } = {}) {
    this.clearCircuit();
    const local = new Map(p.index.map((gi, k) => [gi, k]));
    const first = p.neurons.map(n => n[2]);
    const edges = p.edges.filter(e => local.has(e[0]) && local.has(e[1]))
      .sort((a, b) => first[local.get(a[0])] - first[local.get(b[0])]);
    const lp = new Float32Array(edges.length * 6), lc = new Float32Array(edges.length * 6);
    const inhib = [140 / 255, 150 / 255, 1];
    edges.forEach(([a, b, syn, sign], k) => {
      lp.set(this.at(a), 6 * k); lp.set(this.at(b), 6 * k + 3);
      const w = Math.min(1, 0.35 + Math.log1p(syn) / 6);
      const c = (sign > 0 ? timeColour(first[local.get(a)]) : inhib).map(v => v * w);
      lc.set(c, 6 * k); lc.set(c.map(v => v * 0.55), 6 * k + 3);
    });
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(lp, 3));
    lg.setAttribute('color', new THREE.BufferAttribute(lc, 3));
    const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4, depthWrite: false }) /* normal blending: dense clusters stay coloured instead of burning white */);

    const n = p.index.length, np = new Float32Array(n * 3), nc = new Float32Array(n * 3), ns = new Float32Array(n);
    p.index.forEach((gi, k) => {
      np.set(this.at(gi), 3 * k);
      nc.set(timeColour(p.neurons[k][2]), 3 * k);
      ns[k] = 1 + 1.4 * Math.min(1, p.neurons[k][3] / 150);
    });
    const hot = new Float32Array(n).fill(-1e4);
    const ng = new THREE.BufferGeometry();
    ng.setAttribute('position', new THREE.BufferAttribute(np, 3));
    ng.setAttribute('aColor', new THREE.BufferAttribute(nc, 3));
    ng.setAttribute('aHot', new THREE.BufferAttribute(hot, 1).setUsage(THREE.DynamicDrawUsage));
    ng.setAttribute('aSize', new THREE.BufferAttribute(ns, 1));
    const mat = pointsMaterial({ uBase: { value: 0.32 }, uTau: { value: 0.3 }, uGlow: { value: 0.35 }, uGrow: { value: 0.8 } });
    const dots = new THREE.Points(ng, mat);

    const group = new THREE.Group();
    group.add(lines, dots);
    this.scene.add(group);
    const spikes = p.spikes || [];
    this.circuit = { group, lines, dots, mat, hot, local, spikes, edges: edges.length, t0: this.clock.elapsedTime, idx: 0, last: 0,
                     grow: animate && motionOK() ? 3.2 : 0 };
    lg.setDrawRange(0, this.circuit.grow ? 0 : edges.length * 2);
    this.cloudMat.uniforms.uDim.value = 0.35;
    return { edges: edges.length, neurons: n };
  }
  clearCircuit() {
    if (!this.circuit) return;
    this.scene.remove(this.circuit.group);
    this.circuit.group.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
    this.circuit = null;
    this.cloudMat.uniforms.uDim.value = 1;
  }
  /** Simulated ms of the looping 1 s trial replay (10 times slower than real time), or null. */
  get replayMs() { return this.circuit?.replayMs ?? null; }

  setView(name, animate = true) {
    const d = this.fitDistance();
    const dir = { front: [0, 0, 1], side: [1, 0, 0.02], top: [0, 1, 0.02] }[name] || [0, 0, 1];
    const to = new THREE.Vector3(...dir).normalize().multiplyScalar(d);
    this.controls.target.set(0, 0, 0);
    if (!animate || !motionOK()) { this.camera.position.copy(to); this.controls.update(); return; }
    this.fly = { from: this.camera.position.clone(), to, t0: performance.now(), ms: 700 };
  }
  fitDistance() {
    const half = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const aspect = this.camera.aspect || 1;
    return Math.max(3.4 / Math.tan(half), (WORLD * 0.56) / (Math.tan(half) * aspect)) + 2;
  }

  onKey(e) {
    const c = this.controls, off = this.camera.position.clone().sub(c.target);
    const s = new THREE.Spherical().setFromVector3(off), step = 0.12;
    if (e.key === 'ArrowLeft') s.theta -= step;
    else if (e.key === 'ArrowRight') s.theta += step;
    else if (e.key === 'ArrowUp') s.phi = Math.max(0.05, s.phi - step);
    else if (e.key === 'ArrowDown') s.phi = Math.min(Math.PI - 0.05, s.phi + step);
    else if (e.key === '+' || e.key === '=') s.radius = Math.max(c.minDistance, s.radius * 0.88);
    else if (e.key === '-' || e.key === '_') s.radius = Math.min(c.maxDistance, s.radius * 1.14);
    else if (e.key === '0') { this.setView('front'); return; }
    else return;
    e.preventDefault();
    c.autoRotate = false;
    this.camera.position.copy(c.target).add(new THREE.Vector3().setFromSpherical(s));
    c.update();
    this.emit('interact');
  }

  /**
   * The camera's angle, in the server's convention: yaw is the turn around the vertical axis
   * (0 = front), pitch the tilt (positive looks down from above). Whole degrees.
   */
  angles() {
    const deg = THREE.MathUtils.radToDeg, c = this.controls;
    let yaw = deg(c.getAzimuthalAngle());
    yaw = ((yaw + 180) % 360 + 360) % 360 - 180; // wrap into -180..180
    const pitch = Math.max(-85, Math.min(85, 90 - deg(c.getPolarAngle())));
    return { yaw: Math.round(yaw), pitch: Math.round(pitch) };
  }

  /** Turn the camera to a drawing's angle (the inverse of angles()). */
  setAngles({ yaw = 0, pitch = 0 }, animate = true) {
    const r = THREE.MathUtils.degToRad, d = this.camera.position.distanceTo(this.controls.target) || this.fitDistance();
    const to = new THREE.Vector3().setFromSpherical(new THREE.Spherical(d, r(90 - pitch), r(yaw)));
    this.controls.autoRotate = false;
    if (!animate || !motionOK()) { this.camera.position.copy(to); this.controls.update(); return; }
    this.fly = { from: this.camera.position.clone(), to, t0: performance.now(), ms: 700 };
  }

  setActive(on) { this.active = on; if (on) this.resize(); }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    const wasFit = !this.userMoved;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // world size of a dot -> pixels, so the cloud looks the same at any canvas size
    const px = this.renderer.getDrawingBufferSize(new THREE.Vector2()).y / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
    this.cloudMat.uniforms.uSize.value = 0.045 * px;
    if (this.marksMat) this.marksMat.uniforms.uSize.value = 0.05 * px;
    this.pxScale = px;
    if (wasFit && !this.fly) this.camera.position.setLength(this.fitDistance());
  }

  loop() {
    requestAnimationFrame(this.loop);
    const dt = this.clock.getDelta();
    if (!this.active || document.hidden) return;
    const t = this.clock.elapsedTime;
    this.cloudMat.uniforms.uTime.value = t;
    if (this.marksMat) this.marksMat.uniforms.uTime.value = t;

    if (this.fly) {
      const f = Math.min(1, (performance.now() - this.fly.t0) / this.fly.ms), e = 1 - Math.pow(1 - f, 3);
      this.camera.position.lerpVectors(this.fly.from, this.fly.to, e);
      if (f >= 1) this.fly = null;
    }

    const cc = this.circuit;
    if (cc) {
      cc.mat.uniforms.uTime.value = t;
      cc.mat.uniforms.uSize.value = 0.05 * this.pxScale;
      const age = t - cc.t0;
      if (cc.grow) {
        const f = Math.min(1, age / cc.grow);
        cc.lines.geometry.setDrawRange(0, Math.floor(f * f * cc.edges) * 2);
        if (f >= 1) cc.grow = 0;
      } else if (cc.spikes.length) {
        // replay the recorded 1 s trial, 10 times slower, on the circuit's own neurons
        const ms = ((performance.now() / 10) % 1000);
        if (ms < cc.last) cc.idx = 0;
        cc.last = ms;
        const s = cc.spikes;
        let changed = false;
        while (cc.idx < s.length / 2 && s[2 * cc.idx] <= ms) {
          const k = cc.local.get(s[2 * cc.idx + 1]);
          if (k !== undefined) { cc.hot[k] = t; changed = true; }
          cc.idx++;
        }
        if (changed) cc.dots.geometry.attributes.aHot.needsUpdate = true;
        cc.replayMs = ms;
      }
    }
    this.controls.update(dt);
    this.renderer.render(this.scene, this.camera);
  }
}
