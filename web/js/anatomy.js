// The positions and classes of every neuron, shared by the 3D brain and the wall.

/** Fetch /api/anatomy (N x 3 uint16 positions, then N uint8 class codes) with progress. */
export async function loadAnatomy(onProgress = () => {}) {
  const res = await fetch('/api/anatomy');
  if (!res.ok) throw new Error(`the server answered ${res.status}`);
  const n = +res.headers.get('X-Neurons');
  const classes = (res.headers.get('X-Classes') || '').split(',').filter(Boolean);
  const total = +res.headers.get('Content-Length') || n * 7;
  const buf = new Uint8Array(total);
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got); got += value.length;
    onProgress(got / total);
  }
  const dv = new DataView(buf.buffer);
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = dv.getUint16(6 * i, true) / 65535;
    y[i] = dv.getUint16(6 * i + 2, true) / 65535;
    z[i] = dv.getUint16(6 * i + 4, true) / 65535;
  }
  return { n, x, y, z, cls: buf.slice(6 * n, 7 * n), classes };
}
