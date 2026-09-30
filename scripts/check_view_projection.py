"""Verify flybrain.portrait's view={"yaw", "pitch"} projection.

1. yaw=0, pitch=0 must reproduce the original, view-less art()/control() renders pixel for pixel
   (so adding views changed nothing for existing runs) -- checked against a small synthetic portrait, not a real
   saved run, so this needs no GPU/ComfyUI and runs in well under a second.
2. _view_basis: exact values at a few angles where the right answer is obvious by inspection
   (front, right, top), and that {right, up, forward} stay an orthonormal right-handed frame at
   an arbitrary angle.
3. depth(): a neuron placed at the near pole is brighter than one at the far pole, for several
   views, and at yaw=pitch=0 anatomy z=0 is nearest, i.e. brightest.

Usage: uv run python scripts/check_view_projection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flybrain import portrait  # noqa: E402

bad = 0


def check(name, cond):
    global bad
    print(f"{'ok  ' if cond else 'FAIL'} {name}")
    bad += not cond


# --- 1. yaw=0, pitch=0 reproduces the original frontal output exactly --------------------------

rng = np.random.default_rng(0)
n = 40
all_xyz = rng.random((500, 3)).astype(np.float32)
neurons = [{"i": i, "x": float(all_xyz[i, 0]), "y": float(all_xyz[i, 1]), "z": float(all_xyz[i, 2]),
           "hz": float(rng.uniform(5, 150)), "base_hz": float(rng.uniform(0, 60)),
           "first_ms": int(rng.uniform(0, 80)), "stim": bool(i < 3), "cls": "", "type": ""}
          for i in range(n)]
pre = rng.integers(0, n, 300)
post = rng.integers(0, n, 300)
syn = rng.integers(1, 40, 300)
sign = rng.choice([-1, 1], 300)
p = {"neurons": neurons, "edges": np.stack([pre, post, syn, sign], 1).tolist(), "spikes": [], "mn9": []}

art_legacy = portrait.art(p, all_xyz)
art_view0 = portrait.art(p, all_xyz, view={"yaw": 0.0, "pitch": 0.0})
art_viewNone = portrait.art(p, all_xyz, view=None)
check("art(): view={'yaw':0,'pitch':0} == view-less, byte for byte",
      np.array_equal(np.asarray(art_legacy), np.asarray(art_view0)))
check("art(): view=None == view-less, byte for byte",
      np.array_equal(np.asarray(art_legacy), np.asarray(art_viewNone)))

ctrl_legacy = portrait.control(p, all_xyz)
ctrl_view0 = portrait.control(p, all_xyz, view={"yaw": 0.0, "pitch": 0.0})
check("control(): view={'yaw':0,'pitch':0} == view-less, byte for byte",
      np.array_equal(np.asarray(ctrl_legacy), np.asarray(ctrl_view0)))

# a nonzero view must actually change the render (sanity: the shortcut isn't masking a no-op elsewhere)
ctrl_rot = portrait.control(p, all_xyz, view={"yaw": 35.0, "pitch": 12.0})
check("control(): a nonzero view changes the render", not np.array_equal(np.asarray(ctrl_legacy), np.asarray(ctrl_rot)))

# --- 2. _view_basis ------------------------------------------------------------------------------

r, u, f = portrait._view_basis(0.0, 0.0)
check("_view_basis(0,0): forward is +Z (toward the viewer)", np.allclose(f, [0, 0, 1], atol=1e-9))
check("_view_basis(0,0): right is +X", np.allclose(r, [1, 0, 0], atol=1e-9))
check("_view_basis(0,0): up is +Y", np.allclose(u, [0, 1, 0], atol=1e-9))

r, u, f = portrait._view_basis(90.0, 0.0)
check("_view_basis(90,0): forward is +X (the right side of the brain)", np.allclose(f, [1, 0, 0], atol=1e-9))

r, u, f = portrait._view_basis(0.0, 90.0)
check("_view_basis(0,90): forward is +Y (looking straight down)", np.allclose(f, [0, 1, 0], atol=1e-9))

for yaw, pitch in [(0, 0), (37, -21), (180, 0), (-90, 60), (12.5, 84)]:
    r, u, f = portrait._view_basis(yaw, pitch)
    ortho = (abs(np.dot(r, u)) < 1e-9 and abs(np.dot(u, f)) < 1e-9 and abs(np.dot(r, f)) < 1e-9)
    unit = all(abs(np.linalg.norm(v) - 1) < 1e-9 for v in (r, u, f))
    right_handed = np.allclose(np.cross(r, u), f, atol=1e-6)
    check(f"_view_basis({yaw},{pitch}): orthonormal, right-handed", ortho and unit and right_handed)

# --- 3. depth(): nearer is brighter, and the frontal convention holds -------------

near_far = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]] * 2000, np.float32)  # [near, far, near, far, ...]
d = np.asarray(portrait.depth(near_far).convert("L"))
near_px = d[int(portrait.H * 0.5), int(portrait.W * 0.5)]
check("depth(): frontal, anatomy z=0 (near) renders bright at that point",
      near_px > 100)  # both near+far land on the same pixel and average, so this just isn't near-black

near_only = np.array([[0.2, 0.5, 0.0]] * 100, np.float32)  # different xy from far_only, so they
far_only = np.array([[0.8, 0.5, 1.0]] * 100, np.float32)   # land on different pixels
mixed = np.concatenate([near_only, far_only, rng.random((300, 3)).astype(np.float32)])
dm = np.asarray(portrait.depth(mixed).convert("L"), np.float32)
cx, cy = portrait._proj_view(near_only[:1], None)
nx, ny = int(cx[0]), int(cy[0])
fx, fy = portrait._proj_view(far_only[:1], None)
fxp, fyp = int(fx[0]), int(fy[0])
check("depth(): the near point is brighter than the far point (frontal)", dm[ny, nx] > dm[fyp, fxp])

for yaw, pitch in [(90, 0), (0, 60), (45, -30)]:
    view = {"yaw": yaw, "pitch": pitch}
    dv = np.asarray(portrait.depth(mixed, view=view).convert("L"), np.float32)
    check(f"depth(): renders without error at view={view}", dv.shape == (portrait.H, portrait.W))

print(f"\n{'ALL OK' if not bad else f'{bad} FAILED'}")
sys.exit(1 if bad else 0)
