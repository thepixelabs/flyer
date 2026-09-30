"""Draw the brain's response to its favourite flavour using only data.

Every dot is a neuron that fired, at its real soma position. Every stroke is a real synapse
(>= 5 synapses) between two neurons that fired. Colour = when the neuron first fired (the wave
spreading from the mouth). Size = firing rate, boosted for neurons recruited beyond sugar alone
(hz above their sugar-only base_hz). The faint outline is the density of all 138,639 neurons.

Three renders from the same data, at any view:
  art()      the coloured portrait
  control()  white lines on black, used as the ControlNet structure map
  depth()    a soft depth map of every neuron's real 3D position, nearer = brighter, used as a
             second, always-on ControlNet input so the drawing follows real anatomy

`view={"yaw": deg, "pitch": deg}` rotates all three around the brain's real 3D positions (see
_view_basis). view=None (or yaw=pitch=0) is the plain frontal view and is pixel-identical to the
original, view-less renders (see scripts/check_view_projection.py).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

W, H = 1344, 768  # the brain is ~2:1; SDXL-native size
RECRUIT_SCALE = 40.0  # Hz of (hz - base_hz) that counts as "fully" recruited, for stroke/soma boosts


def _proj(x, y):
    """Frontal view: normalised brain coords -> canvas pixels (brain fills the width)."""
    return x * W, (y - 0.5) * W + H / 2


def _world(xyz: np.ndarray) -> np.ndarray:
    """Normalised anatomy xyz (N, 3) in 0..1 -> world coords matching web/js/brain3d.js's
    positions(): x right, y up, z toward the viewer (both y and z are flipped from the raw,
    image-space anatomy axes, where y grows downward and z grows away from the viewer)."""
    w = np.asarray(xyz, np.float64) - 0.5
    w = w.copy()
    w[:, 1] *= -1
    w[:, 2] *= -1
    return w


def _view_basis(yaw: float, pitch: float):
    """Right/up/forward unit vectors for a camera orbiting the brain by `yaw` degrees (azimuth,
    around the vertical/world-Y axis) then `pitch` degrees (elevation), in the same right-handed,
    y-up convention THREE.Spherical (and so OrbitControls) already uses: theta (yaw) is measured
    around +Y from +Z toward +X, phi is measured from +Y, so pitch = 90 - phi (pitch 0 is the
    horizon = the frontal view; positive pitch looks down from above, matching
    `controls.getAzimuthalAngle()` / `90 - degrees(controls.getPolarAngle())` on the UI side).

    `forward` points from the target toward the camera, matching brain3d.js's world axes (x
    right, y up, z toward the viewer): at yaw=pitch=0, forward is (0, 0, 1).
    """
    theta, phi = np.radians(yaw), np.radians(90.0 - pitch)
    fwd = np.array([np.sin(phi) * np.sin(theta), np.cos(phi), np.sin(phi) * np.cos(theta)])
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, fwd)
    n = np.linalg.norm(right)
    right = right / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])  # looking straight up/down: pick a right arbitrarily
    up = np.cross(fwd, right)
    return right, up, fwd


def _proj_view(xyz: np.ndarray, view: dict | None = None):
    """Orthographic projection of normalised anatomy xyz (N, 3) to canvas pixels at `view`.

    view={"yaw": deg, "pitch": deg}, or None/both-zero for the frontal view. yaw=0, pitch=0
    always takes the exact legacy path (_proj on x, y directly, dropping z) so old output is
    reproduced pixel for pixel -- the general rotation math below is mathematically equivalent
    there too, but floating point (cos(pi/2) etc.) isn't exactly 0, so this short-circuits it.
    """
    if not view or (not view.get("yaw") and not view.get("pitch")):
        return _proj(xyz[:, 0], xyz[:, 1])
    right, up, _ = _view_basis(view.get("yaw", 0.0), view.get("pitch", 0.0))
    w = _world(xyz)
    u, v = w @ right, w @ up
    return u * W + W / 2, -v * W + H / 2  # v is world-up; canvas y grows downward


def _positions(neurons: list, view: dict | None = None) -> dict:
    """{neuron i: (canvas x, canvas y)} for every neuron in `neurons`, projected once."""
    xyz = np.array([[n["x"], n["y"], n.get("z", 0.5)] for n in neurons], np.float64)
    x, y = _proj_view(xyz, view)
    return {n["i"]: (float(x[k]), float(y[k])) for k, n in enumerate(neurons)}


def _recruit(n: dict, scale: float = RECRUIT_SCALE) -> float:
    """0..1: how much this neuron fired beyond its sugar-only baseline (max(0, hz - base_hz)),
    normalised by `scale` Hz. Falls back to 0 (no extra emphasis) when base_hz is unknown, so
    portraits built without it (shouldn't happen for live data, but be defensive) render as before."""
    return float(min(1.0, max(0.0, n["hz"] - n.get("base_hz", n["hz"])) / scale))


def strongest_edges(p: dict, max_edges: int, min_syn: int) -> list:
    """The strongest synaptic connections among responding neurons."""
    e = [x for x in p["edges"] if x[2] >= min_syn]
    e.sort(key=lambda x: -x[2])
    return e[:max_edges]


def _curve(p0, p1, bend=0.18, steps=14):
    """A gently curved stroke between two somata (neurites are rarely straight)."""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    t = np.linspace(0, 1, steps)[:, None]
    pts = (1 - t) ** 2 * np.array([x0, y0]) + 2 * (1 - t) * t * np.array([cx, cy]) + t ** 2 * np.array([x1, y1])
    return [tuple(p) for p in pts]


def _outline(all_xyz: np.ndarray, view: dict | None = None) -> np.ndarray:
    """Soft silhouette of the whole brain from the density of every neuron, at `view`."""
    dens = np.zeros((H, W), np.float32)
    x, y = _proj_view(all_xyz, view)
    px = np.clip(x.astype(int), 0, W - 1)
    py = np.clip(y.astype(int), 0, H - 1)
    np.add.at(dens, (py, px), 1)
    img = Image.fromarray(np.clip(dens * 60, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
    return np.asarray(img, np.float32) / 255.0


def silhouette_mask(all_xyz: np.ndarray, view: dict | None = None) -> Image.Image:
    """Filled brain silhouette (mode "L", 0/255) at `view`: closed (morphological closing), holes
    flood-filled, small islands dropped. The same shape control() outlines, exposed so fidelity()
    can restrict its anatomy-agreement metric to inside the brain."""
    sil = Image.fromarray(((_outline(all_xyz, view) > 0.05) * 255).astype(np.uint8))
    sil = sil.filter(ImageFilter.MaxFilter(21)).filter(ImageFilter.MinFilter(21)).filter(ImageFilter.GaussianBlur(4))
    sil = sil.point(lambda v: 255 if v > 127 else 0)
    bg = sil.copy()
    ImageDraw.floodfill(bg, (0, 0), 128)
    sil = bg.point(lambda v: 0 if v == 128 else 255)
    lab, n = ndimage.label(np.asarray(sil) > 0)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.02 * sizes.max()))
        sil = Image.fromarray((keep * 255).astype(np.uint8))
    return sil


def _time_colour(t_ms: float | None) -> tuple[int, int, int]:
    """Early spikes warm (the mouth), late spikes cool (spreading into the brain)."""
    if t_ms is None:
        return (160, 160, 180)
    f = min(1.0, t_ms / 60.0)
    stops = np.array([[255, 196, 90], [255, 110, 150], [120, 170, 255], [120, 255, 210]], float)
    k = min(2, int(f * 3))
    u = f * 3 - k
    return tuple(int(v) for v in stops[k] * (1 - u) + stops[k + 1] * u)


def art(p: dict, all_xyz: np.ndarray, max_edges: int = 3000, min_syn: int = 8,
        view: dict | None = None) -> Image.Image:
    neurons = {n["i"]: n for n in p["neurons"]}
    pos = _positions(p["neurons"], view)
    sil = _outline(all_xyz, view)
    base = np.zeros((H, W, 3), np.float32) + np.array([6, 7, 12], np.float32)
    base += sil[..., None] * np.array([40, 44, 70], np.float32)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")

    lines = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(lines)
    for a, b, syn, sign in strongest_edges(p, max_edges, min_syn):
        na, nb = neurons[a], neurons[b]
        c = _time_colour(na["first_ms"]) if sign > 0 else (140, 150, 255)
        rec = max(_recruit(na), _recruit(nb))  # neurons recruited beyond sugar alone draw heavier
        alpha = int(min(220, (25 + 12 * np.log1p(syn)) * (1 + 0.6 * rec)))
        width = 2 if rec > 0.6 else 1
        d.line(_curve(pos[a], pos[b]), fill=(*c, alpha), width=width)
    img = Image.alpha_composite(img, lines)

    dots = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(dots)
    for n in sorted(p["neurons"], key=lambda n: n["hz"]):
        x, y = pos[n["i"]]
        r = (1.5 + 2.5 * min(1.0, n["hz"] / 150)) * (1 + 0.5 * _recruit(n))
        c = _time_colour(n["first_ms"])
        d.ellipse([x - r, y - r, x + r, y + r], fill=(*c, 235))
    glow = dots.filter(ImageFilter.GaussianBlur(5))
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, dots)
    return img.convert("RGB")


def control(p: dict, all_xyz: np.ndarray, max_edges: int = 900, min_syn: int = 15,
            view: dict | None = None) -> Image.Image:
    """White-on-black line map: brain outline, synapse strokes, soma circles."""
    neurons = {n["i"]: n for n in p["neurons"]}
    pos = _positions(p["neurons"], view)
    img = Image.new("L", (W, H), 0)
    sil = silhouette_mask(all_xyz, view)
    img.paste(255, mask=sil.filter(ImageFilter.FIND_EDGES))
    d = ImageDraw.Draw(img)
    for a, b, syn, sign in strongest_edges(p, max_edges, min_syn):
        na, nb = neurons[a], neurons[b]
        rec = max(_recruit(na), _recruit(nb))  # neurons recruited beyond sugar alone draw heavier
        w = (1 if syn < 20 else 2) + (1 if rec > 0.5 else 0)
        d.line(_curve(pos[a], pos[b]), fill=255, width=w)
    for n in p["neurons"]:
        x, y = pos[n["i"]]
        r = (2 + 3 * min(1.0, n["hz"] / 150)) * (1 + 0.6 * _recruit(n))
        d.ellipse([x - r, y - r, x + r, y + r], outline=255, width=1)
    return img.convert("RGB")


def depth(all_xyz: np.ndarray, view: dict | None = None, blur: int = 6) -> Image.Image:
    """Depth map from the real 3D positions of every one of the 138,639 neurons: nearer (toward
    the viewer, at `view`) is brighter, blurred into a smooth anatomical relief. Rendered as a
    second, always-on ControlNet input (union type "depth", see stylize.py) chained after the
    lineart one, so the drawing's lobes follow real anatomy and not only the responding circuit's
    lines.
    """
    right, up, fwd = _view_basis(view.get("yaw", 0.0) if view else 0.0, view.get("pitch", 0.0) if view else 0.0)
    x, y = _proj_view(all_xyz, view)
    nearness = _world(all_xyz) @ fwd
    lo, hi = np.percentile(nearness, 1), np.percentile(nearness, 99)
    zn = np.clip((nearness - lo) / max(1e-6, hi - lo), 0, 1)
    acc = np.zeros((H, W), np.float64)
    cnt = np.zeros((H, W), np.float64)
    px = np.clip(x.astype(int), 0, W - 1)
    py = np.clip(y.astype(int), 0, H - 1)
    np.add.at(acc, (py, px), zn)
    np.add.at(cnt, (py, px), 1)
    d = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0)
    img = Image.fromarray((d * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img, np.float32)
    peak = arr.max()
    if peak > 0:
        arr = arr / peak * 255
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def from_saved(saved: dict) -> dict:
    """Reconstruct a portrait dict (the shape Experiment.portrait() returns: neuron dicts with
    i/x/y/z/hz/base_hz/first_ms/stim, plus edges/spikes/mn9) from an extended
    out/runs/<id>/portrait.json (see app.py's `work`/`draw`), so a saved run can be redrawn from
    any view/style/model/palette without re-running the simulation.

    Neuron rows there are [x, y, first_ms_or_0, hz, base_hz, stim, z] -- the first four positions
    are kept exactly where the older, reduced save (and web/js/brain3d.js, web/js/drawing.js,
    which still read this same file) expect them; base_hz/stim/z are appended, so older files
    (without them) are detected by row length and rejected here rather than silently mis-read.
    """
    neurons = []
    for i, t in zip(saved.get("index", []), saved.get("neurons", [])):
        if len(t) < 7:
            raise KeyError("portrait.json has no base_hz/z (an older, reduced save) -- can't be redrawn")
        x, y, first_ms, hz, base_hz, stim, z = t[:7]
        neurons.append({"i": int(i), "x": float(x), "y": float(y), "z": float(z), "first_ms": int(first_ms),
                        "hz": float(hz), "base_hz": float(base_hz), "stim": bool(stim)})
    if not neurons:
        raise KeyError("portrait.json has no neurons -- can't be redrawn")
    return {"neurons": neurons, "edges": saved.get("edges", []), "spikes": saved.get("spikes", []),
            "mn9": saved.get("mn9", [])}
