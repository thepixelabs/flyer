"""Render the data portrait as a hand-drawn artwork, keeping the data's structure.

The structure comes from two chained SDXL union ControlNets: the control map (real neurons, real
synapses, the brain's outline) in line-art mode, and -- always on -- a depth map of every real
neuron's 3D position, so the drawing's lobes follow real anatomy and not only the responding
circuit's lines. The palette is either prompt-driven ("ink", the default) or, for "data", comes
from an img2img start on the coloured data portrait. The prompt names only a medium and style,
never a subject, so the model restyles the lines without inventing content. `fidelity` then
measures how much of the drawing lies on the data and on the anatomy.
"""
from __future__ import annotations

import io
import json
import random
import uuid

import httpx
import numpy as np
import websocket
from PIL import Image, ImageFilter

from . import models

COMFY = "http://localhost:8188"
# Always-on anatomy ControlNet: tuned stronger than the 0.6/0.7 baseline found in an earlier
# depth-map experiment, so real anatomy clearly shapes the lobes without drowning the circuit
# lines -- judged by fidelity's `anatomy` correlation plus a visual check.
DEPTH_STRENGTH = 0.85
DEPTH_END = 0.85
# img2img denoise for palette="data": tuned toward the top of the 0.85-0.9 range that
# experiment recommended (0.8 alone measured 95% on_data but looked a bit flat).
DATA_DENOISE = 0.88

STYLES = {
    "cajal": {
        "label": "Cajal ink plate",
        "description": "Santiago Ram\u00f3n y Cajal (1852 to 1934) stained brain tissue with Camillo Golgi's silver "
                       "method, which blackens only a few whole neurons, and drew what he saw under the microscope in "
                       "India ink. His drawings helped establish that the nervous system is made of separate cells, "
                       "and he shared the 1906 Nobel Prize in Physiology or Medicine with Golgi.",
        "why": "It is the founding way of drawing neurons: single cells and their branches, drawn by hand from real "
               "tissue. Here the cells and connections come from the FlyWire data instead of a microscope slide.",
        "prompt": "antique scientific neuroanatomy plate drawn in the style of Santiago Ramon y Cajal, black india ink "
                  "pen drawing on aged cream paper, fine delicate linework, stippled cell bodies, hatching, "
                  "1900s histology illustration, archival scan, paper texture, masterful draughtsmanship",
        "negative": "color, colorful, photograph, 3d render, neon, glow, text, letters, numbers, labels, "
                    "annotations, watermark, signature, frame, border, blurry, digital art",
    },
    "etching": {
        "label": "Copperplate etching",
        "description": "Engraving and etching are intaglio prints: lines are cut into a copper plate with a burin, or "
                       "bitten into it with acid, then the plate is inked, wiped and pressed onto damp paper. It was a "
                       "main way to print the detailed plates in 18th- and 19th-century natural-history books.",
        "why": "The medium is made of fine, deliberate lines and cross-hatching, which suits a map of thin "
               "connections, and it comes from plates meant to record specimens accurately.",
        "prompt": "fine copperplate etching, engraved lines, cross-hatching, sepia ink on handmade paper, "
                  "19th century natural history engraving, intricate detail",
        "negative": "color, photograph, 3d render, neon, text, watermark, signature, frame, blurry",
    },
    "sumi": {
        "label": "Sumi-e ink wash",
        "description": "Sumi-e is the Japanese name for East Asian ink-wash painting, a tradition that grew up in "
                       "China. It uses a brush and black ink ground from an ink stick, with greys made by adding "
                       "water, and it prizes economy: ink on absorbent paper can't be corrected once laid down.",
        "why": "It is the loosest of the four styles, so it tests whether the data's structure survives expressive "
               "brushwork. The fidelity score shows how much of it does.",
        "prompt": "japanese sumi-e ink wash painting, expressive brush strokes, black ink on rice paper, "
                  "subtle grey washes, zen, minimal, elegant",
        "negative": "color, photograph, 3d render, text, watermark, signature, red seal, frame, blurry",
    },
    "cyanotype": {
        "label": "Cyanotype",
        "description": "A photographic process published by John Herschel in 1842: paper coated with iron salts turns "
                       "Prussian blue where light reaches it. Anna Atkins laid algae directly on such paper for "
                       "Photographs of British Algae (from 1843), widely considered the first book illustrated "
                       "with photographic images.",
        "why": "A cyanotype of a specimen is a direct trace, made without a hand drawing, much as this portrait is "
               "traced from data. Its white lines on deep blue also echo the white-on-black line map the model follows.",
        "prompt": "cyanotype print, prussian blue and white, sun print botanical photogram, delicate white lines "
                  "on deep blue paper, vintage scientific photogram",
        "negative": "photograph, 3d render, text, watermark, signature, frame, blurry, multicolor",
    },
}


def _graph(style: dict, image_name: str, seed: int, strength: float, w: int, h: int, *,
           checkpoint: str = models.CHECKPOINT, controlnet: str = models.CONTROLNET,
           depth_image_name: str | None = None, depth_strength: float = DEPTH_STRENGTH,
           depth_end: float = DEPTH_END, data_image_name: str | None = None, denoise: float = 1.0,
           prompt_prefix: str = "", negative_prefix: str = "") -> dict:
    positive = f"{prompt_prefix}{style['prompt']}"
    negative = f"{negative_prefix}{style['negative']}"
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet}},
        "6": {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["5", 0],
                                                                 "type": "canny/lineart/anime_lineart/mlsd"}},
        "7": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["2", 0], "negative": ["3", 0], "control_net": ["6", 0], "image": ["4", 0],
            "strength": strength, "start_percent": 0.0, "end_percent": 0.9, "vae": ["1", 2]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["8", 0],
            "seed": seed, "steps": 30, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "denoise": denoise}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "flybrain/portrait"}},
    }
    if depth_image_name is not None:  # always-on anatomy ControlNet, chained after the lineart one
        g["20"] = {"class_type": "LoadImage", "inputs": {"image": depth_image_name}}
        g["21"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["5", 0], "type": "depth"}}
        g["22"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["7", 0], "negative": ["7", 1], "control_net": ["21", 0], "image": ["20", 0],
            "strength": depth_strength, "start_percent": 0.0, "end_percent": depth_end, "vae": ["1", 2]}}
        g["9"]["inputs"]["positive"] = ["22", 0]
        g["9"]["inputs"]["negative"] = ["22", 1]
    if data_image_name is not None:  # palette="data": img2img from the colour data portrait
        g["23"] = {"class_type": "LoadImage", "inputs": {"image": data_image_name}}
        g["24"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["23", 0], "vae": ["1", 2]}}
        g["9"]["inputs"]["latent_image"] = ["24", 0]
    return g


def stylize(control_img: Image.Image, depth_img: Image.Image, style: str = "cajal", strength: float = 0.85,
            palette: str = "ink", data_img: Image.Image | None = None, denoise: float | None = None,
            checkpoint: str | None = None, prompt_prefix: str = "", negative_prefix: str = "",
            seed: int | None = None, on_progress=None, on_preview=None) -> Image.Image:
    """Render through ComfyUI, chaining the lineart ControlNet (`control_img`, at `strength`) with
    an always-on anatomy ControlNet (`depth_img`, at DEPTH_STRENGTH/DEPTH_END).

    palette="ink" (default): colours come only from the style prompt, starting from noise.
    palette="data": img2img from `data_img` (the coloured data portrait) at `denoise` (default
    DATA_DENOISE), so the palette comes from the data instead.

    `checkpoint` selects the SDXL checkpoint (default: the installed default checkpoint); `prompt_prefix`/
    `negative_prefix` add a checkpoint's own prompt conventions on top of the style's (see
    models.prompt_quirk). on_progress(step, total) reports sampling steps; on_preview(jpeg,
    step, total) receives ComfyUI's live preview of the image as it forms (TAESD decoder, one per
    step).
    """
    if palette not in ("ink", "data"):
        raise ValueError(f"unknown palette {palette!r}")
    if palette == "data" and data_img is None:
        raise ValueError("palette='data' needs data_img")
    if denoise is None:
        denoise = DATA_DENOISE if palette == "data" else 1.0
    seed = seed if seed is not None else random.randint(0, 2**31)
    buf = io.BytesIO()
    control_img.save(buf, "PNG")
    dbuf = io.BytesIO()
    depth_img.save(dbuf, "PNG")
    client_id = uuid.uuid4().hex
    ws = websocket.create_connection(f"{COMFY.replace('http', 'ws', 1)}/ws?clientId={client_id}", timeout=600)
    try:
        with httpx.Client(base_url=COMFY, timeout=60) as c:
            up = c.post("/upload/image", files={"image": ("flybrain_control.png", buf.getvalue(), "image/png")},
                        data={"overwrite": "true"}).json()
            dup = c.post("/upload/image", files={"image": ("flybrain_depth.png", dbuf.getvalue(), "image/png")},
                        data={"overwrite": "true"}).json()
            aup = None
            if palette == "data":
                abuf = io.BytesIO()
                data_img.save(abuf, "PNG")
                aup = c.post("/upload/image", files={"image": ("flybrain_data.png", abuf.getvalue(), "image/png")},
                             data={"overwrite": "true"}).json()
            graph = _graph(STYLES[style], up["name"], seed, strength, *control_img.size,
                           checkpoint=checkpoint or models.default_checkpoint(), controlnet=models.default_controlnet(),
                           depth_image_name=dup["name"],
                           data_image_name=aup["name"] if aup else None, denoise=denoise,
                           prompt_prefix=prompt_prefix, negative_prefix=negative_prefix)
            r = c.post("/prompt", json={"prompt": graph, "client_id": client_id,
                                        "extra_data": {"preview_method": "taesd"}})
            if r.status_code != 200:
                raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:500]}")
            pid = r.json()["prompt_id"]
            step, total = 0, graph["9"]["inputs"]["steps"]
            while True:
                msg = ws.recv()
                if isinstance(msg, bytes):
                    # binary frame: 4-byte event type (1 = preview image), 4-byte format, image bytes
                    if on_preview and int.from_bytes(msg[:4], "big") == 1:
                        on_preview(msg[8:], step, total)
                    continue
                m = json.loads(msg)
                d = m.get("data", {})
                if d.get("prompt_id") not in (None, pid):
                    continue
                if m["type"] == "progress":
                    step, total = d["value"], d["max"]
                    if on_progress:
                        on_progress(step, total)
                elif m["type"] == "execution_error":
                    raise RuntimeError(f"ComfyUI error: {d.get('exception_message', '')[:800]}")
                elif m["type"] == "executing" and d.get("node") is None:
                    break
            outs = [o for o in c.get(f"/history/{pid}").json()[pid]["outputs"].values() if "images" in o]
            im = outs[0]["images"][0]
            data = c.get("/view", params={"filename": im["filename"], "subfolder": im["subfolder"],
                                          "type": im["type"]}).content
            return Image.open(io.BytesIO(data)).convert("RGB")
    finally:
        ws.close()


def strict(control_img: Image.Image, drawing: Image.Image, reach: int = 9) -> Image.Image:
    """Keep the drawing only near real data lines; fade everything else to the paper tone."""
    ctrl = control_img.convert("L").point(lambda v: 255 if v > 127 else 0)
    mask = ctrl.filter(ImageFilter.MaxFilter(2 * reach + 1)).filter(ImageFilter.GaussianBlur(reach / 2))
    arr = np.asarray(drawing, np.float32)
    # paper tone = the drawing's own brightest common colour
    lum = arr.mean(2)
    paper = np.median(arr[lum >= np.quantile(lum, 0.8)], 0)
    m = np.asarray(mask, np.float32)[..., None] / 255.0
    return Image.fromarray(np.clip(arr * m + paper * (1 - m), 0, 255).astype(np.uint8))


def _edges(img: Image.Image, top: float) -> np.ndarray:
    """Strongest `top` fraction of Sobel gradient pixels."""
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(1)), np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1] = g[2:] - g[:-2]
    mag = np.hypot(gx, gy)
    return mag >= np.quantile(mag, 1 - top)


def fidelity(control_img: Image.Image, drawing: Image.Image, tol: int = 4,
            depth_img: Image.Image | None = None, silhouette: Image.Image | None = None) -> dict:
    """How faithful the drawing is to the data.

    on_data:  share of the drawing's strongest lines that lie on a data line (within tol px)
    coverage: share of the data lines that the drawing reproduces
    anatomy:  (only when depth_img is given) {corr, chance} -- how well the drawing's
              tonal structure matches the depth map inside the brain silhouette: Pearson
              correlation of blurred *ink density* (how far each pixel is from the drawing's own
              paper tone, the same notion strict() fades toward -- not raw luminance, since a
              dark-ink-on-light style like Cajal legitimately puts MORE ink, i.e. LOWER
              luminance, where the anatomy is nearer, which a raw-luminance correlation would
              misread as disagreement; measured on real renders and confirmed: it flips the sign
              on every dark-ink style here, while ink density comes out positive as intended,
              and the same density measure also fits a light-ink style like cyanotype) with the
              depth map, with a chance level from the same correlation against a left-right
              mirrored depth map (same statistics, wrong structure). `silhouette` restricts this
              to inside the brain (silhouette_mask()); it falls back to the depth map's own
              nonzero region when omitted.
    """
    ctrl = np.asarray(control_img.convert("L")) > 127
    near_ctrl = np.asarray(Image.fromarray(ctrl.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * tol + 1))) > 0
    e = _edges(drawing, top=max(0.02, ctrl.mean() * 1.5))
    near_e = np.asarray(Image.fromarray(e.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * tol + 1))) > 0
    on_data = float((e & near_ctrl).sum() / max(1, e.sum()))
    coverage = float((ctrl & near_e).sum() / max(1, ctrl.sum()))
    # chance level: a drawing with its lines placed at random
    chance = float(near_ctrl.mean())
    out = {"on_data": round(on_data, 3), "coverage": round(coverage, 3), "chance": round(chance, 3)}
    if depth_img is not None:
        dep = np.asarray(depth_img.convert("L"), np.float32)
        mask = np.asarray(silhouette.convert("L"), np.float32) > 127 if silhouette is not None else dep > 8
        if mask.sum() > 200:
            lum_full = np.asarray(drawing.convert("L"), np.float32)
            paper = np.median(lum_full[lum_full >= np.quantile(lum_full, 0.8)])  # same notion strict() uses
            ink = np.abs(lum_full - paper)
            lum = np.asarray(Image.fromarray(np.clip(ink, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)),
                             np.float32)
            a, b = lum[mask], dep[mask]
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else 0.0
            bm = dep[:, ::-1][mask]  # mirrored depth map: same brightness statistics, wrong structure
            corr_chance = float(np.corrcoef(a, bm)[0, 1]) if bm.std() > 1e-6 else 0.0
            out["anatomy"] = {"corr": round(corr, 3), "chance": round(corr_chance, 3)}
    return out
