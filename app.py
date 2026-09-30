"""Flyer: let a simulated fly brain choose its favourite flavour, then draw its response.

Run: uv run uvicorn app:app --port 8420   (then open http://localhost:8420)

One experiment runs at a time. Its events are kept on the server, so any page
(including one that was refreshed) can attach to the run in progress with
/api/stream and receive everything that has happened so far, then live updates.
"""
from __future__ import annotations

import base64
import json
import queue
import re
import shutil
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from flybrain import models, portrait, stylize
from flybrain.experiment import Experiment

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "out" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
RUN_ID = re.compile(r"\d{8}-\d{6}")  # run folders are named by timestamp (see start)

# Events that are replaced by the next one of the same kind rather than kept in the log:
# a page that attaches late only needs the latest frame of these.
TRANSIENT = {"live", "preview", "progress"}

# Third-party assets are vendored under web/ (see web/vendor and web/fonts), so this hash is the
# only inline script left: the importmap that points "three" at its local file.
IMPORTMAP_SHA256 = "sha256-UXCJSeGZv4Pmn69ikzRk9QkqV1KGwt6IUlzSJdOTWdM="
CSP = ("default-src 'self'; img-src 'self' data: blob:; connect-src 'self'; "
       # the page sets element.style.* directly in several places (progress bars, opacity fades) -
       # CSP treats that the same as an inline style attribute, and those values are computed at
       # runtime so they can't be hashed
       f"style-src 'self' 'unsafe-inline'; script-src 'self' '{IMPORTMAP_SHA256}'")

app = FastAPI(title="Flyer")
# This is a local, single-user tool: reject any request whose Host header isn't localhost or
# 127.0.0.1 (any port), which blocks DNS rebinding attacks from a page on another domain.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")


@app.middleware("http")
async def revalidate_page_files(request, call_next):
    """Make browsers re-check the page, styles and scripts on every load, so an update is never masked by a cache."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/web/"):
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Content-Security-Policy"] = CSP
    return response


def cross_site(request: Request) -> bool:
    """True if an Origin header is present and doesn't match this request's own Host (as localhost or 127.0.0.1).

    Same-origin browser requests either omit Origin or set it to the page's own origin, so this only
    ever rejects a request made by a page loaded from somewhere else (e.g. an evil.example fetch/img).
    """
    origin = request.headers.get("origin")
    if origin is None:
        return False
    host = request.headers.get("host", "")
    port = f":{host.split(':', 1)[1]}" if ":" in host else ""
    return origin not in (f"http://localhost{port}", f"http://127.0.0.1{port}")


exp = Experiment()


class RunHub:
    """The current experiment's event log, fanned out to every attached page."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.run_id: str | None = None
        self.params: dict = {}
        self.log: list[dict] = []
        self.latest: dict[str, dict] = {}
        self.subscribers: list[queue.Queue] = []

    def begin(self, run_id: str, params: dict) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running, self.run_id, self.params = True, run_id, params
            self.log, self.latest = [], {}
            return True

    def publish(self, msg: dict):
        with self.lock:
            if msg["kind"] in TRANSIENT:
                self.latest[msg["kind"]] = msg
            else:
                self.log.append(msg)
            subs = list(self.subscribers)
        for q in subs:
            # a slow page skips live frames rather than falling behind
            if msg["kind"] in TRANSIENT and q.qsize() > 50:
                continue
            q.put(msg)

    def end(self):
        with self.lock:
            self.running = False
            subs = list(self.subscribers)
        for q in subs:
            q.put(None)

    def attach(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            q.put({"kind": "attached", "run": self.run_id, "running": self.running, **self.params})
            for msg in self.log:
                q.put(msg)
            for msg in self.latest.values():
                q.put(msg)
            if self.running:
                self.subscribers.append(q)
            else:
                q.put(None)
        return q

    def detach(self, q: queue.Queue):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


hub = RunHub()


def _validate_model(name: str | None) -> str:
    """The default checkpoint is always allowed with no network round trip; anything else must be
    marked qualified in out/models.json (see scripts/qualify_models.py)."""
    if not name:
        return models.default_checkpoint()
    if models.is_default_checkpoint(name):
        return name
    try:
        ok = models.is_selectable(name, results_path=ROOT / "out" / "models.json")
    except httpx.HTTPError as e:
        raise HTTPException(503, f"couldn't reach ComfyUI to validate the model: {e}")
    if not ok:
        raise HTTPException(400, f"model {name!r} isn't qualified (see GET /api/models)")
    return name


def _validate_palette(palette: str) -> str:
    if palette not in ("ink", "data"):
        raise HTTPException(400, f"unknown palette {palette!r}")
    return palette


def _maps(p: dict, view: dict):
    """The three always-computed renders for a portrait at `view`: lineart control map, anatomy
    depth map, and the coloured data portrait (used for display, and as the img2img source when
    palette="data")."""
    return (portrait.control(p, exp.anatomy.xyz, view=view),
            portrait.depth(exp.anatomy.xyz, view=view),
            portrait.art(p, exp.anatomy.xyz, view=view))


def _finish_drawing(run_dir: Path, index: int, p: dict, ctrl, depth_img, data_img, *, style: str, model: str,
                    palette: str, strength: float, yaw: float, pitch: float, emit, run_id: str) -> dict:
    """Sample the drawing (through ComfyUI) and save everything under
    out/runs/<run_id>/drawings/<index>/: control/depth/data/free/strict images plus params.json.
    Shared by the first drawing (work(), right after an experiment) and by /api/draw on a saved
    run. Emits the same "stage"/"progress"/"preview"/"drawing" events either way; at index 0 the
    "drawing" event's top-level strict/free URLs stay the pre-existing top-level ones, for
    backward compatibility with web/js/drawing.js.
    """
    problem = models.setup_problem(model)
    if problem:
        raise models.SetupError(problem)
    d = run_dir / "drawings" / str(index)
    d.mkdir(parents=True, exist_ok=True)
    ctrl.save(d / "control.png")
    depth_img.save(d / "depth.png")
    data_img.save(d / "data.png")
    emit("stage", stage="draw", text=f"Drawing it as a {stylize.STYLES[style]['label']}")
    quirk = models.prompt_quirk(model) or {}
    free = stylize.stylize(
        ctrl, depth_img, style=style, strength=strength, palette=palette,
        data_img=data_img if palette == "data" else None, checkpoint=model,
        prompt_prefix=quirk.get("positive_prefix", ""), negative_prefix=quirk.get("negative_prefix", ""),
        on_progress=lambda step, total: emit("progress", stage="draw", value=step / max(1, total),
                                             step=step, total=total),
        on_preview=lambda jpg, step, total: emit("preview", step=step, total=total,
                                                 image="data:image/jpeg;base64," + base64.b64encode(jpg).decode()))
    free.save(d / "free.png")
    strict = stylize.strict(ctrl, free)
    strict.save(d / "strict.png")
    silhouette = portrait.silhouette_mask(exp.anatomy.xyz, view={"yaw": yaw, "pitch": pitch})
    fid = {"free": stylize.fidelity(ctrl, free, depth_img=depth_img, silhouette=silhouette),
          "strict": stylize.fidelity(ctrl, strict, depth_img=depth_img, silhouette=silhouette)}
    params = {"index": index, "style": style, "model": model, "palette": palette, "yaw": yaw, "pitch": pitch,
             "strength": strength, "fidelity": fid, "created": time.time()}
    (d / "params.json").write_text(json.dumps(params, indent=1))
    base = f"/runs/{run_id}/drawings/{index}"
    record = {**params, "control": f"{base}/control.png", "depth": f"{base}/depth.png",
             "data": f"{base}/data.png", "free": f"{base}/free.png", "strict": f"{base}/strict.png"}
    top_free = f"/runs/{run_id}/free.png" if index == 0 else record["free"]
    top_strict = f"/runs/{run_id}/strict.png" if index == 0 else record["strict"]
    emit("drawing", strict=top_strict, free=top_free, fidelity=fid, run=run_id, drawing=record)
    return record


def _error_text(e: Exception) -> str:
    """What the page shows for a failed job: setup problems already read as plain sentences."""
    return str(e) if isinstance(e, models.SetupError) else f"{type(e).__name__}: {e}"


def work(run_id: str, style: str, trials: int, strength: float, model: str, palette: str, yaw: float, pitch: float):
    out = RUNS / run_id
    out.mkdir()
    emit = lambda kind, **data: hub.publish({"kind": kind, **data})
    try:
        res = exp.run(n_run=trials, seed=int(time.time()) % 100000, emit=emit)
        p = res["portrait"]
        view = {"yaw": yaw, "pitch": pitch}
        ctrl, depth_img, data_img = _maps(p, view)
        # the files a fresh page's <img> tags need must exist before the "portrait" event below
        # announces their URLs, so save them (both under drawings/0/ and, for backward
        # compatibility, at the run's top level) before emitting anything
        d0 = out / "drawings" / "0"
        d0.mkdir(parents=True)
        ctrl.save(d0 / "control.png")
        depth_img.save(d0 / "depth.png")
        data_img.save(d0 / "data.png")
        shutil.copyfile(d0 / "control.png", out / "control.png")
        shutil.copyfile(d0 / "data.png", out / "data.png")

        # neuron rows: [x, y, first_ms_or_0, hz, base_hz, stim, z] -- the first four stay exactly
        # where web/js/brain3d.js and web/js/drawing.js already read them (x, y, first_ms, hz);
        # base_hz/stim/z are appended so any saved run can be redrawn later from any view
        # (flybrain.portrait.from_saved), without disturbing either of those existing readers.
        neuron_rows = [[n["x"], n["y"], n["first_ms"] or 0, n["hz"], n["base_hz"], int(n["stim"]), n["z"]]
                       for n in p["neurons"]]
        full = {"neurons": neuron_rows, "index": [n["i"] for n in p["neurons"]], "spikes": p["spikes"],
               "edges": p["edges"], "n_neurons": len(p["neurons"]), "n_edges": len(p["edges"]),
               "mn9": [int(i) for i in p["mn9"]]}
        (out / "portrait.json").write_text(json.dumps(full))
        # the live event (and the saved run's 3D circuit view) keep the same 1500-edge cap as
        # before, for the same reason it was there originally (perf); portrait.json above has the
        # FULL edge list, for a future redraw's own fidelity
        event_edges = [[a, b, s, sg] for a, b, s, sg in portrait.strongest_edges(p, 1500, 8)]
        emit("portrait", data=f"/runs/{run_id}/data.png", control=f"/runs/{run_id}/control.png",
             **{**full, "edges": event_edges})

        d0_record = _finish_drawing(out, 0, p, ctrl, depth_img, data_img, style=style, model=model, palette=palette,
                                    strength=strength, yaw=yaw, pitch=pitch, emit=emit, run_id=run_id)
        shutil.copyfile(d0 / "free.png", out / "free.png")
        shutil.copyfile(d0 / "strict.png", out / "strict.png")

        meta = {"result": res["result"], "style": style, "strength": strength, "model": model, "palette": palette,
               "yaw": yaw, "pitch": pitch, "fidelity": d0_record["fidelity"], "n_neurons": full["n_neurons"],
               "n_edges": full["n_edges"], "drawings": [d0_record]}
        (out / "run.json").write_text(json.dumps(meta, indent=1))
        emit("done", run=run_id)
    except Exception as e:  # surface every failure in the UI
        emit("error", text=_error_text(e))
    finally:
        hub.end()


def redraw_work(run_dir: Path, run_id: str, index: int, p: dict, style: str, model: str, palette: str,
                yaw: float, pitch: float, strength: float):
    """POST /api/draw's background job: render a new drawing for an already-saved run's portrait,
    without touching the simulation or the run's existing files."""
    emit = lambda kind, **data: hub.publish({"kind": kind, **data})
    try:
        view = {"yaw": yaw, "pitch": pitch}
        ctrl, depth_img, data_img = _maps(p, view)
        record = _finish_drawing(run_dir, index, p, ctrl, depth_img, data_img, style=style, model=model,
                                 palette=palette, strength=strength, yaw=yaw, pitch=pitch, emit=emit, run_id=run_id)
        meta_path = run_dir / "run.json"
        meta = json.loads(meta_path.read_text())
        meta.setdefault("drawings", [])
        meta["drawings"].append(record)
        meta_path.write_text(json.dumps(meta, indent=1))
        emit("done", run=run_id)
    except Exception as e:
        emit("error", text=_error_text(e))
    finally:
        hub.end()


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/styles")
def styles():
    """Label, art-historical note and the exact prompt for each drawing style (the UI shows all three)."""
    return {k: {"label": v["label"], "description": v["description"], "why": v["why"], "prompt": v["prompt"]}
            for k, v in stylize.STYLES.items()}


@app.get("/api/anatomy")
def anatomy():
    """3D soma positions of every neuron, for the rotatable brain.

    Binary, little-endian: N x 3 uint16 (x, y, z scaled to 0..65535), then N uint8 class codes
    (see the X-Classes header). Header X-Neurons gives N.
    """
    xyz = (exp.anatomy.xyz * 65535).astype("<u2")
    sc = exp.anatomy.labels.super_class.fillna("").to_numpy()
    classes = sorted({c for c in sc if c})
    code = np.array([classes.index(c) + 1 if c else 0 for c in sc], np.uint8)
    return Response(xyz.tobytes() + code.tobytes(), media_type="application/octet-stream",
                    headers={"X-Neurons": str(len(code)), "X-Classes": ",".join(classes)})


@app.get("/api/channels")
def channels():
    """Every input channel with the brain indices of its sensory neurons, plus the sugar and MN9 cells.

    Lets the page name each condition before the screen finishes and mark in 3D where each input enters
    the brain and where the feeding readout sits.
    """
    return {"sugar": [int(i) for i in exp.sugar], "mn9": [int(i) for i in exp.mn9],
            "channels": [{**{k: v for k, v in c.items() if k != "neurons"}, "n": len(c["neurons"]),
                          "neurons": [int(i) for i in c["neurons"]]} for c in exp.channels]}


@app.get("/api/models")
def list_models():
    """Every checkpoint ComfyUI has loaded, with its detected family and whether it's qualified
    to draw with (an installed union ControlNet supports its family, and
    scripts/qualify_models.py measured its fidelity close enough to the default's). The picker
    should only offer qualified ones."""
    try:
        return models.list_models(results_path=ROOT / "out" / "models.json")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"couldn't reach ComfyUI: {e}")


@app.get("/api/history")
def history():
    runs = []
    for d in sorted(RUNS.iterdir(), reverse=True):
        meta = d / "run.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            has_portrait = (d / "portrait.json").exists()
            runs.append({"id": d.name, "style": m.get("style"), "blend": [b["name"] for b in m["result"]["blend"]],
                         "ids": [b["id"] for b in m["result"]["blend"]],
                         "gain": m["result"]["confirm"]["gain"], "thumb": f"/runs/{d.name}/strict.png",
                         # lets the page skip fetching a portrait.json that doesn't exist, and offer Redraw only
                         # for runs saved with the full portrait (those also have a drawings/ folder)
                         "portrait": has_portrait, "redrawable": has_portrait and (d / "drawings").is_dir()})
    return runs[:30]


@app.get("/api/run/{run_id}")
def get_run(run_id: str):
    meta = RUNS / run_id / "run.json"
    if not RUN_ID.fullmatch(run_id) or not meta.exists():
        raise HTTPException(404)
    return json.loads(meta.read_text())


@app.get("/api/status")
def status():
    return {"running": hub.running, "run": hub.run_id, **hub.params}


@app.post("/api/start")
def start(request: Request, style: str = "cajal", trials: int = Query(16, ge=2, le=64),
          strength: float = Query(1.0, ge=0.0, le=2.0), model: str | None = None, palette: str = "ink",
          yaw: float = Query(0.0, ge=-180, le=180), pitch: float = Query(0.0, ge=-85, le=85)):
    """Start an experiment. If one is already running, report it instead (attach with /api/stream)."""
    if cross_site(request):
        raise HTTPException(403, "cross-site request rejected")
    if style not in stylize.STYLES:
        raise HTTPException(400, f"unknown style {style}")
    palette = _validate_palette(palette)
    model = _validate_model(model)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    if not hub.begin(run_id, {"style": style, "trials": trials, "model": model, "palette": palette,
                              "yaw": yaw, "pitch": pitch, "started": time.time()}):
        return {"started": False, "running": True, "run": hub.run_id}
    threading.Thread(target=work, args=(run_id, style, trials, strength, model, palette, yaw, pitch),
                     daemon=True).start()
    return {"started": True, "running": True, "run": run_id}


@app.post("/api/draw")
def draw(request: Request, run: str = Query(...), style: str = "cajal", model: str | None = None,
        palette: str = "ink", yaw: float = Query(0.0, ge=-180, le=180), pitch: float = Query(0.0, ge=-85, le=85)):
    """Redraw an existing saved run (any style/model/palette/view) without re-running the
    4-minute simulation. Uses the same hub as /api/start: one job at a time, the same SSE stream
    (stage/progress/preview/drawing/done events), so the page's existing draw UI can attach the
    same way. The new drawing is added under out/runs/<run>/drawings/<n>/ and to run.json's
    "drawings" list (GET /api/run/{run} then shows it)."""
    if cross_site(request):
        raise HTTPException(403, "cross-site request rejected")
    if not RUN_ID.fullmatch(run):
        raise HTTPException(404, "no such run")
    run_dir = RUNS / run
    meta_path, portrait_path = run_dir / "run.json", run_dir / "portrait.json"
    if not meta_path.exists():
        raise HTTPException(404, "no such run")
    if not portrait_path.exists():
        raise HTTPException(404, "this run has no portrait.json and can't be redrawn")
    saved = json.loads(portrait_path.read_text())
    try:
        p = portrait.from_saved(saved)
    except KeyError as e:
        raise HTTPException(404, f"this run can't be redrawn: {e}")
    if style not in stylize.STYLES:
        raise HTTPException(400, f"unknown style {style}")
    palette = _validate_palette(palette)
    model = _validate_model(model)
    meta = json.loads(meta_path.read_text())
    index = len(meta.get("drawings") or [])
    if not hub.begin(run, {"style": style, "model": model, "palette": palette, "yaw": yaw, "pitch": pitch,
                           "redraw": True, "index": index, "started": time.time()}):
        return {"started": False, "running": True, "run": hub.run_id}
    strength = meta.get("strength", 1.0)
    threading.Thread(target=redraw_work, args=(run_dir, run, index, p, style, model, palette, yaw, pitch, strength),
                     daemon=True).start()
    return {"started": True, "running": True, "run": run, "index": index}


@app.get("/api/stream")
def stream():
    """Server-sent events for the current (or most recent) experiment: everything so far, then live."""
    q = hub.attach()

    def events():
        try:
            while (msg := q.get()) is not None:
                yield f"data: {json.dumps(msg)}\n\n"
        finally:
            hub.detach(q)

    return StreamingResponse(events(), media_type="text/event-stream")
