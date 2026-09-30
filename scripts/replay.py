"""Play a recorded experiment as if it were running, to develop the page without using the GPU.

    uv run python scripts/replay.py out/stream1.txt --port 8433 --source http://localhost:8432

The recording is the raw text of /api/stream (lines of `data: {json}`). This server:
  * serves web/ and out/runs/ from disk, so page edits show up on reload;
  * plays the recording through /api/status, /api/start and /api/stream with the same attach
    rules as app.py (a page that attaches late gets everything so far, then live), so refreshing
    mid-run can be tested;
  * forwards /api/anatomy and /api/channels to --source (read-only GETs, cached in memory).
The recording has no timestamps, so events are paced by kind; --speed scales the pacing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flybrain.stylize import STYLES  # noqa: E402  (no GPU work on import)

RUNS = ROOT / "out" / "runs"
TRANSIENT = {"live", "preview", "progress"}
# seconds before each event, by kind (a real run is about 1.5 s per live frame; this is faster)
PACE = {"live": 0.4, "preview": 0.45, "stage": 1.2, "live_init": 0.8, "progress": 0.0}


def load(path: Path, speed: float) -> list[tuple[float, dict]]:
    schedule, t = [], 0.0
    for line in path.read_text().splitlines():
        if not line.startswith("data: "):
            continue
        msg = json.loads(line[6:])
        if msg["kind"] == "attached":
            continue
        t += PACE.get(msg["kind"], 0.6) / speed
        schedule.append((t, msg))
    return schedule


def make_app(recording: Path, source: str, speed: float) -> FastAPI:
    app = FastAPI(title="Flyer (replay)")
    app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
    app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")
    state = {"t0": time.time(), "schedule": load(recording, speed)}
    cache: dict[str, Response] = {}

    def elapsed() -> float:
        return time.time() - state["t0"]

    def running() -> bool:
        return bool(state["schedule"]) and elapsed() < state["schedule"][-1][0]

    def run_id() -> str | None:
        for _, m in state["schedule"]:
            if m.get("run"):
                return m["run"]
        return "replay"

    def params() -> dict:
        return {"style": "cajal", "trials": 16, "started": state["t0"]}

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/api/status")
    def status():
        return {"running": running(), "run": run_id(), **params()}

    @app.post("/api/start")
    def start(style: str = "cajal", trials: int = 16):
        if running():
            return {"started": False, "running": True, "run": run_id()}
        state["t0"], state["schedule"] = time.time(), load(recording, speed)
        return {"started": True, "running": True, "run": run_id()}

    @app.get("/api/stream")
    async def stream():
        async def events():
            now, sched = elapsed(), state["schedule"]
            yield f"data: {json.dumps({'kind': 'attached', 'run': run_id(), 'running': running(), **params()})}\n\n"
            past = [m for t, m in sched if t <= now]
            latest = {}
            for m in past:
                if m["kind"] in TRANSIENT:
                    latest[m["kind"]] = m
                else:
                    yield f"data: {json.dumps(m)}\n\n"
            for m in latest.values():
                yield f"data: {json.dumps(m)}\n\n"
            for t, m in sched[len(past):]:
                await asyncio.sleep(max(0.0, state["t0"] + t - time.time()))
                yield f"data: {json.dumps(m)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/styles")
    def styles():
        return {k: {"label": v["label"], "description": v["description"], "why": v["why"], "prompt": v["prompt"]}
                for k, v in STYLES.items()}

    @app.get("/api/history")
    def history():
        out = []
        for d in sorted(RUNS.iterdir(), reverse=True):
            meta = d / "run.json"
            if meta.exists():
                m = json.loads(meta.read_text())
                out.append({"id": d.name, "style": m.get("style"), "blend": [b["name"] for b in m["result"]["blend"]],
                            "ids": [b["id"] for b in m["result"]["blend"]],
                            "gain": m["result"]["confirm"]["gain"], "thumb": f"/runs/{d.name}/strict.png"})
        return out[:30]

    @app.get("/api/run/{rid}")
    def get_run(rid: str):
        meta = RUNS / rid / "run.json"
        if not re.fullmatch(r"\d{8}-\d{6}", rid) or not meta.exists():
            raise HTTPException(404)
        return json.loads(meta.read_text())

    async def forward(path: str) -> Response:
        if path not in cache:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.get(source + path)
            if r.status_code != 200:
                raise HTTPException(r.status_code)
            keep = {k: v for k, v in r.headers.items() if k.lower().startswith("x-")}
            cache[path] = Response(r.content, media_type=r.headers.get("content-type"), headers=keep)
        return cache[path]

    @app.get("/api/anatomy")
    async def anatomy():
        return await forward("/api/anatomy")

    @app.get("/api/channels")
    async def channels():
        return await forward("/api/channels")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--port", type=int, default=8433)
    ap.add_argument("--source", default="http://localhost:8420", help="server to take anatomy and channels from")
    ap.add_argument("--speed", type=float, default=1.0, help="pacing multiplier (2 = twice as fast)")
    a = ap.parse_args()
    uvicorn.run(make_app(a.recording, a.source.rstrip("/"), a.speed), port=a.port, log_level="warning")
