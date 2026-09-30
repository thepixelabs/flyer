"""Qualify installed SDXL checkpoints against our ControlNet.

Renders a fixed saved run's control and depth maps with each candidate checkpoint (Cajal and
cyanotype, one fixed seed), and records fidelity to out/models.json (local, git-ignored; app.py's
GET /api/models reads it). A checkpoint is "qualified" when its on_data/coverage stay within
MARGIN of the default checkpoint's own (also measured here, from the same run/seed/styles, so the
threshold moves with whatever run this was last qualified against) -- generous enough for
legitimate style variation between checkpoints, but comfortably above the "chance" level fidelity()
reports for a drawing that ignores the ControlNet entirely.

Usage: uv run python scripts/qualify_models.py [run_id] [checkpoint.safetensors ...]
  run_id defaults to the most recent saved run with a full portrait.json (base_hz/z; see
  flybrain/portrait.py's from_saved). checkpoint args (repeatable) default to every SDXL
  "candidate" checkpoint ComfyUI reports (an installed union ControlNet supports sdxl).

Needs ComfyUI running on :8188. Renders 2 images per checkpoint (one per style); for N candidate
checkpoints that's 2N renders through the sampler, so this takes a while -- that's fine, it's
meant to be run once per checkpoint set, not on every draw.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flybrain import models, portrait, stylize  # noqa: E402
from flybrain.brain import FlyBrain  # noqa: E402
from flybrain.experiment import Anatomy  # noqa: E402

SEED = 4242
STYLES = ["cajal", "cyanotype"]
OUT = ROOT / "out" / "models.json"
MARGIN = 0.75  # a candidate may fall to 75% of the default's on_data/coverage and still qualify


def _find_run(run_id: str | None) -> Path:
    runs = sorted(p for p in (ROOT / "out" / "runs").iterdir() if (p / "portrait.json").exists())
    if run_id:
        p = ROOT / "out" / "runs" / run_id
        if not (p / "portrait.json").exists():
            raise SystemExit(f"no such run (or no portrait.json): {run_id}")
        return p
    for p in reversed(runs):
        saved = json.loads((p / "portrait.json").read_text())
        if saved.get("neurons") and len(saved["neurons"][0]) >= 7:  # full (not reduced) save
            return p
    raise SystemExit("no saved run has a full portrait.json (base_hz/z) to qualify against -- "
                     "run one experiment first (POST /api/start)")


def main():
    args = sys.argv[1:]
    names = [a for a in args if a.endswith(".safetensors")]
    run_args = [a for a in args if not a.endswith(".safetensors")]
    run_dir = _find_run(run_args[0] if run_args else None)
    saved = json.loads((run_dir / "portrait.json").read_text())
    p = portrait.from_saved(saved)

    anatomy = Anatomy.load(FlyBrain(device="cpu"))
    ctrl = portrait.control(p, anatomy.xyz)
    depth_img = portrait.depth(anatomy.xyz)
    silhouette = portrait.silhouette_mask(anatomy.xyz)

    all_ckpts = models.checkpoints()
    candidates = [c for c in all_ckpts if c["candidate"]]
    if names:
        candidates = [c for c in candidates if c["name"] in names]
    if not candidates:
        raise SystemExit("no candidate SDXL checkpoints (with a matching installed union "
                         "ControlNet) found among " + (str(names) if names else "the installed checkpoints"))
    DEFAULT = models.default_checkpoint()
    if not any(c["name"] == DEFAULT for c in candidates):
        print(f"note: default checkpoint {DEFAULT!r} isn't in the candidate set being "
             f"qualified -- adding it, so there's a real baseline to measure everyone else against")
        candidates = [c for c in all_ckpts if c["name"] == DEFAULT] + candidates

    print(f"qualifying against {run_dir.name} ({len(p['neurons'])} neurons), seed {SEED}, styles {STYLES}")
    results = {}
    for c in candidates:
        quirk = models.prompt_quirk(c["name"]) or {}
        per_style = {}
        for style in STYLES:
            t0 = time.time()
            img = stylize.stylize(ctrl, depth_img, style=style, checkpoint=c["name"], seed=SEED,
                                  prompt_prefix=quirk.get("positive_prefix", ""),
                                  negative_prefix=quirk.get("negative_prefix", ""))
            fid = stylize.fidelity(ctrl, img, depth_img=depth_img, silhouette=silhouette)
            per_style[style] = fid
            anat = fid.get("anatomy", {}).get("corr")
            print(f"  {c['name']:35s} {style:10s} on_data={fid['on_data']:.3f} "
                 f"coverage={fid['coverage']:.3f} anatomy={anat}  ({time.time() - t0:.0f}s)")
        avg = {k: round(sum(per_style[s][k] for s in STYLES) / len(STYLES), 3) for k in ("on_data", "coverage")}
        results[c["name"]] = {"family": c["family"], "per_style": per_style, "fidelity": avg}

    default_fid = results[DEFAULT]["fidelity"]
    for name, r in results.items():
        ok = (r["fidelity"]["on_data"] >= default_fid["on_data"] * MARGIN and
             r["fidelity"]["coverage"] >= default_fid["coverage"] * MARGIN)
        r["qualified"] = ok or name == DEFAULT
        r["reason"] = ("within margin of the default checkpoint's fidelity" if ok
                       else f"fidelity too far below the default {default_fid} (margin {MARGIN}) "
                            "to trust its ControlNet adherence")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"reference_run": run_dir.name, "seed": SEED, "styles": STYLES,
                               "default": DEFAULT, "default_fidelity": default_fid, "margin": MARGIN,
                               "models": results, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    n_ok = sum(r["qualified"] for r in results.values())
    print(f"wrote {OUT}: {n_ok}/{len(results)} qualified")


if __name__ == "__main__":
    main()
