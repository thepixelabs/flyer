"""Which SDXL checkpoints work well with our ControlNet, and how to find them.

flybrain draws through a union ControlNet (xinsir's SDXL ProMax), which only ever saw SDXL
training, so a checkpoint has to be (a) an SDXL model and (b) actually behave well under that
ControlNet. This module does (a) -- family detection, from the checkpoint's own safetensors
header when the model folder can be found, else a filename heuristic -- plus cross-platform
discovery of that folder. (b) is scripts/qualify_models.py, which renders a fixed saved run with
each candidate checkpoint and records fidelity to out/models.json (local, git-ignored); this
module merges those results in for GET /api/models.
"""
from __future__ import annotations

import json
import os
import platform
import struct
from pathlib import Path

import httpx

COMFY = "http://localhost:8188"
# The default checkpoint and union ControlNet, under either the filename Hugging Face gives the download
# (what `hf download` saves) or the name earlier versions of this project asked you to rename it to.
# Whichever one is installed in ComfyUI is used; the first entry is the fallback if ComfyUI can't be asked.
DEFAULT_CHECKPOINT_FILES = ("Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", "JuggernautXL_v9.safetensors")
DEFAULT_CONTROLNET_FILES = ("diffusion_pytorch_model_promax.safetensors", "controlnet-union-sdxl-promax.safetensors")
CHECKPOINT = DEFAULT_CHECKPOINT_FILES[0]   # default checkpoint (always selectable)
CONTROLNET = DEFAULT_CONTROLNET_FILES[0]   # default union ControlNet

# Checkpoint filename substrings (case-insensitive) needing their own prompt conventions on top of
# a style's prompt. Pony and Illustrious are SDXL architecturally (same safetensors header
# signature as Juggernaut/RealVis/epiCRealism), so family detection alone can't tell them apart --
# this is a name-based overlay, applied regardless of how family was detected. Best-effort: these
# are the conventions their own model cards document, not validated against every style here.
PROMPT_QUIRKS = {
    "pony": {"positive_prefix": "score_9, score_8_up, score_7_up, score_6_up, ",
             "negative_prefix": "score_6, score_5, score_4, source_pony, "},
    "illustrious": {"positive_prefix": "masterpiece, best quality, very aesthetic, absurdres, ",
                    "negative_prefix": "lowres, worst quality, bad anatomy, "},
}

# --- family detection by safetensors header ---------------------------------------------------
# Each family's signature is a key prefix that (among these four) only it has; checked in this
# order so a file matching more than one (shouldn't happen) still classifies once, flux/sd3 first
# since they're structurally the most distinctive.
_FAMILY_SIGNATURES = [
    ("flux", "double_blocks."),
    ("sd3", "joint_blocks."),
    ("sdxl", "conditioner.embedders.1."),   # SDXL's second (dual) text encoder; SD1.5 has only one
    ("sd15", "cond_stage_model.transformer."),
]

_NAME_HINTS = [
    ("flux", ("flux",)),
    ("sd3", ("sd3", "sd_3", "stablediffusion3")),
    ("sdxl", ("xl", "sdxl")),
    ("sd15", ("sd15", "sd1.5", "sd-1.5", "v1-5", "v1_5")),
]


def _safetensors_header(path: Path) -> dict:
    """The JSON header of a .safetensors file (tensor names -> dtype/shape/offsets), without
    loading any tensor data: the format is an 8-byte little-endian header length, then that many
    bytes of JSON, then the raw tensor bytes."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _family_from_header(path: Path) -> str | None:
    try:
        keys = _safetensors_header(path).keys()
    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        return None
    for family, prefix in _FAMILY_SIGNATURES:
        # substring, not startswith: some checkpoints (e.g. SD3.5) nest these under a
        # "model.diffusion_model." prefix rather than at the top level
        if any(prefix in k for k in keys):
            return family
    return None


def _family_from_name(name: str) -> str | None:
    low = name.lower()
    for family, hints in _NAME_HINTS:
        if any(h in low for h in hints):
            return family
    return None


# --- ComfyUI's model folder, cross-platform -----------------------------------------------------

def comfyui_dir() -> Path | None:
    """Where ComfyUI's model files live, so family detection can read their safetensors headers.

    Order: $COMFYUI_DIR, then common install locations (~/ComfyUI, ~/Documents/ComfyUI,
    ~/Documents/git/ComfyUI, the Windows/macOS/Linux ComfyUI Desktop app data directories), first
    one with a models/checkpoints subfolder wins. Returns None if none is found, and callers fall
    back to the filename heuristic (pathlib only, so this works unmodified on Windows/macOS/Linux).
    """
    env = os.environ.get("COMFYUI_DIR")
    candidates = [Path(env)] if env else []
    home = Path.home()
    candidates += [home / "ComfyUI", home / "Documents" / "ComfyUI", home / "Documents" / "git" / "ComfyUI"]
    system = platform.system()
    if system == "Windows":
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            candidates.append(Path(userprofile) / "Documents" / "ComfyUI")
        appdata = os.environ.get("APPDATA")
        if appdata:  # ComfyUI Desktop's default data directory on Windows
            candidates.append(Path(appdata) / "ComfyUI")
    elif system == "Darwin":
        candidates.append(home / "Library" / "Application Support" / "ComfyUI")  # ComfyUI Desktop
    else:
        candidates.append(home / ".config" / "ComfyUI")  # ComfyUI Desktop on Linux
    for c in candidates:
        if (c / "models" / "checkpoints").is_dir():
            return c
    return None


def _find_model(dir_: Path | None, subfolder: str, name: str) -> Path | None:
    if dir_ is None:
        return None
    p = dir_ / "models" / subfolder / name  # ComfyUI reports subfolder models with forward slashes
    return p if p.is_file() else None


# --- ComfyUI's installed checkpoints and ControlNets --------------------------------------------

def _object_info(field: str, comfy: str) -> list[str]:
    r = httpx.get(f"{comfy}/object_info/{field}", timeout=10)
    r.raise_for_status()
    key = "ckpt_name" if field == "CheckpointLoaderSimple" else "control_net_name"
    return r.json()[field]["input"]["required"][key][0]


def _default_of(candidates: tuple[str, ...], field: str, comfy: str) -> str:
    try:
        installed = _object_info(field, comfy)
    except (httpx.HTTPError, KeyError, ValueError):
        return candidates[0]
    return next((n for n in candidates if n in installed), candidates[0])


def default_checkpoint(comfy: str = COMFY) -> str:
    """The default checkpoint's filename as ComfyUI has it (either accepted name)."""
    return _default_of(DEFAULT_CHECKPOINT_FILES, "CheckpointLoaderSimple", comfy)


def default_controlnet(comfy: str = COMFY) -> str:
    """The default union ControlNet's filename as ComfyUI has it (either accepted name)."""
    return _default_of(DEFAULT_CONTROLNET_FILES, "ControlNetLoader", comfy)


class SetupError(RuntimeError):
    """Drawing can't start because ComfyUI or one of its model files isn't set up. The message is
    written for the person using the app, so callers show it as is."""


_HOW = "See \"Set up drawing\" in the README (github.com/thepixelabs/flyer#3-set-up-drawing-once)."


def setup_problem(checkpoint: str | None = None, comfy: str = COMFY) -> str | None:
    """A plain-language message if drawing can't work yet, else None. Checks that ComfyUI answers, that
    the checkpoint (the default unless `checkpoint` names another) and the union ControlNet are
    installed where ComfyUI looks, and says which folder a missing file belongs in."""
    try:
        ckpts = _object_info("CheckpointLoaderSimple", comfy)
        nets = _object_info("ControlNetLoader", comfy)
    except (httpx.HTTPError, KeyError, ValueError):
        return (f"Drawing needs ComfyUI, and Flyer can't reach it at {comfy}. Start ComfyUI and try again. "
                f"The search and the 3D view work without it. {_HOW}")
    if checkpoint and not is_default_checkpoint(checkpoint):
        if checkpoint not in ckpts:
            return (f"ComfyUI doesn't have the model {checkpoint!r}. Put it in ComfyUI/models/checkpoints "
                    f"and restart ComfyUI.")
    elif not any(n in ckpts for n in DEFAULT_CHECKPOINT_FILES):
        return (f"ComfyUI doesn't have the default image model. Download {DEFAULT_CHECKPOINT_FILES[0]} into "
                f"ComfyUI/models/checkpoints and restart ComfyUI. {_HOW}")
    if not any(n in nets for n in DEFAULT_CONTROLNET_FILES):
        return (f"ComfyUI doesn't have the union ControlNet that makes the drawing follow the data. Download "
                f"{DEFAULT_CONTROLNET_FILES[0]} into ComfyUI/models/controlnet and restart ComfyUI. {_HOW}")
    return None


def is_default_checkpoint(name: str) -> bool:
    return name in DEFAULT_CHECKPOINT_FILES


def _is_union_sdxl_controlnet(name: str, path: Path | None) -> bool:
    """xinsir's SDXL union ControlNet (ProMax or compatible): its union/task-conditioning layers
    (control_add_embedding, task_embedding) have no equivalent in a single-task ControlNet, so
    their presence is a reliable header signature; falls back to the filename ("union"/"promax")
    when the file can't be read."""
    if path is not None:
        try:
            keys = _safetensors_header(path).keys()
            return any(k.startswith("task_embedding") or k.startswith("control_add_embedding") for k in keys)
        except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
            pass
    low = name.lower()
    return "union" in low or "promax" in low


def checkpoints(comfy: str = COMFY) -> list[dict]:
    """Every checkpoint ComfyUI has loaded: {name, family, family_source, candidate}. `candidate`
    is True when an installed ControlNet supports that family (today: only sdxl, via a union
    ControlNet detected the same way)."""
    ckpt_dir = comfyui_dir()
    names = _object_info("CheckpointLoaderSimple", comfy)
    controlnets = _object_info("ControlNetLoader", comfy)
    supported = {"sdxl"} if any(_is_union_sdxl_controlnet(cn, _find_model(ckpt_dir, "controlnet", cn))
                                for cn in controlnets) else set()
    out = []
    for name in names:
        path = _find_model(ckpt_dir, "checkpoints", name)
        family = _family_from_header(path) if path else None
        source = "header"
        if family is None:
            family = _family_from_name(name)
            source = "name" if family else None
        out.append({"name": name, "family": family, "family_source": source, "candidate": family in supported})
    return out


def list_models(comfy: str = COMFY, results_path: Path | None = None) -> list[dict]:
    """{name, family, family_source, qualified, fidelity, reason} for every checkpoint ComfyUI
    reports, merging scripts/qualify_models.py's results (out/models.json) when present. Only
    "candidate" checkpoints (a matching union ControlNet is installed) can ever be qualified."""
    results, saved = {}, None
    if results_path and results_path.exists():
        saved = json.loads(results_path.read_text())
        results = saved.get("models", {})
    out = []
    for c in checkpoints(comfy):
        r = results.get(c["name"])
        if is_default_checkpoint(c["name"]) and r is None:
            out.append({**c, "qualified": True, "fidelity": None, "reason": "the default checkpoint"})
        elif r is not None:
            out.append({**c, "qualified": bool(r.get("qualified")), "fidelity": r.get("fidelity"),
                       "reason": r.get("reason", "")})
        elif not c["candidate"]:
            fam = c["family"] or "this checkpoint's family"
            out.append({**c, "qualified": False, "fidelity": None,
                       "reason": f"no installed union ControlNet supports {fam}"})
        else:
            out.append({**c, "qualified": False, "fidelity": None,
                       "reason": "not yet qualified -- run scripts/qualify_models.py"})
    return out


def is_selectable(name: str, comfy: str = COMFY, results_path: Path | None = None) -> bool:
    """True if `name` is the default checkpoint or scripts/qualify_models.py marked it qualified."""
    if is_default_checkpoint(name):
        return True
    return any(m["name"] == name and m["qualified"] for m in list_models(comfy, results_path))


def prompt_quirk(checkpoint: str) -> dict | None:
    """{"positive_prefix", "negative_prefix"} for checkpoints needing their own prompt
    conventions (see PROMPT_QUIRKS), or None."""
    low = checkpoint.lower()
    for hint, quirk in PROMPT_QUIRKS.items():
        if hint in low:
            return quirk
    return None
