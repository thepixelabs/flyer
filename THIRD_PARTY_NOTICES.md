# Third-party notices

Repository: https://github.com/thepixelabs/flyer
Site: https://flyer.pixelabs.net/

This file lists everything in this repository, or published on its site, that comes from someone
else, the terms it comes under, and what we changed. It was checked on 2026-09-26.

## What covers what

| Part of the repository | Terms |
|---|---|
| Source code (`flybrain/`, `scripts/`, `app.py`, `web/`, `docs/css/`, HTML) | MIT, Copyright (c) 2026 PixeLabs, see [`LICENSE`](LICENSE). Includes a reimplementation of the Shiu et al. model, see section 1. |
| Per-channel results in `docs/screen.csv` | Simulation results computed on FlyWire data, adapted under CC BY 4.0, see sections 2 to 4. Our adaptation is also offered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). |
| Images in `docs/img/` | Drawn from our simulation of the FlyWire data; the stylised ones were rendered locally with the image models in section 6. Offered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), to the extent PixeLabs holds any rights in them. The FlyWire attribution in section 2 applies to them too. |
| Fonts in `web/fonts/` and three.js in `web/vendor/three/` | Vendored copies used by the local app, see sections 5 and 7. |
| Downloaded at run time, not in this repository | FlyWire connectivity and Shiu et al. files, FlyWire annotations (fetched by `scripts/prepare.py` into the git-ignored `data/`); Python packages (installed by `uv`); image-model weights (installed by you into ComfyUI). Each comes under its own terms. |

## 1. Drosophila brain model (Shiu et al.)

- **Used for:** `flybrain/brain.py` reimplements the leaky integrate-and-fire model of Shiu et al. in PyTorch:
  the same equations and parameter values (the `Params` defaults). The code text was written anew and is not
  copied from their repository. `scripts/prepare.py` downloads their connectivity files (`Connectivity_783.parquet`,
  `Completeness_783.csv`) and reads the taste-neuron ID lists from their `figures.ipynb` at run time; none of these
  files is committed here.
- **Source:** https://github.com/philshiu/Drosophila_brain_model, commit
  `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` (the files we used match it byte for byte).
- **Paper:** Shiu, P. K. et al. A Drosophila computational brain model reveals sensorimotor processing.
  *Nature* 634, 210 to 219 (2024). https://doi.org/10.1038/s41586-024-07763-9
- **Changes:** see "Statement of changes" below (neurotransmitter sign correction, GPU event-driven implementation).
- **Licence:** MIT. Full text, as published in their repository:

```
MIT License

Copyright (c) 2023 Philip Shiu and Nico Spiller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. FlyWire connectome, materialization 783

- **Creator:** the FlyWire Consortium. Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain.
  *Nature* 634, 124 to 138 (2024). https://doi.org/10.1038/s41586-024-07558-y
- **Source:** FlyWire Whole-brain Connectome Connectivity Data, version 783.0, https://doi.org/10.5281/zenodo.10676866,
  also served through Codex (https://codex.flywire.ai). We use the connectivity as processed by Shiu et al.
  (section 1), which gives synapse counts between neurons with a sign from each neuron's predicted neurotransmitter.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/ (the licence recorded on the Zenodo release).
- **Disclaimer:** the licensed material is provided as is, without warranties of any kind, as set out in section 5
  of CC BY 4.0. Nothing here suggests that the FlyWire Consortium or any author endorses this project.
- **Redistributed here, in changed form:** simulation results computed on the connectome (`docs/screen.csv`) and
  pictures drawn from them (`docs/img/`). No FlyWire root IDs or neuron-level data are included. See "Statement of changes".

## 3. FlyWire annotations (flyconnectome/flywire_annotations)

- **Used for:** soma positions (`soma_x/y/z`, falling back to `pos_x/y/z`), `super_class`, `cell_class`,
  `top_nt` and `known_nt` for every neuron, from `supplemental_files/Supplemental_file1_neuron_annotations.tsv`.
- **Source:** https://github.com/flyconnectome/flywire_annotations, release
  [v3.1.0](https://github.com/flyconnectome/flywire_annotations/releases/tag/v3.1.0) (the file we used matches the
  file at that tag byte for byte; `scripts/prepare.py` downloads from the `main` branch).
- **Terms:** the repository has no LICENSE file. Its README asks users of version 3.0.0 and later to
  "cite Berg et al. (2025), Schlegel et al. (2024), Matsliah et al. (2024) and Dorkenwald et al. (2024) when using
  annotations from this repository". The annotations describe the FlyWire 783 release, which is published under
  CC BY 4.0, and the four papers are published under CC BY 4.0. We follow the citation request and give the same
  attribution and statement of changes as for section 2. Citations:
  - Berg, S. et al. Sexual dimorphism in the complete *Drosophila* male central nervous system connectome.
    *Cell* 189, 5504 to 5526.e15 (2026). https://doi.org/10.1016/j.cell.2026.08.015
    (preprint: *bioRxiv* (2025), https://doi.org/10.1101/2025.10.09.680999, the version the README names)
  - Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing of *Drosophila*.
    *Nature* 634, 139 to 152 (2024). https://doi.org/10.1038/s41586-024-07686-5
  - Matsliah, A. et al. Neuronal parts list and wiring diagram for a visual system.
    *Nature* 634, 166 to 180 (2024). https://doi.org/10.1038/s41586-024-07981-1
  - Dorkenwald, S. et al., as in section 2.

## Statement of changes (for the CC BY 4.0 material in sections 2 and 3)

Earlier change, kept from Shiu et al.: the FlyWire connectivity was reduced to synapse counts between neurons, each
signed by the presynaptic neuron's predicted neurotransmitter.

Our changes:

1. **Neurotransmitter sign correction.** Antennal-lobe local neurons (`cell_class` ALLN) are made inhibitory unless
   they are known to be cholinergic, and neurons predicted to release dopamine, serotonin or octopamine are given no
   fast synaptic effect (`neuron_sign_fix` in `flybrain/brain.py`). `FlyBrain(nt_fix=False)` restores the original
   signs.
2. **Reimplementation.** The model runs as an event-driven simulation on a GPU in PyTorch rather than in Brian2,
   with the same equations and constants.
3. **Units.** Positions, given in FlyWire voxels of 4 × 4 × 40 nm, are converted to nanometres on all three axes,
   then centred, scaled into a unit box and rounded.
4. **Reduction.** Only pictures and per-channel results are published. FlyWire root IDs and neuron-level data are not.
5. **New results.** Simulated spike times and firing rates from our model, and the results of our search experiment
   (screen, re-test, greedy combination and confirmation of taste and smell channels) were added.
6. **Drawings.** The images in `docs/img/` are drawn from these data by our code; the drawings were then restyled
   by an image model (section 6).

## 4. Where to cite from

If you use `docs/screen.csv` or the images, credit the FlyWire Consortium, cite the five papers above, link
the CC BY 4.0 licence, and keep this statement of changes.

## 5. Fonts

The landing page uses system fonts and ships none.

### Fonts used by the local app (`web/fonts/`)

Vendored from Google Fonts (previously loaded live from `fonts.googleapis.com`/`fonts.gstatic.com`) so the app works
offline and under a strict CSP. Each file is shipped next to its full OFL licence text.

| Font | Google Fonts version | Copyright | Licence text |
|---|---|---|---|
| Fraunces (variable, opsz/wght) | v38 | Copyright 2018 The Fraunces Project Authors (https://github.com/undercasetype/Fraunces) | [`OFL-Fraunces.txt`](web/fonts/OFL-Fraunces.txt) |
| Inter (variable, wght) | v20 | Copyright 2020 The Inter Project Authors (https://github.com/rsms/inter) | [`OFL-Inter.txt`](web/fonts/OFL-Inter.txt) |
| JetBrains Mono (variable, wght) | v24 | Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono) | [`OFL-JetBrainsMono.txt`](web/fonts/OFL-JetBrainsMono.txt) |

Licence: SIL Open Font License, Version 1.1, https://openfontlicense.org.

## 6. Image models used for the published drawings

The stylised drawings (`docs/img/free*`, `docs/img/strict*`) were rendered on our own machine with ComfyUI. **No model
weights are included in this repository or on the site** (`*.safetensors` is ignored by git). The licences below
apply to the weights, which you install yourself if you run the drawing step.

- **Juggernaut XL v9 + RunDiffusion Photo v2**, by RunDiffusion and KandooAI, a fine-tune of Stable Diffusion XL.
  File used: `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` (saved locally as `JuggernautXL_v9.safetensors`),
  SHA-256 `c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca`.
  Source: https://huggingface.co/RunDiffusion/Juggernaut-XL-v9 and https://civitai.com/models/133005/juggernaut-xl.
  Licence: CreativeML Open RAIL-M as stated by its authors; the model card also says it "may not be deployed behind
  paid API services" without a separate licence, and the Civitai listing requires credit to the creator and allows
  commercial use of generated images.
- **Stable Diffusion XL 1.0**, by Stability AI, the base model of Juggernaut XL. Licence: CreativeML Open RAIL++-M
  (July 26, 2023), https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md. Its
  paragraph 6 reads: "Licensor claims no rights in the Output You generate using the Model. You are accountable for
  the Output you generate and its subsequent uses." Both RAIL licences carry use restrictions (Attachment A) that
  bind anyone who runs the models; the drawings here are abstract renderings of neuron data and fall under none of
  them.
- **ControlNet Union SDXL 1.0, ProMax**, by xinsir. File used: `diffusion_pytorch_model_promax.safetensors` (saved
  locally as `controlnet-union-sdxl-promax.safetensors`), SHA-256
  `9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc`.
  Source: https://huggingface.co/xinsir/controlnet-union-sdxl-1.0. Licence: Apache License 2.0.

## 7. Vendored or installed at run time

- **three.js** 0.180.0 (MIT, Copyright 2010 to 2025 three.js authors), including the `OrbitControls` addon.
  Downloaded from cdn.jsdelivr.net (the npm package's `build/` and `examples/jsm/` files) and vendored under
  `web/vendor/three/` so the local app works offline and under a strict CSP; its upstream `LICENSE` is kept
  alongside at `web/vendor/three/LICENSE`. The published site does not use it.
- **Python packages** listed in `pyproject.toml` and `uv.lock` (PyTorch, NumPy, pandas, SciPy, PyArrow, Pillow,
  FastAPI, Uvicorn, HTTPX, websocket-client), installed by `uv` under their own licences.
- **ComfyUI** (GPL-3.0), run as a separate program over its HTTP API; no ComfyUI code is included here.
