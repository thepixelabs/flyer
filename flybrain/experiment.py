"""Let the connectome choose its favourite flavour.

1. Baseline: the 21 labellar sugar neurons from Shiu et al. at a fixed rate, the
   one input known to drive feeding on its own. Everything else is measured as
   a change on top of it.
2. Screen: add each taste/smell channel alone; measure MN9 (proboscis motor
   neuron) firing, the feeding readout of Shiu et al.
3. Re-test the promising channels with many more trials, then combine by
   greedy forward selection: add whichever remaining channel raises MN9 most
   over the current blend (measured side by side); stop when no addition
   beats twice the noise.
4. Confirm: re-run baseline vs. the final blend with new random seeds.
5. Portrait: the neurons that responded, where they sit, when they first fired,
   and the synapses between them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .brain import DATA, FlyBrain
from .channels import build_channels

SUGAR_HZ = 80   # baseline sugar drive: mid-range, so both boosts and cuts are visible
CHANNEL_HZ = 30  # drive for each added channel
VOXEL_NM = np.array([4.0, 4.0, 40.0])  # FlyWire (FAFB) voxel size: x, y, z
RETEST_TRIALS = 64
COMBINE_TRIALS = 48


@dataclass
class Anatomy:
    labels: pd.DataFrame  # annotation rows aligned to brain indices
    xyz: np.ndarray       # (N, 3) soma (or representative) position, normalised

    @classmethod
    def load(cls, brain: FlyBrain) -> "Anatomy":
        a = pd.read_csv(DATA / "annotations.tsv", sep="\t", low_memory=False).set_index("root_id")
        a = a.reindex(brain.flyids).reset_index()
        p = np.stack([a.soma_x.fillna(a.pos_x), a.soma_y.fillna(a.pos_y), a.soma_z.fillna(a.pos_z)], 1).astype(float)
        p *= VOXEL_NM  # FlyWire positions are in 4 x 4 x 40 nm voxels; convert to nm so depth isn't squashed
        ok = ~np.isnan(p).any(1)
        lo, hi = np.nanpercentile(p[ok], 0.5, 0), np.nanpercentile(p[ok], 99.5, 0)
        span = (hi - lo).max()
        xyz = (p - (lo + hi) / 2) / span + 0.5  # centred, common scale
        xyz[~ok] = 0.5
        return cls(labels=a, xyz=np.clip(xyz, 0, 1).astype(np.float32))


class Experiment:
    def __init__(self, brain: FlyBrain | None = None):
        self.brain = brain or FlyBrain()
        self.anatomy = Anatomy.load(self.brain)
        pops = json.loads((DATA / "populations.json").read_text())
        self.sugar = self.brain.ids_to_idx(pops["sugar"])
        self.mn9 = self.brain.ids_to_idx(pops["MN9"])
        # Sugar is the fixed baseline, so the other sugar sensors are left out of the
        # search ("more sugar" would trivially win); water gets its own channel.
        water = self.brain.ids_to_idx(pops["water"])
        lab = self.anatomy.labels
        sugar_water = np.flatnonzero((lab.cell_class == "gustatory") & (lab.cell_sub_class == "sugar/water"))
        self.channels = build_channels(lab, exclude=set(sugar_water.tolist()) | set(self.sugar))
        self.channels.insert(0, {"id": "taste:water", "kind": "taste", "name": "labellar water sensors",
                                 "receptor": "ppk28", "detects": "water", "neurons": water})
        self.by_id = {c["id"]: c for c in self.channels}

    # --- stimuli -------------------------------------------------------------
    def stim(self, channel_ids=()) -> dict[int, float]:
        s = {i: SUGAR_HZ for i in self.sugar}
        for cid in channel_ids:
            s.update({i: CHANNEL_HZ for i in self.by_id[cid]["neurons"]})
        return s

    def measure(self, blends: list[list[str]], n_run: int, seed: int, on_progress=None, on_live=None, **kw):
        out = self.brain.run_batch([self.stim(b) for b in blends], n_run=n_run, seed=seed,
                                   on_progress=on_progress, watch=self.mn9, on_live=on_live, **kw)
        per_trial = out["trial_rates"].mean(2)  # (C, R): MN9 rate per trial, mean of both sides
        mean = per_trial.mean(1)
        sem = per_trial.std(1, ddof=1) / np.sqrt(per_trial.shape[1])
        return mean, sem, out

    # --- the search ----------------------------------------------------------
    def run(self, n_run: int = 16, seed: int = 0, emit=lambda kind, **d: None) -> dict:
        ids = [c["id"] for c in self.channels]

        def live(stage: str, blends: list[list[str]], n_trials: int):
            """Announce the conditions of a simulation batch, then stream its spikes as it runs."""
            emit("live_init", stage=stage, conditions=blends, trials=n_trials)
            # at most 300 spikes per condition per frame keeps the stream light; it's for display only
            return lambda t_ms, frames, mn9: emit("live", stage=stage, t=t_ms, frames=[f[:300] for f in frames],
                                                  mn9=mn9)

        emit("stage", stage="screen", text=f"Screening {len(ids)} taste & smell channels "
                                           f"on top of sugar ({n_run} trials each)…")
        conds = [[]] + [[c] for c in ids]
        mean, sem, _ = self.measure(conds, n_run, seed, record_spikes=False,
                                    on_progress=lambda f: emit("progress", stage="screen", value=f),
                                    on_live=live("screen", conds, n_run))
        base, base_sem = float(mean[0]), float(sem[0])
        screen = []
        for c, m, e in zip(self.channels, mean[1:], sem[1:]):
            screen.append({**{k: v for k, v in c.items() if k != "neurons"}, "n": len(c["neurons"]),
                           "mn9": float(m), "delta": float(m - base), "sem": float(np.hypot(e, base_sem))})
        screen.sort(key=lambda r: -r["delta"])
        emit("screen", baseline=base, baseline_sem=base_sem, channels=screen)

        # re-test the shortlist (anything that looked helpful) with many more trials
        short = [r["id"] for r in screen if r["delta"] > r["sem"]][:10]
        retest = []
        if short:
            emit("stage", stage="retest", text=f"Re-testing {len(short)} promising channels ({RETEST_TRIALS} trials each)…")
            conds = [[]] + [[c] for c in short]
            m, e, _ = self.measure(conds, RETEST_TRIALS, seed + 50, record_spikes=False,
                                   on_progress=lambda f: emit("progress", stage="retest", value=f),
                                   on_live=live("retest", conds, RETEST_TRIALS))
            retest = [{"id": c, "delta": float(m[i + 1] - m[0]), "sem": float(np.hypot(e[i + 1], e[0]))}
                      for i, c in enumerate(short)]
            emit("retest", channels=retest)

        # greedy forward selection over channels that held up (> 2 SEM) in the re-test.
        # Each round measures the current blend alongside the candidates, so gains are paired.
        helpful = [r["id"] for r in sorted(retest, key=lambda r: -r["delta"]) if r["delta"] > 2 * r["sem"]]
        blend, rounds, rnd = [], [], 0
        while helpful:
            rnd += 1
            emit("stage", stage="combine", text=f"Combination round {rnd}: trying {len(helpful)} additions "
                                                f"({COMBINE_TRIALS} trials each)…")
            conds = [list(blend)] + [blend + [c] for c in helpful]
            m, e, _ = self.measure(conds, COMBINE_TRIALS, seed + 100 * rnd, record_spikes=False,
                                   on_progress=lambda f: emit("progress", stage="combine", value=f),
                                   on_live=live("combine", conds, COMBINE_TRIALS))
            k = int(np.argmax(m[1:]))
            gain, noise = float(m[k + 1] - m[0]), float(np.hypot(e[k + 1], e[0]))
            tried = [{"add": c, "mn9": float(x)} for c, x in zip(helpful, m[1:])]
            accepted = gain > 2 * noise
            rounds.append({"round": rnd, "tried": tried, "added": helpful[k] if accepted else None,
                           "mn9": float(m[k + 1]), "gain": gain, "noise": noise})
            emit("round", **rounds[-1])
            if not accepted:
                break
            blend.append(helpful.pop(k))

        emit("stage", stage="confirm", text="Confirming with fresh random seeds (32 trials each)…")
        m, e, out = self.measure([[], blend], 32, seed + 9999,
                                 on_progress=lambda f: emit("progress", stage="confirm", value=f),
                                 on_live=live("confirm", [[], list(blend)], 32))
        confirm = {"baseline": float(m[0]), "baseline_sem": float(e[0]), "blend": float(m[1]),
                   "blend_sem": float(e[1]), "gain": float(m[1] - m[0]), "noise": float(np.hypot(e[0], e[1]))}
        result = {"baseline": base, "screen": screen, "retest": retest, "rounds": rounds,
                  "blend": [{k: v for k, v in self.by_id[c].items() if k != "neurons"} for c in blend],
                  "confirm": confirm, "sugar_hz": SUGAR_HZ, "channel_hz": CHANNEL_HZ, "trials": n_run}
        emit("result", **result)

        emit("stage", stage="portrait", text="Drawing the portrait from the neurons that responded…")
        portrait = self.portrait(out["rates"][1], out["rates"][0], out["spikes"][1], set(self.stim(blend)))
        return {"result": result, "portrait": portrait}

    # --- portrait data -------------------------------------------------------
    def portrait(self, rates, base_rates, spikes, stimulated: set[int], min_syn: int = 5) -> dict:
        """Everything needed to draw the brain's response to the favourite blend."""
        active = np.flatnonzero(rates > 0)
        first = np.full(self.brain.n, np.nan)
        if len(spikes):
            t, i = spikes[:, 0], spikes[:, 1]
            order = np.argsort(t, kind="stable")
            uniq, pos = np.unique(i[order], return_index=True)
            first[uniq] = t[order][pos]
        pre, post, syn, w = self.brain.edges
        is_act = np.zeros(self.brain.n, bool)
        is_act[active] = True
        m = is_act[pre] & is_act[post] & (syn >= min_syn)
        lab = self.anatomy.labels
        cls = lab.super_class.fillna("unlabelled").to_numpy()
        return {
            "neurons": [{"i": int(i), "x": float(self.anatomy.xyz[i, 0]), "y": float(self.anatomy.xyz[i, 1]),
                         "z": float(self.anatomy.xyz[i, 2]), "hz": round(float(rates[i]), 1),
                         "base_hz": round(float(base_rates[i]), 1),
                         "first_ms": None if np.isnan(first[i]) else int(first[i]),
                         "stim": int(i) in stimulated, "cls": cls[i],
                         "type": lab.cell_type.iloc[i] if isinstance(lab.cell_type.iloc[i], str) else ""}
                        for i in active],
            "edges": np.stack([pre[m], post[m], syn[m], np.sign(w[m])], 1).astype(int).tolist(),
            "spikes": spikes.ravel().tolist(),
            "mn9": self.mn9,
        }
