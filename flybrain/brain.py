"""Whole-brain leaky integrate-and-fire model of the FlyWire v783 connectome.

A PyTorch port of Shiu et al. 2024 (Nature, "A Drosophila computational brain
model reveals sensorimotor processing"), github.com/philshiu/Drosophila_brain_model.
Same equations and constants as their Brian2 model; synaptic transmission is
event-driven (only neurons that spiked propagate), and trials run as a batch on
the GPU: NVIDIA (CUDA) or Apple (MPS), chosen automatically; set FLYBRAIN_DEVICE
to cuda, mps or cpu to override.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path(__file__).resolve().parent.parent / "data"

# On Apple GPUs, one fused Metal kernel does the per-neuron LIF update; the
# torch.where fallback below does the same on other devices.
_MSL = """
kernel void lif_step(device float* v [[buffer(0)]], device float* g [[buffer(1)]],
                     device short* refr [[buffer(2)]], device bool* spk [[buffer(3)]],
                     device int* counts [[buffer(4)]], constant short* rlen [[buffer(5)]],
                     constant float* prm [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  uint N = (uint)prm[5];
  float vi = v[i], gi = g[i]; short r = refr[i];
  if (r <= 0) { vi = vi + prm[0] * (prm[1] - vi + gi); gi = gi * prm[2]; }
  bool s = vi > prm[3];
  if (s) { vi = prm[4]; gi = 0; r = rlen[i % N]; counts[i] += 1; } else { r = r > 0 ? r - 1 : 0; }
  v[i] = vi; g[i] = gi; refr[i] = r; spk[i] = s;
}
"""
_lib = None


def default_device() -> str:
    """NVIDIA GPU (CUDA) if present, then Apple GPU (MPS), else CPU (slow)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mps_lib():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_MSL)
    return _lib


@dataclass
class Params:
    # Shiu et al. default_params (mV / ms)
    v_0: float = -52.0      # resting potential
    v_rst: float = -52.0    # reset potential
    v_th: float = -45.0     # spike threshold
    t_mbr: float = 20.0     # membrane time constant
    tau: float = 5.0        # synaptic time constant
    t_rfc: float = 2.2      # refractory period
    t_dly: float = 1.8      # synaptic delay
    w_syn: float = 0.275    # weight per synapse
    f_poi: float = 250.0    # Poisson input strength multiplier
    dt: float = 0.1         # integration step
    t_run: float = 1000.0   # trial duration
    n_run: int = 30         # trials per experiment


def neuron_sign_fix(flyids: np.ndarray) -> np.ndarray:
    """Per-neuron multiplier correcting the fast-transmission sign of Shiu et al.

    Their model treats every non-GABA/glutamate prediction as excitatory. That
    makes antennal-lobe local neurons (mostly GABAergic in vivo, often
    mispredicted as serotonin/dopamine) and neuromodulatory neurons act as fast
    excitation, and any odour input ignites ~10k neurons. We (1) make AL local
    neurons inhibitory unless experimentally known to be cholinergic, and
    (2) give dopamine/serotonin/octopamine no fast synaptic effect.
    Returns 1 (unchanged), -1 (flip excitatory to inhibitory) or 0 (silent).
    """
    a = pd.read_csv(DATA / "annotations.tsv", sep="\t", usecols=["root_id", "cell_class", "top_nt", "known_nt"],
                    low_memory=False).set_index("root_id").reindex(flyids)
    known_ach = a.known_nt.fillna("").str.startswith("acetylcholine").to_numpy()
    top = a.top_nt.fillna("").to_numpy()
    fix = np.ones(len(flyids), dtype=np.float32)
    fix[np.isin(top, ["dopamine", "serotonin", "octopamine"])] = 0.0
    lln = (a.cell_class == "ALLN").to_numpy() & ~known_ach
    # Shiu already made GABA/glutamate LNs negative; only flip the ones they made positive.
    fix[lln & ~np.isin(top, ["gaba", "glutamate"])] = -1.0
    return fix


class FlyBrain:
    def __init__(self, device: str | None = None, params: Params | None = None, nt_fix: bool = True):
        self.p = params or Params()
        self.device = torch.device(device or os.environ.get("FLYBRAIN_DEVICE") or default_device())

        comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
        con = pd.read_parquet(DATA / "Connectivity_783.parquet")
        self.flyids = comp.index.to_numpy()
        self.n = len(self.flyids)
        self.flyid2i = {f: i for i, f in enumerate(self.flyids)}

        # CSR by presynaptic neuron, so a spike can fetch its outgoing edges.
        con = con.sort_values("Presynaptic_Index", kind="stable")
        pre = con["Presynaptic_Index"].to_numpy()
        post = con["Postsynaptic_Index"].to_numpy()
        w = con["Excitatory x Connectivity"].to_numpy().astype(np.float32) * self.p.w_syn
        if nt_fix:
            w = w * neuron_sign_fix(self.flyids)[pre]
        rowptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(rowptr, pre + 1, 1)
        rowptr = np.cumsum(rowptr)

        self.edges = (pre.astype(np.int32), post.astype(np.int32), con["Connectivity"].to_numpy().astype(np.int32), w)

        d = self.device
        self.n_syn = len(pre)
        self.rowptr = torch.from_numpy(rowptr).to(d)
        self.deg = (self.rowptr[1:] - self.rowptr[:-1])
        self.post = torch.from_numpy(post.astype(np.int64)).to(d)
        self.w = torch.from_numpy(w).to(d)

    def ids_to_idx(self, flyids) -> list[int]:
        return [self.flyid2i[f] for f in flyids if f in self.flyid2i]

    def run(self, stim: dict[int, float], **kw) -> dict:
        """Single condition; see run_batch."""
        out = self.run_batch([stim], **kw)
        return {"rates": out["rates"][0], "spikes": out["spikes"][0]}

    @torch.no_grad()
    def run_batch(self, conditions: list[dict[int, float]], n_run: int | None = None,
                  t_run: float | None = None, seed: int = 0, on_progress=None,
                  watch: list[int] | None = None, record_spikes: bool = True,
                  on_live=None, live_every_ms: float = 25.0) -> dict:
        """Simulate several stimulus conditions at once, n_run trials each.

        conditions: list of {neuron index: Poisson rate in Hz}
        Returns
          rates:  (C, N) firing rate in Hz, mean over trials
          spikes: per condition, (K, 2) int array of (time ms, neuron index)
                  from the first trial, for replay/visualisation (if record_spikes)
          trial_rates: (C, R, len(watch)) per-trial rates of the `watch` neurons

        on_live(t_ms, frames, watch_hz) is called every `live_every_ms` of simulated
        time while the simulation runs: frames[c] = neuron indices that spiked in the
        first trial of condition c during that window; watch_hz[c] = mean firing rate
        so far of the `watch` neurons (all trials), or None.
        """
        p, d, N = self.p, self.device, self.n
        R = n_run or p.n_run
        C = len(conditions)
        B = C * R
        T = t_run or p.t_run
        steps = int(round(T / p.dt))
        delay = int(round(p.t_dly / p.dt))
        rfc_steps = int(round(p.t_rfc / p.dt))
        torch.manual_seed(seed)

        stim_all = sorted({i for c in conditions for i in c})
        col = {n: j for j, n in enumerate(stim_all)}
        prob = torch.zeros(C, max(len(stim_all), 1))
        for ci, c in enumerate(conditions):
            for n, r in c.items():
                prob[ci, col[n]] = r * p.dt / 1000.0
        prob = prob.repeat_interleave(R, 0).to(d)  # (B, S): trial b belongs to condition b // R
        stim_idx = torch.tensor(stim_all, dtype=torch.long, device=d)
        poi_kick = p.w_syn * p.f_poi

        v = torch.full((B, N), p.v_0, device=d)
        g = torch.zeros((B, N), device=d)
        refr = torch.zeros((B, N), dtype=torch.int16, device=d)
        # Poisson-driven neurons have no refractory period (as in Shiu et al.)
        refr_len = torch.full((N,), rfc_steps, dtype=torch.int16, device=d)
        if stim_all:
            refr_len[stim_idx] = 0
        use_msl = d.type == "mps"
        if use_msl:
            lib = _mps_lib()
            prm = torch.tensor([p.dt / p.t_mbr, p.v_0, float(np.exp(-p.dt / p.tau)), p.v_th, p.v_rst, N],
                               dtype=torch.float32, device=d)
            spk = torch.zeros((B, N), dtype=torch.bool, device=d)
        refr_len_b = refr_len.expand(B, N)
        counts = torch.zeros((B, N), dtype=torch.int32, device=d)
        ring = [None] * delay  # (b, n) spikes awaiting delivery
        a_mbr = p.dt / p.t_mbr
        decay_g = float(np.exp(-p.dt / p.tau))
        v_rst = torch.full_like(v, p.v_rst)
        zeros = torch.zeros_like(g)
        rec = []  # spikes of the first trial of each condition
        live_every = max(1, int(round(live_every_ms / p.dt)))
        live_buf = []
        watch_t = torch.tensor(watch, dtype=torch.long, device=d) if watch else None

        for t in range(steps):
            if on_progress and t % 500 == 0:
                on_progress(t / steps)
            # deliver spikes emitted `delay` steps ago
            arrived = ring[t % delay]
            if arrived is not None and arrived.numel():
                b, n = arrived[:, 0], arrived[:, 1]
                cnt = self.deg[n]
                ends = torch.cumsum(cnt, 0)
                total = int(ends[-1])
                if total:
                    rep = torch.repeat_interleave(torch.arange(len(n), device=d), cnt, output_size=total)
                    e = self.rowptr[n][rep] + torch.arange(total, device=d) - (ends - cnt)[rep]
                    g.view(-1).index_add_(0, b[rep] * N + self.post[e], self.w[e])

            if stim_all:
                v[:, stim_idx] += (torch.rand(B, len(stim_all), device=d) < prob) * poi_kick

            # integrate (neurons in refractory period are clamped), spike, reset
            if use_msl:
                lib.lif_step(v, g, refr, spk, counts, refr_len, prm)
            else:
                active = refr <= 0
                v = torch.where(active, v + a_mbr * (p.v_0 - v + g), v)
                g = torch.where(active, g * decay_g, g)
                spk = v > p.v_th
                v = torch.where(spk, v_rst, v)
                g = torch.where(spk, zeros, g)
                refr = torch.where(spk, refr_len_b, (refr - 1).clamp_min(0))
                counts += spk
            nz = spk.nonzero()
            ring[t % delay] = nz
            if record_spikes or on_live:
                first = nz[nz[:, 0] % R == 0]
                if first.numel():
                    if record_spikes:
                        rec.append(torch.cat([torch.full((len(first), 1), t, device=d), first], 1))
                    if on_live:
                        live_buf.append(first)
            if on_live and (t + 1) % live_every == 0:
                frames = [[] for _ in range(C)]
                if live_buf:
                    f = torch.cat(live_buf).cpu().numpy()
                    for c in range(C):
                        frames[c] = f[f[:, 0] == c * R, 1].tolist()
                live_buf = []
                wh = None
                if watch_t is not None:
                    sim_s = (t + 1) * p.dt / 1000.0
                    wh = (counts[:, watch_t].view(C, R, -1).float().mean((1, 2)) / sim_s).cpu().tolist()
                on_live((t + 1) * p.dt, frames, wh)

        per_trial = counts.view(C, R, N).float() / (T / 1000.0)
        rates = per_trial.mean(1).cpu().numpy()
        trial_rates = per_trial[:, :, watch].cpu().numpy() if watch else None
        rec = torch.cat(rec).cpu().numpy() if rec else np.zeros((0, 3), dtype=np.int64)
        spikes = []
        for ci in range(C):
            r = rec[rec[:, 1] == ci * R]
            spikes.append(np.stack([(r[:, 0] * p.dt).astype(np.int32), r[:, 2].astype(np.int32)], 1))
        return {"rates": rates, "spikes": spikes, "trial_rates": trial_rates}
