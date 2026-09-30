"""Every taste and smell input channel the fly has, as groups of FlyWire neurons.

Smell: all olfactory receptor neurons (ORNs) of one antennal-lobe glomerulus.
Taste: gustatory receptor neurons grouped by the FlyWire annotation sub-class.
Descriptions give the receptor and best-characterised ligands from the
literature (DoOR database and receptor papers); blank where not well established.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# glomerulus: (receptor, what it is known to detect)
GLOMERULI = {
    "D": ("Or69a", "terpenes, fruit volatiles"),
    "DA1": ("Or67d", "cVA, male pheromone"),
    "DA2": ("Or56a", "geosmin, mould"),
    "DA3": ("Or23a", ""),
    "DA4l": ("Or43a", "cyclohexanol"),
    "DA4m": ("Or2a", ""),
    "DC1": ("Or19a", "citrus terpenes (valencene)"),
    "DC2": ("Or13a", "octenol"),
    "DC3": ("Or83c", "farnesol, citrus peel"),
    "DC4": ("Ir64a", "acids"),
    "DL1": ("Or10a", "methyl salicylate"),
    "DL2d": ("Ir75b/c", "short-chain acids"),
    "DL2v": ("Ir75b/c", "short-chain acids"),
    "DL3": ("Or65a", "cVA"),
    "DL4": ("Or49a/Or85f", "parasitoid wasp odour"),
    "DL5": ("Or7a", "green leaf volatiles (E2-hexenal)"),
    "DM1": ("Or42b", "ethyl acetate, fermenting fruit"),
    "DM2": ("Or22a", "fruity esters (ethyl hexanoate)"),
    "DM3": ("Or47a", "pentyl acetate"),
    "DM4": ("Or59b", "methyl acetate"),
    "DM5": ("Or85a", "ethyl 3-hydroxybutyrate"),
    "DM6": ("Or67a", ""),
    "DP1l": ("Ir75a", "acetic acid, vinegar"),
    "DP1m": ("Ir64a", "acids"),
    "V": ("Gr21a/Gr63a", "carbon dioxide"),
    "VA1d": ("Or88a", "fly odours"),
    "VA1v": ("Or47b", "fly odours (pheromone)"),
    "VA2": ("Or92a", "diacetyl, yeast"),
    "VA3": ("Or67b", ""),
    "VA4": ("Or85d", ""),
    "VA5": ("Or49b", "cresols"),
    "VA6": ("Or82a", "geranyl acetate"),
    "VA7l": ("Or46a", ""),
    "VA7m": ("", ""),
    "VC1": ("Or33c", ""),
    "VC2": ("Or71a", ""),
    "VC3": ("Or35a", ""),
    "VC4": ("Or67c", ""),
    "VC5": ("Ir41a", "amines"),
    "VL1": ("Ir75d", ""),
    "VL2a": ("Ir84a", "phenylacetaldehyde"),
    "VL2p": ("Ir31a", ""),
    "VM1": ("Ir92a", "ammonia, amines"),
    "VM2": ("Or43b", "ethyl butyrate"),
    "VM3": ("Or9a", ""),
    "VM4": ("Ir76a", ""),
    "VM5d": ("Or85b", ""),
    "VM5v": ("Or98a", ""),
    "VM6l": ("", ""),
    "VM6m": ("", ""),
    "VM6v": ("", ""),
    "VM7d": ("Or42a", "fruity esters"),
    "VM7v": ("Or59c", ""),
}

TASTES = {
    "bitter": "labellar bitter sensors",
    "low-salt": "labellar low-salt sensors",
    "taste peg": "taste pegs inside the mouth",
    "pharyngeal_nerve_sensory_group1": "pharyngeal (throat) taste, group 1",
    "pharyngeal_nerve_sensory_group2": "pharyngeal (throat) taste, group 2",
    "pharyngeal_nerve_sensory_group3": "pharyngeal (throat) taste, group 3",
    "accessory_pharyngeal_nerve_sensory_group1": "accessory pharyngeal taste, group 1",
    "accessory_pharyngeal_nerve_sensory_group2": "accessory pharyngeal taste, group 2",
    "SA_VTV_pro_meso_meta": "taste sensors arriving from the legs/body",
}


def build_channels(labels: pd.DataFrame, exclude: set[int]) -> list[dict]:
    """labels: per-neuron annotation rows aligned to brain indices."""
    chans = []
    ct = labels.cell_type.fillna("")
    for g, (rec, lig) in GLOMERULI.items():
        idx = [i for i in np.flatnonzero(ct == f"ORN_{g}") if i not in exclude]
        if idx:
            chans.append({"id": f"smell:{g}", "kind": "smell", "name": g, "receptor": rec, "detects": lig,
                          "neurons": idx})
    sub = labels.cell_sub_class.fillna("")
    gust = labels.cell_class == "gustatory"
    for s, desc in TASTES.items():
        idx = [i for i in np.flatnonzero(gust & (sub == s)) if i not in exclude]
        if idx:
            chans.append({"id": f"taste:{s}", "kind": "taste", "name": desc, "receptor": "", "detects": "",
                          "neurons": idx})
    return chans
