"""Download the FlyWire v783 connectome + annotations and extract neuron populations.

Sources (public, no login):
  - Shiu et al. 2024 model repo: connectivity + neuron list, and their published
    taste-neuron ID lists (figures.ipynb)
  - FlyWire annotations (Schlegel et al. 2024): cell types, soma positions
"""
import json
import re
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SHIU = "https://github.com/philshiu/Drosophila_brain_model/raw/main/"
ANNO = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/"
FILES = {
    "Connectivity_783.parquet": SHIU + "Connectivity_783.parquet",
    "Completeness_783.csv": SHIU + "Completeness_783.csv",
    "figures.ipynb": SHIU + "figures.ipynb",
    "annotations.tsv": ANNO + "Supplemental_file1_neuron_annotations.tsv",
}


def main():
    DATA.mkdir(exist_ok=True)
    for name, url in FILES.items():
        path = DATA / name
        if not path.exists() or path.stat().st_size == 0:
            print("downloading", name)
            urllib.request.urlretrieve(url, path)

    nb = json.loads((DATA / "figures.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    pops = {}
    for key, var in [("sugar", "neu_sugar"), ("bitter", "neu_bitter"), ("salt", "neu_ir94e"), ("water", "neu_water")]:
        m = re.search(var + r"\s*=\s*\[(.*?)\]", src, re.S)
        pops[key] = [int(x) for x in re.findall(r"\d{18}", m.group(1))]
    # MN9: proboscis motor neuron pair (CB0701 in FlyWire v783), the feeding readout in Shiu et al.
    pops["MN9"] = [720575940660219265, 720575940618238523]
    (DATA / "populations.json").write_text(json.dumps(pops, indent=1))
    print({k: len(v) for k, v in pops.items()})


if __name__ == "__main__":
    main()
