"""JEPA on RPE cell-cycle data (RPE_37C_Rep1 + Rep2).

Gene list: cophaser_CC_genes.csv (98 cell-cycle genes, mouse names → upper-cased
to match human gene symbols in the h5ad files).

No true phase labels are available; the script trains JEPA unsupervised and saves
the inferred embeddings for downstream analysis.
"""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from circadian_jepa.data.dataset import CircadianDataset
from circadian_jepa.data.views import ViewGenerator
from circadian_jepa.model.jepa import CircadianJEPA
from circadian_jepa.training.trainer import Trainer
from circadian_jepa.eval.inference import infer_phase

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/maxine/Documents/paychere/CoPhaser/data/cellcycle_maxine")
GENE_LIST_CSV = Path(
    "/home/maxine/Documents/andrea/context_repo/data/params_g/cophaser_CC_genes.csv"
)
OUT_DIR = Path(__file__).parent.parent / "results"

SEED = 42


# ── data loading ──────────────────────────────────────────────────────────────

def load_cellcycle() -> ad.AnnData:
    """Concatenate Rep1 + Rep2, subset to CC genes, expose spliced as 'counts'."""
    rep1 = ad.read_h5ad(DATA_DIR / "RPE_37C_Rep1_full.h5ad")
    rep2 = ad.read_h5ad(DATA_DIR / "RPE_37C_Rep2_full.h5ad")
    print(f"Rep1: {rep1.shape}  Rep2: {rep2.shape}")

    adata = ad.concat([rep1, rep2], label="replicate", keys=["Rep1", "Rep2"])
    adata.obs_names_make_unique()
    print(f"Concatenated: {adata.shape}")

    # CC gene list uses mouse-style capitalisation; RPE data has human ALL-CAPS
    cc_genes_mouse = pd.read_csv(GENE_LIST_CSV)["Gene"].tolist()
    cc_genes_human = [g.upper() for g in cc_genes_mouse]

    available = [g for g in cc_genes_human if g in adata.var_names]
    missing = [g for g in cc_genes_human if g not in adata.var_names]
    if missing:
        print(f"Warning: {len(missing)} CC genes not found in data: {missing}")
    print(f"Using {len(available)}/{len(cc_genes_human)} CC genes")

    adata = adata[:, available].copy()
    adata.layers["counts"] = adata.layers["spliced"].copy()

    return adata, available


# ── training ──────────────────────────────────────────────────────────────────

def run_jepa(
    adata: ad.AnnData,
    gene_list: list[str],
    view_mode: str,
    ema_mom: float,
    n_epochs: int,
    device: str,
) -> dict:
    label = f"cellcycle_{view_mode}_ema{ema_mom}"
    print(f"\n--- {label} ---")
    sys.stdout.flush()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    vg = ViewGenerator(view_mode=view_mode, seed=SEED)
    dataset = CircadianDataset(adata, view_generator=vg)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0
    )

    model = CircadianJEPA(n_genes=len(gene_list), ema_momentum=ema_mom)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device=device, log_every=9999)
    history = trainer.fit(
        loader, n_epochs=n_epochs, lambda_collapse=1.0, lambda_amplitude=0.1
    )

    result = infer_phase(model, adata, gene_list, device=device)

    lp = history[-1]["predict"]
    lc = history[-1]["collapse"]
    la = history[-1]["amplitude"]
    print(f"  loss: pred={lp:.4f}  collapse={lc:.4f}  amplitude={la:.4f}")

    inferred = result.obs["inferred_phase"].values
    amplitude = result.obs["inferred_amplitude"].values
    z0 = result.obs["z_0"].values
    z1 = result.obs["z_1"].values

    return {
        "label": label,
        "view_mode": view_mode,
        "ema_momentum": ema_mom,
        "n_epochs": n_epochs,
        "final_predict": round(float(lp), 4),
        "final_collapse": round(float(lc), 4),
        "final_amplitude": round(float(la), 4),
        "epoch_losses": history,
        "inferred_phase": inferred.tolist(),
        "inferred_amplitude": amplitude.tolist(),
        "z_0": z0.tolist(),
        "z_1": z1.tolist(),
        "cell_ids": list(adata.obs_names),
        "replicate": list(adata.obs["replicate"]) if "replicate" in adata.obs.columns else [],
        "batch": list(adata.obs["batch"].astype(str)),
    }


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    adata, gene_list = load_cellcycle()
    print(f"Dataset ready: {adata.n_obs} cells × {adata.n_vars} genes")

    all_results = {}

    for ema_mom in [0.95, 0.99]:
        res = run_jepa(
            adata,
            gene_list,
            view_mode="symmetric_split",
            ema_mom=ema_mom,
            n_epochs=40,
            device=device,
        )
        all_results[res["label"]] = res

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "cellcycle_jepa_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
