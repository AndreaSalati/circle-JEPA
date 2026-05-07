"""Run JEPA per-celltype on MEGA data — JEPA counterpart of run_MEGA_separate.py.

Mirrors the celltype list and gene-set logic from run_MEGA_separate.py but trains a
JEPA model per celltype instead of the scritmo context model.

Configuration:
- view_mode  = symmetric_split
- ema_mom    in [0.95, 0.99]
- n_epochs   = 100
- no cell sub-sampling
- per-cell error is recorded with the cell barcode
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

import scritmo as sr

from circadian_jepa.data.dataset import CircadianDataset
from circadian_jepa.data.views import ViewGenerator
from circadian_jepa.eval.circular import (
    align_phase,
    circular_correlation,
    circular_distance,
    mae,
)
from circadian_jepa.eval.inference import infer_phase
from circadian_jepa.model.jepa import CircadianJEPA
from circadian_jepa.training.trainer import Trainer
from circadian_jepa.paths import get_data_root, get_repo_root

# ── paths / constants ────────────────────────────────────────────────────────

MAIN_PATH = get_repo_root()
DATA_PATH = get_data_root() / "MEGA_data2.h5ad"
PARAMS_DIR = get_data_root() / "params_g"
GENE_SET_JSON = PARAMS_DIR / "gene_set_BIG_withImmune.json"
CCG_CSV = PARAMS_DIR / "ccg_zhang_context.csv"
RESULT_PATH = MAIN_PATH / "results" / "separate_jepa"

CONTEXT_COL = "organ_ncelltype"
SEED = 42

GOOD_CT = [
    "AORTA_Fibroblast",
    "AORTA_Smc",
    "LIVER_Hepatocyte",
    "SKIN_Fibroblast",
    "SKIN_Lyve+_Macrophage",
]


# ── helpers ──────────────────────────────────────────────────────────────────


def load_data() -> tuple[ad.AnnData, np.ndarray]:
    """Load MEGA_data2 with same filtering as run_MEGA_separate.py."""
    adata = sc.read_h5ad(DATA_PATH)
    adata = adata[adata.obs["new_celltype"] != "?"]
    adata = adata[~adata.obs["new_celltype"].str.contains("NK")].copy()
    adata = adata[~adata.obs.index.duplicated(keep=False)].copy()
    print(f"Loaded adata: {adata.shape}")

    ext_phase = adata.obs["ZTmod"].values.astype(float) * sr.w  # hours -> rad
    return adata, ext_phase


def get_gene_list(set_type: str, ct: str, params_g_ref: sr.Beta) -> list[str]:
    """Return the gene list to use for this celltype.

    set_type='big' uses the curated cell-type-specific list from
    gene_set_BIG_withImmune.json; 'ccg' uses the 15 core CCG clock genes.
    """
    if set_type == "big":
        with open(GENE_SET_JSON) as f:
            gene_sets = json.load(f)
        if ct not in gene_sets:
            raise KeyError(f"{ct} not in {GENE_SET_JSON.name}")
        return list(gene_sets[ct])
    if set_type == "ccg":
        return list(params_g_ref.index)
    raise ValueError(f"unknown set_type {set_type!r}")


def get_batch_size(n_cells: int) -> int:
    if n_cells < 500:
        return 32
    return 128


def run_jepa_for_celltype(
    adata_ct: ad.AnnData,
    ema_mom: float,
    n_epochs: int,
    view_mode: str,
    device: str,
    seed: int = SEED,
) -> dict:
    n_genes = adata_ct.n_vars
    gene_list = list(adata_ct.var_names)
    true = adata_ct.obs["true_phase"].values.astype(float)
    barcodes = adata_ct.obs.index.tolist()

    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = get_batch_size(adata_ct.n_obs)

    vg = ViewGenerator(view_mode=view_mode, seed=seed)
    dataset = CircadianDataset(adata_ct, view_generator=vg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    model = CircadianJEPA(n_genes=n_genes, ema_momentum=ema_mom)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device=device, log_every=9999)
    history = trainer.fit(
        loader, n_epochs=n_epochs, lambda_collapse=1.0, lambda_amplitude=0.1
    )

    result = infer_phase(model, adata_ct, gene_list, device=device)
    inferred = result.obs["inferred_phase"].values.astype(float)
    z_0 = result.obs["z_0"].values.astype(float)
    z_1 = result.obs["z_1"].values.astype(float)

    aligned, offset, sgn = align_phase(inferred, true)
    align_corr = circular_correlation(aligned, true)
    median_mae_h = mae(aligned, true) * 24 / (2 * math.pi)

    per_cell_err_rad = circular_distance(aligned, true)
    per_cell_err_h = per_cell_err_rad * 24 / (2 * math.pi)

    try:
        shifted, _ = sr.optimal_shift(inferred, true, verbose=False)
        scritmo_corr = float(circular_correlation(shifted, true))
    except Exception:
        scritmo_corr = None

    last = (
        history[-1]
        if history
        else {"predict": None, "collapse": None, "amplitude": None}
    )

    return {
        "n_cells": int(adata_ct.n_obs),
        "n_genes": int(n_genes),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "history": history,
        "barcodes": barcodes,
        "true_phase": true.tolist(),
        "inferred_phase": inferred.tolist(),
        "aligned_phase": aligned.tolist(),
        "per_cell_err_rad": per_cell_err_rad.tolist(),
        "per_cell_err_h": per_cell_err_h.tolist(),
        "align_corr": float(align_corr),
        "scritmo_corr": scritmo_corr,
        "median_mae_h": float(median_mae_h),
        "offset": float(offset),
        "sign": int(sgn),
        "z_0": z_0.tolist(),
        "z_1": z_1.tolist(),
        "final_predict": last["predict"],
        "final_collapse": last["collapse"],
        "final_amplitude": last["amplitude"],
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main(set_type: str, n_epochs: int, ema_moms: list[float], view_mode: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    adata, ext_phase = load_data()
    adata.obs["true_phase"] = ext_phase.astype(np.float32)
    params_g_ref = sr.Beta(str(CCG_CSV))

    context_celltype = adata.obs[CONTEXT_COL]
    all_results: dict = {}

    for ct in GOOD_CT:
        mask = (context_celltype == ct).values
        n_cells = int(mask.sum())
        if n_cells == 0:
            print(f"Skipping {ct}: 0 cells")
            continue
        organ = ct.split("_")[0]
        celltype = "_".join(ct.split("_")[1:])
        print(f"\n=== {ct} ({organ}/{celltype}) | {n_cells} cells ===")
        sys.stdout.flush()

        try:
            gene_names = get_gene_list(set_type, ct, params_g_ref)
        except Exception as e:
            print(f"Skipping {ct}: could not get gene list: {e}")
            continue

        common = [g for g in gene_names if g in adata.var_names]
        adata_ct = adata[mask, common].copy()
        if "spliced" in adata_ct.layers:
            adata_ct.layers["counts"] = adata_ct.layers["spliced"].copy()
        print(f"  genes: {len(common)} / {len(gene_names)} requested")

        for ema_mom in ema_moms:
            key = f"{ct}_{view_mode}_ema{ema_mom}"
            print(f"\n  --- {key} ---")
            sys.stdout.flush()
            res = run_jepa_for_celltype(adata_ct, ema_mom, n_epochs, view_mode, device)
            all_results[key] = {
                "celltype": ct,
                "organ": organ,
                "ema_mom": ema_mom,
                "view_mode": view_mode,
                "set_type": set_type,
                **res,
            }
            print(
                f"    corr={res['align_corr']:.4f}  MAE={res['median_mae_h']:.2f}h  "
                f"pred_loss={res['final_predict']}"
            )
            sys.stdout.flush()

        del adata_ct
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── persist ──────────────────────────────────────────────────────────────
    suffix = f"{set_type}_{view_mode}_e{n_epochs}"
    out_json = RESULT_PATH / f"jepa_separate_{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # also write a long-format per-cell CSV (one row per cell, easy to merge later)
    rows = []
    for key, v in all_results.items():
        for i, bc in enumerate(v["barcodes"]):
            rows.append(
                {
                    "key": key,
                    "celltype": v["celltype"],
                    "organ": v["organ"],
                    "ema_mom": v["ema_mom"],
                    "view_mode": v["view_mode"],
                    "barcode": bc,
                    "true_phase": v["true_phase"][i],
                    "inferred_phase": v["inferred_phase"][i],
                    "aligned_phase": v["aligned_phase"][i],
                    "err_rad": v["per_cell_err_rad"][i],
                    "err_h": v["per_cell_err_h"][i],
                }
            )
    if rows:
        df = pd.DataFrame(rows)
        out_csv = RESULT_PATH / f"jepa_separate_{suffix}_per_cell.csv"
        df.to_csv(out_csv, index=False)
        print(f"Per-cell CSV saved to {out_csv}")

    # summary table
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")
    print(f"{'Experiment':50s} {'n':>6s} {'corr':>7s} {'MAE(h)':>8s}")
    print("-" * 78)
    for k, v in sorted(all_results.items()):
        print(
            f"  {k:48s} {v['n_cells']:>6d} {v['align_corr']:>7.4f} "
            f"{v['median_mae_h']:>8.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JEPA per-celltype on MEGA data.")
    parser.add_argument(
        "--set_type",
        type=str,
        default="big",
        choices=["big", "ccg"],
        help="Gene set: 'big' = curated per-celltype list, 'ccg' = 15 core clock genes.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Training epochs per (celltype, ema_mom) run.",
    )
    parser.add_argument(
        "--ema_moms",
        type=float,
        nargs="+",
        default=[0.95, 0.99],
        help="EMA momenta to sweep over.",
    )
    parser.add_argument(
        "--view_mode",
        type=str,
        default="symmetric_split",
        help="ViewGenerator mode (default symmetric_split).",
    )
    parser.add_argument(
        "--no_train",
        action="store_true",
        help="Force n_epochs=0 (smoke-test the pipeline).",
    )
    args = parser.parse_args()
    n_epochs = 0 if args.no_train else args.n_epochs
    main(
        set_type=args.set_type,
        n_epochs=n_epochs,
        ema_moms=args.ema_moms,
        view_mode=args.view_mode,
    )
