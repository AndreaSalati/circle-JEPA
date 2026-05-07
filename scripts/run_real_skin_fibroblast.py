"""Real-data benchmark on skin fibroblasts (SKIN/concatenated.h5ad).

Mirrors run_phase5_comparison.py but uses:
- Real scRNA-seq data (skin fibroblasts, 6 circadian timepoints ZT2..ZT22)
- 4 000 cells subsampled from ~11 k fibroblasts
- Only symmetric_split view mode (best on synthetic data)
- scritmo context model as reference baseline
"""

import math, json, sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader

from circadian_jepa.data.dataset import CircadianDataset
from circadian_jepa.data.views import ViewGenerator
from circadian_jepa.model.jepa import CircadianJEPA
from circadian_jepa.training.trainer import Trainer
from circadian_jepa.eval.circular import circular_correlation, align_phase, mae
from circadian_jepa.eval.inference import infer_phase
import scritmo
import scritmo.ml as cr
from circadian_jepa.data.gene_lists import get_default_beta_path
from circadian_jepa.paths import get_data_root

DATA_PATH = get_data_root() / "SKIN/concatenated.h5ad"
N_CELLS = 4_000
SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def load_fibroblasts() -> ad.AnnData:
    """Load skin data, keep fibroblasts, subsample to N_CELLS, subset to CCG genes."""
    print(f"Loading {DATA_PATH} …")
    adata_full = ad.read_h5ad(DATA_PATH)

    fib = adata_full[adata_full.obs["celltype"] == "Fibroblast"].copy()
    print(f"  Fibroblasts: {fib.n_obs} cells × {fib.n_vars} genes")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(fib.n_obs, size=N_CELLS, replace=False)
    fib = fib[np.sort(idx)].copy()

    # ZT → radians (ZTmod is in hours, range 0–24)
    fib.obs["true_phase"] = (
        fib.obs["ZTmod"].values.astype(float) / 24.0 * 2 * math.pi
    ).astype(np.float32)

    # Subset to CCG genes and expose raw spliced counts as 'counts' layer
    params_g = scritmo.Beta(str(get_default_beta_path()))
    ccg_genes = [g for g in params_g.index if g in fib.var_names]
    adata = fib[:, ccg_genes].copy()
    adata.layers["counts"] = adata.layers["spliced"].copy()

    print(
        f"  After subsetting: {adata.n_obs} cells × {adata.n_vars} CCG genes"
    )
    print(f"  ZT timepoints: {sorted(adata.obs['ZTmod'].unique())}")
    return adata


def analyze_data(adata: ad.AnnData, label: str) -> np.ndarray:
    """Print per-gene cosinor R² against known ZT phase."""
    true = adata.obs["true_phase"].values
    print(f"\n{'='*60}")
    print(f"DATA: {label}  ({adata.n_obs} cells, {adata.n_vars} genes)")
    print(f"{'='*60}")
    import scipy.sparse as sp

    X = adata.layers["counts"]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    for i, gene in enumerate(adata.var_names):
        expr = X[:, i]
        X_mat = np.column_stack(
            [np.ones(len(true)), np.cos(true), np.sin(true)]
        )
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, expr, rcond=None)
        y_pred = X_mat @ coeffs
        ss_res = ((expr - y_pred) ** 2).sum()
        ss_tot = ((expr - expr.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        amp = math.sqrt(coeffs[1] ** 2 + coeffs[2] ** 2)
        print(f"  {gene:12s}  R2={r2:.3f}  amp={amp:.3f}")
    return true


def run_jepa(
    adata: ad.AnnData,
    view_mode: str,
    ema_mom: float,
    label: str,
    all_results: dict,
    device: str = "cpu",
) -> None:
    """Train JEPA and record results."""
    n_genes = adata.n_vars
    gene_list = list(adata.var_names)
    true = adata.obs["true_phase"].values

    key = f"{label}_{view_mode}_ema{ema_mom}"
    print(f"\n  --- {key} ---")
    sys.stdout.flush()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    vg = ViewGenerator(view_mode=view_mode, seed=SEED)
    dataset = CircadianDataset(adata, view_generator=vg)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0
    )

    model = CircadianJEPA(n_genes=n_genes, ema_momentum=ema_mom)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device=device, log_every=9999)
    history = trainer.fit(
        loader, n_epochs=40, lambda_collapse=1.0, lambda_amplitude=0.1
    )

    result = infer_phase(model, adata, gene_list, device=device)
    inferred = result.obs["inferred_phase"].values

    aligned, _offset, _sign = align_phase(inferred, true)
    align_corr = circular_correlation(aligned, true)
    m_ae = mae(aligned, true) * 24 / (2 * math.pi)

    try:
        shifted, _best_mad = scritmo.optimal_shift(inferred, true, verbose=False)
        scritmo_corr = circular_correlation(shifted, true)
    except Exception:
        scritmo_corr = None

    lp = history[-1]["predict"]
    lc = history[-1]["collapse"]
    la = history[-1]["amplitude"]
    print(f"    loss: pred={lp:.4f} coll={lc:.4f} amp={la:.4f}")
    print(f"    corr={align_corr:.4f}  MAE={m_ae:.2f}h  scritmo_corr={scritmo_corr}")
    sys.stdout.flush()

    all_results[key] = {
        "data": label,
        "view_mode": view_mode,
        "ema_momentum": ema_mom,
        "align_corr": round(align_corr, 4),
        "scritmo_corr": round(scritmo_corr, 4) if scritmo_corr is not None else None,
        "mae_hours": round(m_ae, 4),
        "final_predict": round(lp, 4),
        "final_collapse": round(lc, 4),
        "final_amplitude": round(la, 4),
        "epoch_losses": history,
        "inferred_phases": inferred.tolist(),
        "true_phases": true.tolist(),
    }


def run_scritmo_baseline(adata: ad.AnnData, all_results: dict) -> None:
    """Run scritmo context model as reference baseline."""
    print(f"\n{'='*60}")
    print("Running scritmo context model (reference baseline)")
    print(f"{'='*60}")
    sys.stdout.flush()

    params_g = scritmo.Beta(str(get_default_beta_path()))
    true_phase_rad = adata.obs["true_phase"].values

    adata_sr = adata.copy()
    adata_sr.obs["context"] = "skin_fibroblast"

    cmodel, _losses, _mad_epochs = cr.warmup_and_train(
        adata_sr,
        params_g,
        context=adata_sr.obs["context"],
        context_mode="none",
        fix_phase=True,
        noise_model="nb",
        batch_size=128,
        n_epochs=70,
        true_phase=true_phase_rad,
        init_mean=True,
        layer="spliced",
        kill_amps=True,
        n_theta_post=100,
        device="cpu",  # MPS produces NaN in scritmo's NB distribution
    )

    df_res = cmodel.create_results_df(
        adata_sr,
        ext_phase=true_phase_rad,
        context_col="context",
    )

    median_mae = float(np.median(df_res["MAE"]))
    pred_rad = df_res["pred_phase"].values
    aligned_sr, _off, _sgn = align_phase(pred_rad, true_phase_rad)
    sr_corr = circular_correlation(aligned_sr, true_phase_rad)
    print(f"  median MAE: {median_mae:.2f}h   corr: {sr_corr:.4f}")
    sys.stdout.flush()

    all_results["scritmo_context_model"] = {
        "data": "skin_fibroblast",
        "method": "scritmo_context_model",
        "align_corr": round(sr_corr, 4),
        "mae_hours": round(median_mae, 4),
        "pred_phase": df_res["pred_phase"].tolist(),
        "pred_phase_h": df_res["pred_phase_h"].tolist(),
        "true_phase": df_res["true_phase"].tolist(),
        "ext_time_hours": df_res["ext_time_hours"].tolist(),
        "mae_per_cell": df_res["MAE"].tolist(),
        "post_mean": df_res["post_mean"].tolist(),
        "post_std_c": df_res["post_std_c"].tolist(),
    }


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    adata = load_fibroblasts()
    analyze_data(adata, "skin fibroblast (real)")

    all_results: dict = {}

    for ema_mom in [0.95, 0.99]:
        run_jepa(adata, "symmetric_split", ema_mom, "skin_fib", all_results, device=device)

    run_scritmo_baseline(adata, all_results)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — all experiments")
    print(f"{'='*70}")
    print(
        f"{'Experiment':50s} {'corr':>7s} {'opt_corr':>8s} {'MAE(h)':>8s} {'pred':>7s}"
    )
    print("-" * 85)
    for k, v in sorted(all_results.items()):
        sc = (
            f"{v['scritmo_corr']:.4f}"
            if v.get("scritmo_corr") is not None
            else "     N/A"
        )
        pred = f"{v['final_predict']:.4f}" if "final_predict" in v else "     N/A"
        print(
            f"  {k:48s}  {v['align_corr']:>7.4f} {sc:>8s}"
            f" {v['mae_hours']:>8.2f} {pred:>7s}"
        )

    out = Path(__file__).parent.parent / "results" / "real_skin_fibroblast_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
