"""Evaluate JEPA cell-cycle inference using CoPhaser-style quality metrics.

Metrics (no ground truth required):
  1. Histone fraction profile vs. inferred phase
  2. Library size profile + coherence score across replicates
  3. Jensen-Shannon divergence between replicates
  4. CCG acrophase ordering (inversions from G1→S→G2→M)
  5. Marker gene scatter profiles
  6. Mutual information between inferred phase and replicate label
  7. Amplitude distribution

All metric functions are ported directly from CoPhaser source to avoid the
dependency on the CoPhaser package.
"""

import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
from sklearn.metrics import mutual_info_score

# ── paths ─────────────────────────────────────────────────────────────────────

RESULTS_JSON = Path(__file__).parent.parent / "results" / "cellcycle_jepa_results.json"
OUT_DIR = Path(__file__).parent.parent / "results" / "cellcycle_metrics"
DATA_DIR = Path("/home/maxine/Documents/paychere/CoPhaser/data/cellcycle_maxine")
GENE_LIST_CSV = Path(
    "/home/maxine/Documents/andrea/context_repo/data/params_g/cophaser_CC_genes.csv"
)
CCG_ANNOTATED = Path(
    "/home/maxine/Documents/paychere/CoPhaser/src/CoPhaser/resources/CCG_annotated.csv"
)
COPHASER_SRC = Path("/home/maxine/Documents/paychere/CoPhaser/src")

# Load histone gene list from CoPhaser source (sys.path trick, no pip install needed)
sys.path.insert(0, str(COPHASER_SRC))
from CoPhaser import gene_sets  # noqa: E402

HISTONE_GENES = set(gene_sets.human_canonical_histones)
MARKERS = ["TOP2A", "PCNA", "MKI67", "MCM6", "AURKA", "CDK1"]
PHASE_ORDER = ["G1", "G1/S", "S", "G2", "G2/M", "M"]


# ── metric utilities (ported from CoPhaser) ────────────────────────────────────

def normalize_angles(x: np.ndarray) -> np.ndarray:
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def circular_std(angles: np.ndarray) -> float:
    R = np.sqrt(np.sin(angles).mean() ** 2 + np.cos(angles).mean() ** 2)
    return float(np.sqrt(-2 * np.log(R + 1e-12)))


def jensenshannon_phases(theta_a: np.ndarray, theta_b: np.ndarray, n_bins: int = 50) -> float:
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    p, _ = np.histogram(theta_a, bins=bins, density=True)
    q, _ = np.histogram(theta_b, bins=bins, density=True)
    p = p + 1e-10
    q = q + 1e-10
    m = (p + q) / 2
    jsd = 0.5 * np.sum(p * np.log2(p / m) + q * np.log2(q / m)) / n_bins
    return float(np.clip(jsd, 0.0, 1.0))


def pseudotime_mi(phase: np.ndarray, category: np.ndarray, n_bins: int = 50) -> float:
    pt_bins = pd.qcut(phase, q=n_bins, duplicates="drop")
    return float(mutual_info_score(pt_bins, category))


def coherence_score(curves: list[np.ndarray]) -> float:
    """Mean across-bin variance across groups (lower = more consistent)."""
    mat = np.stack(curves, axis=0)
    return float(mat.var(axis=0).mean())


def smooth_by_phase(
    X: np.ndarray, theta: np.ndarray, n_bins: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Bin cells by theta ∈ [-π, π], return (bin_centers, per-bin mean)."""
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    means = np.zeros((n_bins, X.shape[1]))
    for b in range(n_bins):
        mask = (theta >= bins[b]) & (theta < bins[b + 1])
        if mask.sum() > 0:
            means[b] = X[mask].mean(axis=0)
    return centers, means


def get_ptp_phase(bin_centers: np.ndarray, bin_means: np.ndarray, gene_names: list[str]) -> pd.DataFrame:
    """Peak-to-trough amplitude and peak phase per gene."""
    ptp = bin_means.max(axis=0) - bin_means.min(axis=0)
    peak_bin = bin_means.argmax(axis=0)
    peak_phase = bin_centers[peak_bin]
    return pd.DataFrame({"peak_to_peak": ptp, "phase": peak_phase}, index=gene_names)


def count_inversions(order: list, expected: list) -> int:
    idx_map = {g: i for i, g in enumerate(expected)}
    indices = [idx_map[g] for g in order if g in idx_map]
    inv = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] > indices[j]:
                inv += 1
    return inv


def best_order(phases_dict: dict, expected: list = PHASE_ORDER) -> tuple[list, int, int]:
    rotated = sorted(phases_dict, key=phases_dict.get)
    best_inv = 10**9
    best_rot = rotated
    best_dir = 1
    for _ in range(len(rotated)):
        for direction, seq in [(1, rotated), (-1, rotated[::-1])]:
            inv = count_inversions(seq, expected)
            if inv < best_inv:
                best_inv = inv
                best_rot = seq
                best_dir = direction
        rotated = rotated[1:] + [rotated[0]]
        if best_inv == 0:
            break
    return best_rot, best_inv, best_dir


# ── data loading ──────────────────────────────────────────────────────────────

def load_adata() -> tuple[ad.AnnData, list[str]]:
    rep1 = ad.read_h5ad(DATA_DIR / "RPE_37C_Rep1_full.h5ad")
    rep2 = ad.read_h5ad(DATA_DIR / "RPE_37C_Rep2_full.h5ad")
    adata = ad.concat([rep1, rep2], label="replicate", keys=["Rep1", "Rep2"])
    adata.obs_names_make_unique()

    cc_genes_human = [
        g.upper()
        for g in pd.read_csv(GENE_LIST_CSV)["Gene"].tolist()
    ]
    available = [g for g in cc_genes_human if g in adata.var_names]
    adata = adata[:, available].copy()
    adata.layers["counts"] = adata.layers["spliced"].copy()
    adata.obs["library_size"] = np.asarray(adata.layers["spliced"].sum(axis=1)).ravel()
    return adata, available


def add_histone_fraction(adata: ad.AnnData) -> None:
    hist_in_data = [g for g in adata.var_names if g.upper() in HISTONE_GENES]
    # histone fraction needs the full gene set, so use the original spliced from the full adata
    # we only have CC genes here — fall back to HIST-prefix match across full data
    # Since we subsetted to CC genes, histones won't be in adata.var_names.
    # We reload from the full adata for this metric only.
    rep1 = ad.read_h5ad(DATA_DIR / "RPE_37C_Rep1_full.h5ad")
    rep2 = ad.read_h5ad(DATA_DIR / "RPE_37C_Rep2_full.h5ad")
    full = ad.concat([rep1, rep2], label="replicate", keys=["Rep1", "Rep2"])
    full.obs_names_make_unique()

    hist_genes = [g for g in full.var_names if g.upper() in HISTONE_GENES]
    print(f"  Histone genes found in full data: {len(hist_genes)}")
    total = np.asarray(full.layers["spliced"].sum(axis=1)).ravel()
    hist_counts = np.asarray(full[:, hist_genes].layers["spliced"].sum(axis=1)).ravel()
    # align to adata obs order
    frac = pd.Series(hist_counts / (total + 1e-6), index=full.obs_names)
    adata.obs["histones_fraction"] = frac.reindex(adata.obs_names).values


# ── per-run evaluation ────────────────────────────────────────────────────────

def evaluate_run(
    run_key: str,
    run_data: dict,
    adata: ad.AnnData,
    gene_list: list[str],
    out_dir: Path,
    ccg_df: pd.DataFrame,
) -> dict:
    label = run_data["label"]
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    # Attach inferred phase to obs (convert [0, 2π) → [-π, π])
    phase_raw = np.array(run_data["inferred_phase"])  # [0, 2π)
    theta = normalize_angles(phase_raw)               # [-π, π]
    amplitude = np.array(run_data["inferred_amplitude"])

    adata = adata.copy()
    adata.obs["theta"] = theta
    adata.obs["amplitude"] = amplitude

    # Normalised expression matrix for smoothing (log1p CP10k, CC genes only)
    X_raw = np.asarray(adata.layers["spliced"].toarray()
                       if scipy.sparse.issparse(adata.layers["spliced"])
                       else adata.layers["spliced"])
    lib = adata.obs["library_size"].values[:, None]
    X_norm = np.log1p(X_raw / (lib + 1e-6) * 1e4)

    bin_centers, bin_means = smooth_by_phase(X_norm, theta, n_bins=50)
    df_ptp = get_ptp_phase(bin_centers, bin_means, gene_list)

    run_out = out_dir / label
    run_out.mkdir(parents=True, exist_ok=True)
    metrics = {"label": label}

    # ── 1. Histone fraction profile ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    rep_vals = adata.obs["replicate"].values
    for rep, color in [("Rep1", "tab:blue"), ("Rep2", "tab:orange")]:
        mask = rep_vals == rep
        th_rep = theta[mask]
        hf_rep = adata.obs["histones_fraction"].values[mask]
        centers, smoothed = smooth_by_phase(hf_rep[:, None], th_rep, n_bins=40)
        ax.plot(centers, smoothed[:, 0], label=rep, color=color)
    ax.set_xlabel("Inferred θ (rad)")
    ax.set_ylabel("Histone fraction")
    ax.set_title(f"Histone fraction profile — {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_out / "histone_profile.png", dpi=120)
    plt.close(fig)
    print("  [1] Histone fraction profile saved")

    # ── 2. Library size profile + coherence score ────────────────────────────
    curves = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for rep, color in [("Rep1", "tab:blue"), ("Rep2", "tab:orange")]:
        mask = rep_vals == rep
        th_rep = theta[mask]
        ls_rep = adata.obs["library_size"].values[mask]
        centers, smoothed = smooth_by_phase(ls_rep[:, None], th_rep, n_bins=40)
        ax.plot(centers, smoothed[:, 0], label=rep, color=color)
        curves.append(smoothed[:, 0])
    C = coherence_score(curves)
    ax.set_xlabel("Inferred θ (rad)")
    ax.set_ylabel("Median library size")
    ax.set_title(f"Library size profile — {label}  (C={C:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_out / "library_size_profile.png", dpi=120)
    plt.close(fig)
    metrics["coherence_score"] = round(C, 4)
    print(f"  [2] Coherence score C = {C:.4f}")

    # ── 3. Jensen-Shannon divergence between replicates ──────────────────────
    theta_rep1 = theta[rep_vals == "Rep1"]
    theta_rep2 = theta[rep_vals == "Rep2"]
    jsd = jensenshannon_phases(theta_rep1, theta_rep2)
    metrics["jsd_replicates"] = round(jsd, 4)
    print(f"  [3] JSD(Rep1, Rep2) = {jsd:.4f}")

    # ── 4. CCG acrophase ordering ────────────────────────────────────────────
    # Map gene → Peaktime label using CCG_annotated
    gene_to_peaktime = {}
    for _, row in ccg_df.iterrows():
        gene = str(row["Primary name"]).upper()
        pt = str(row["Peaktime"]).strip()
        if gene in gene_list and pt in PHASE_ORDER:
            gene_to_peaktime[gene] = pt

    # For each Peaktime group, get the mean peak phase of member genes
    df_ptp_filtered = df_ptp[df_ptp["peak_to_peak"] > 0.05]
    peaktime_phases = {}
    for gene, pt in gene_to_peaktime.items():
        if gene in df_ptp_filtered.index:
            peaktime_phases.setdefault(pt, []).append(df_ptp_filtered.loc[gene, "phase"])

    # Circular mean per phase category
    phase_category_mean = {}
    for pt, phases in peaktime_phases.items():
        a = np.array(phases)
        phase_category_mean[pt] = float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))

    if len(phase_category_mean) >= 2:
        order, n_inv, direction = best_order(phase_category_mean)
        metrics["ccg_inversions"] = n_inv
        metrics["ccg_direction"] = direction
        metrics["ccg_order"] = order
        print(f"  [4] CCG order: {order}  inversions={n_inv}  direction={direction}")
    else:
        metrics["ccg_inversions"] = None
        print(f"  [4] Not enough phase categories annotated ({len(phase_category_mean)})")

    # Polar plot of peak phases per gene coloured by Peaktime
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    cmap = plt.get_cmap("tab10")
    pt_colors = {pt: cmap(i) for i, pt in enumerate(PHASE_ORDER)}
    for gene, pt in gene_to_peaktime.items():
        if gene in df_ptp_filtered.index:
            ph = df_ptp_filtered.loc[gene, "phase"]
            ptp = df_ptp_filtered.loc[gene, "peak_to_peak"]
            ax.scatter(ph, ptp, color=pt_colors.get(pt, "gray"), alpha=0.7, s=40)
    for pt, color in pt_colors.items():
        ax.scatter([], [], color=color, label=pt, s=40)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    ax.set_title(f"CCG acrophase — {label}", pad=20)
    fig.tight_layout()
    fig.savefig(run_out / "ccg_acrophase.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  [4] CCG acrophase plot saved")

    # ── 5. Marker gene scatter profiles ─────────────────────────────────────
    markers_avail = [g for g in MARKERS if g in gene_list]
    if markers_avail:
        ncols = min(3, len(markers_avail))
        nrows = int(np.ceil(len(markers_avail) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axs = np.array(axs).ravel()
        for i, gene in enumerate(markers_avail):
            gi = gene_list.index(gene)
            y = X_norm[:, gi]
            # thin scatter for speed
            rng = np.random.default_rng(0)
            idx = rng.choice(len(theta), size=min(3000, len(theta)), replace=False)
            axs[i].scatter(theta[idx], y[idx], alpha=0.15, s=2, color="tab:red")
            # smoothed line
            axs[i].plot(bin_centers, bin_means[:, gi], color="black", lw=1.5)
            axs[i].set_title(gene)
            axs[i].set_xlabel("θ (rad)")
            axs[i].set_ylabel("log1p CP10k")
        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)
        fig.suptitle(f"Marker gene profiles — {label}", fontsize=13)
        fig.tight_layout()
        fig.savefig(run_out / "marker_profiles.png", dpi=120)
        plt.close(fig)
        print(f"  [5] Marker profiles saved ({markers_avail})")

    # ── 6. Mutual information with replicate ─────────────────────────────────
    rep_numeric = (rep_vals == "Rep1").astype(int)
    mi = pseudotime_mi(theta, rep_numeric)
    metrics["mi_phase_replicate"] = round(mi, 4)
    print(f"  [6] MI(phase, replicate) = {mi:.4f}")

    # ── 7. Amplitude distribution ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(amplitude, bins=60, color="steelblue", edgecolor="none")
    ax.set_xlabel("||z||  (JEPA amplitude)")
    ax.set_ylabel("# cells")
    ax.set_title(f"Amplitude distribution — {label}")
    fig.tight_layout()
    fig.savefig(run_out / "amplitude_dist.png", dpi=120)
    plt.close(fig)
    metrics["mean_amplitude"] = round(float(amplitude.mean()), 4)
    metrics["std_amplitude"] = round(float(amplitude.std()), 4)
    print(f"  [7] Amplitude: mean={amplitude.mean():.4f}  std={amplitude.std():.4f}")

    # ── 8. 2D latent space coloured by replicate ──────────────────────────────
    z0 = np.array(run_data["z_0"])
    z1 = np.array(run_data["z_1"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for rep, color, ax in zip(["Rep1", "Rep2"], ["tab:blue", "tab:orange"], axs):
        mask = rep_vals == rep
        ax.scatter(z0[mask], z1[mask], s=1, alpha=0.2, color=color, label=rep)
        ax.set_aspect("equal")
        ax.set_title(rep)
        ax.set_xlabel("z₀"); ax.set_ylabel("z₁")
    fig.suptitle(f"Latent space — {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(run_out / "latent_space.png", dpi=120)
    plt.close(fig)
    print("  [8] Latent space plot saved")

    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    adata, gene_list = load_adata()
    print(f"  {adata.n_obs} cells × {adata.n_vars} CC genes")

    print("Computing histone fractions (reads full h5ad) …")
    add_histone_fraction(adata)
    print(f"  Histone fraction: mean={adata.obs['histones_fraction'].mean():.4f}")

    print("Loading CCG annotation …")
    ccg_df = pd.read_csv(CCG_ANNOTATED)

    print("Loading inference results …")
    with open(RESULTS_JSON) as f:
        all_results = json.load(f)

    summary = {}
    for run_key, run_data in all_results.items():
        metrics = evaluate_run(run_key, run_data, adata, gene_list, OUT_DIR, ccg_df)
        summary[run_key] = metrics

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':45s}  {'JSD':>6s}  {'C':>6s}  {'MI':>6s}  {'Inv':>4s}  {'amp_μ':>6s}")
    print("-" * 85)
    for k, m in summary.items():
        inv = m.get("ccg_inversions", "N/A")
        inv_str = str(inv) if inv is not None else "N/A"
        print(
            f"  {k:43s}  {m['jsd_replicates']:>6.4f}  {m['coherence_score']:>6.1f}"
            f"  {m['mi_phase_replicate']:>6.4f}  {inv_str:>4s}  {m['mean_amplitude']:>6.4f}"
        )

    out = OUT_DIR / "metrics_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out}")
    print(f"Figures saved under {OUT_DIR}/")
