import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Real Data Results — JEPA vs scritmo baseline (Skin Fibroblasts)

    Comparison of circadian phase inference on **real scRNA-seq data**.
    **4 000 cells · 15 clock genes · 6 discrete timepoints (ZT2, ZT6, ZT10, ZT14, ZT18, ZT22) · skin fibroblasts.**
    JEPA trained with `symmetric_split` view mode only (best on synthetic benchmarks).
    """)
    return


@app.cell
def _():
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    _JSON_PATH = Path(__file__).parent.parent / "results" / "real_skin_fibroblast_results.json"
    with open(_JSON_PATH) as _f:
        results = json.load(_f)

    TWO_PI = 2 * np.pi
    return TWO_PI, np, pd, plt, results, sns


@app.cell
def _(TWO_PI, np):
    """Shared circular-geometry helpers, returned so other cells can use them."""

    def align_phase_local(pred, true):
        _best_err, _best_aligned = np.inf, pred
        for _sign in (1, -1):
            _flipped = _sign * pred
            _mu = np.arctan2(
                np.sin(true - _flipped).mean(),
                np.cos(true - _flipped).mean(),
            )
            _aligned = (_flipped + _mu) % TWO_PI
            _err = np.abs(((_aligned - true + np.pi) % TWO_PI) - np.pi).mean()
            if _err < _best_err:
                _best_err, _best_aligned = _err, _aligned
        return _best_aligned

    def circ_dist_hours(a, b):
        _diff = np.abs(a - b) % TWO_PI
        return np.minimum(_diff, TWO_PI - _diff) * 24 / TWO_PI

    def shorten_vm(view_mode):
        return (
            view_mode
            .replace("symmetric_split", "sym_split")
            .replace("light_independent", "light_ind")
        )

    return align_phase_local, circ_dist_hours, shorten_vm


@app.cell
def _(align_phase_local, circ_dist_hours, np, pd, results, shorten_vm):
    """Build long-form dataframe of per-cell absolute errors (hours) for all experiments."""

    _rows = []
    for _key, _exp in results.items():
        if _key == "scritmo_context_model":
            _errors = np.array(_exp["mae_per_cell"])
            _label = "scritmo"
        else:
            _inferred = np.array(_exp["inferred_phases"])
            _true = np.array(_exp["true_phases"])
            _aligned = align_phase_local(_inferred, _true)
            _errors = circ_dist_hours(_aligned, _true)
            _vm = shorten_vm(_exp["view_mode"])
            _label = f"{_vm}\n(ema={_exp['ema_momentum']})"

        for _e in _errors:
            _rows.append({"experiment": _label, "error_h": float(_e)})

    df_errors = pd.DataFrame(_rows)

    # sort experiments by median error, scritmo always last for visual separation
    _order = (
        df_errors.groupby("experiment")["error_h"]
        .median()
        .sort_values()
        .index.tolist()
    )
    order_map = [x for x in _order if x != "scritmo"] + (
        ["scritmo"] if "scritmo" in _order else []
    )
    return df_errors, order_map


@app.cell
def _(mo):
    mo.md("""
    ## Boxplots — per-cell absolute phase error
    """)
    return


@app.cell
def _(df_errors, order_map, plt, sns):
    _pal = {
        exp: ("#d62728" if exp == "scritmo" else c)
        for exp, c in zip(order_map, sns.color_palette("muted", len(order_map)))
    }

    fig_box, _ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df_errors,
        x="experiment",
        y="error_h",
        hue="experiment",
        order=order_map,
        hue_order=order_map,
        palette=_pal,
        width=0.55,
        legend=False,
        flierprops=dict(marker=".", markersize=2, alpha=0.25),
        ax=_ax,
    )
    _ax.axhline(12, color="gray", linestyle=":", linewidth=0.9, label="max possible (12 h)")
    _ax.set_xlabel("")
    _ax.set_ylabel("Absolute phase error (hours)", fontsize=10)
    _ax.set_title(
        "Per-cell phase error — JEPA (sym_split) vs scritmo baseline\nSkin fibroblasts · real data",
        fontsize=11,
    )
    _ax.legend(fontsize=8)
    _ax.tick_params(axis="x", labelsize=8)
    sns.despine(ax=_ax)
    plt.tight_layout()
    fig_box
    return


@app.cell
def _(mo):
    mo.md("""
    ## Circular correlation — bar chart
    """)
    return


@app.cell
def _(np, plt, results, shorten_vm, sns):
    _labels, _corrs = [], []
    for _key, _exp in sorted(results.items()):
        if _key == "scritmo_context_model":
            _short = "scritmo"
        else:
            _short = f"{shorten_vm(_exp['view_mode'])}\n(ema={_exp['ema_momentum']})"
        _labels.append(_short)
        _corrs.append(_exp["align_corr"])

    _idx_sort = np.argsort(_corrs)[::-1]
    _labels_s = [_labels[i] for i in _idx_sort]
    _corrs_s = [_corrs[i] for i in _idx_sort]
    _colors = [
        "#d62728" if l == "scritmo" else ("#4c72b0" if c > 0 else "#aaaaaa")
        for l, c in zip(_labels_s, _corrs_s)
    ]

    fig_corr, _ax = plt.subplots(figsize=(8, 4))
    _bars = _ax.bar(range(len(_labels_s)), _corrs_s, color=_colors,
                    edgecolor="white", linewidth=0.5)
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xticks(range(len(_labels_s)))
    _ax.set_xticklabels(_labels_s, fontsize=9)
    _ax.set_ylabel("Circular correlation (post-alignment)", fontsize=10)
    _ax.set_ylim(-0.8, 0.8)
    _ax.set_title(
        "Circular correlation — all experiments\n"
        "(blue = positive, grey = negative even after best alignment)",
        fontsize=10,
    )
    for _bar, _val in zip(_bars, _corrs_s):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _val + (0.02 if _val >= 0 else -0.06),
            f"{_val:.3f}",
            ha="center",
            va="bottom" if _val >= 0 else "top",
            fontsize=8,
        )
    sns.despine(ax=_ax)
    plt.tight_layout()
    fig_corr
    return


@app.cell
def _(mo):
    mo.md("""
    ## Polar histograms — predicted phase per true timepoint
    """)
    return


@app.cell
def _(TWO_PI, align_phase_local, np, plt, results, shorten_vm):
    import scritmo as sr

    _jepa_keys = [k for k in results if k != "scritmo_context_model"]
    _n = len(_jepa_keys) + 1
    _ncols = 3
    _nrows = -(-_n // _ncols)

    fig_polar, _axes = plt.subplots(
        _nrows, _ncols,
        figsize=(_ncols * 4.5, _nrows * 4.5),
        subplot_kw={"projection": "polar"},
    )
    _axes_flat = _axes.flatten() if _nrows * _ncols > 1 else [_axes]

    # scritmo first panel — convert predicted hours → radians
    _sm = results["scritmo_context_model"]
    sr.plot_phase_polar_population(
        np.array(_sm["pred_phase_h"]) * TWO_PI / 24,
        np.array(_sm["ext_time_hours"]),
        plot_type="histogram",
        bins=24,
        inner_ring_size=0.2,
        title="scritmo (context model)",
        xtick_fontsize=9,
        ax=_axes_flat[0],
    )

    for _i, _key in enumerate(_jepa_keys):
        _exp = results[_key]
        _aln = align_phase_local(
            np.array(_exp["inferred_phases"]),
            np.array(_exp["true_phases"]),
        )
        sr.plot_phase_polar_population(
            _aln,
            np.array(_exp["true_phases"]) * 24 / TWO_PI,
            plot_type="histogram",
            bins=24,
            inner_ring_size=0.2,
            title=f"{shorten_vm(_exp['view_mode'])}  ema={_exp['ema_momentum']}",
            xtick_fontsize=9,
            ax=_axes_flat[_i + 1],
        )

    for _ax in _axes_flat[_n:]:
        _ax.set_visible(False)

    fig_polar.suptitle(
        "Polar histograms of predicted phases per true timepoint — skin fibroblasts",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig_polar
    return


@app.cell
def _(mo):
    mo.md("""
    ## Training curves (JEPA only)
    """)
    return


@app.cell
def _(np, plt, results, shorten_vm):
    _jepa = [(k, v) for k, v in results.items() if k != "scritmo_context_model"]
    _loss_keys = ["predict", "collapse", "amplitude"]
    _loss_titles = ["Prediction loss", "Collapse loss", "Amplitude loss"]
    _cmap = plt.get_cmap("tab10")

    fig_loss, _axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for _ax, _lk, _lt in zip(_axes, _loss_keys, _loss_titles):
        for _i, (_key, _exp) in enumerate(_jepa):
            _hist = _exp["epoch_losses"]
            _vals = [_h[_lk] for _h in _hist]
            _ep = np.arange(1, len(_vals) + 1)
            _lbl = f"{shorten_vm(_exp['view_mode'])} ema={_exp['ema_momentum']}"
            _ax.plot(_ep, _vals, color=_cmap(_i), linewidth=1.5, label=_lbl, alpha=0.85)
        _ax.set_title(_lt, fontsize=10)
        _ax.set_xlabel("Epoch")
        _ax.set_ylabel("Loss")

    _handles, _labels = _axes[0].get_legend_handles_labels()
    fig_loss.legend(_handles, _labels, loc="upper right", fontsize=7.5, ncol=1, frameon=False)
    fig_loss.suptitle("JEPA training curves — skin fibroblasts (real data)", fontsize=12)
    plt.tight_layout()
    fig_loss
    return


if __name__ == "__main__":
    app.run()
