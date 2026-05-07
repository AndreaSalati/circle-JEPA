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
    # Separate-JEPA Results — per cell-type

    JEPA trained independently per cell-type on MEGA data (`symmetric_split`, ema ∈ {0.95, 0.99}).
    Cell types on x-axis, EMA momentum as hue.
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

    _RESULT_DIR = Path(__file__).parent.parent / "results" / "separate_jepa"

    # pick the latest e100 run; fall back to whatever is available
    _candidates = sorted(_RESULT_DIR.glob("jepa_separate_*_e100.json"))
    if not _candidates:
        _candidates = sorted(_RESULT_DIR.glob("jepa_separate_*.json"))
    _JSON_PATH = _candidates[-1]
    print(f"Loading JEPA results: {_JSON_PATH.name}")

    with open(_JSON_PATH) as _f:
        results = json.load(_f)

    _first = next(iter(results.values()))
    print(f"JEPA gene set: {_first['set_type']}  |  view_mode: {_first['view_mode']}")

    # scritmo baseline (same 'big' gene set, same 5 celltypes)
    _SCRITMO_CSV = Path("/home/maxine/Documents/andrea/context_repo/results/separate/big_run.csv")
    df_scritmo_raw = pd.read_csv(_SCRITMO_CSV, index_col=0)
    print(f"Scritmo contexts: {sorted(df_scritmo_raw['context'].unique())}")

    TWO_PI = 2 * np.pi
    return df_scritmo_raw, np, pd, plt, results, sns


@app.cell
def _(df_scritmo_raw, pd, results):
    """Build long-form per-cell error dataframe for JEPA + scritmo baseline."""

    # ── JEPA rows ────────────────────────────────────────────────────────────
    _rows = []
    for _key, _exp in results.items():
        _ct = _exp["celltype"]
        _ema = _exp["ema_mom"]
        _model_label = f"ema={_ema}"
        for _err_h in _exp["per_cell_err_h"]:
            _rows.append(
                {
                    "celltype": _ct,
                    "organ": _exp["organ"],
                    "ema_mom": _ema,
                    "model": _model_label,
                    "err_h": float(_err_h),
                }
            )
    df_jepa = pd.DataFrame(_rows)

    # ── scritmo rows (MAE column is already per-cell absolute error in hours) ─
    df_sm = df_scritmo_raw[["context", "organ", "MAE"]].copy()
    df_sm = df_sm.rename(columns={"context": "celltype", "MAE": "err_h"})
    df_sm["ema_mom"] = float("nan")
    df_sm["model"] = "scritmo"

    df = pd.concat([df_jepa, df_sm], ignore_index=True)
    df["ct_short"] = df["celltype"].str.split("_", n=1).str[1]

    # order celltypes by median JEPA error (ignore scritmo for ordering)
    ct_order = (
        df_jepa.assign(ct_short=df_jepa["celltype"].str.split("_", n=1).str[1])
        .groupby("ct_short")["err_h"]
        .median()
        .sort_values()
        .index.tolist()
    )
    # scritmo always last in hue order, ema variants sorted first
    model_order = sorted(m for m in df["model"].unique() if m != "scritmo") + ["scritmo"]

    print(df.groupby(["ct_short", "model"])["err_h"].median().unstack()[model_order])
    return ct_order, df, model_order


@app.cell
def _(mo):
    mo.md("""
    ## Boxplots — per-cell absolute phase error (hours)
    """)
    return


@app.cell
def _(ct_order, df, model_order, plt, sns):
    _jepa_models = [m for m in model_order if m != "scritmo"]
    _jepa_colors = sns.color_palette("muted", len(_jepa_models))
    _pal = {m: c for m, c in zip(_jepa_models, _jepa_colors)}
    _pal["scritmo"] = "#d62728"  # red, matching skin-fibroblast notebook convention

    fig_box, _ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(
        data=df,
        x="ct_short",
        y="err_h",
        hue="model",
        order=ct_order,
        hue_order=model_order,
        palette=_pal,
        width=0.6,
        showfliers=False,
        ax=_ax,
    )
    _ax.set_xlabel("Cell type", fontsize=10)
    _ax.set_ylabel("Absolute phase error (hours)", fontsize=10)
    _ax.set_title(
        "Per-cell phase error — JEPA (sym_split) vs scritmo baseline\nMEGA data · big gene set",
        fontsize=11,
    )
    _ax.legend(title="Model", fontsize=8, title_fontsize=8)
    _ax.tick_params(axis="x", labelsize=8)
    sns.despine(ax=_ax)
    plt.tight_layout()
    fig_box
    return


@app.cell
def _(mo):
    mo.md("""
    ## Circular correlation — grouped bar chart (celltypes × EMA)
    """)
    return


@app.cell
def _(ct_order, model_order, np, plt, results, sns):
    _data = {_m: [] for _m in model_order}

    for _ct_short in ct_order:
        for _m in model_order:
            if _m == "scritmo":
                _data[_m].append(float("nan"))
                continue
            _ema = float(_m.split("=")[1])
            _match = [
                v for v in results.values()
                if v["celltype"].split("_", 1)[1] == _ct_short
                and abs(v["ema_mom"] - _ema) < 1e-6
            ]
            _data[_m].append(_match[0]["align_corr"] if _match else float("nan"))

    _x = np.arange(len(ct_order))
    _n_models = len(model_order)
    _width = 0.35
    _offsets = np.linspace(-(_n_models - 1) / 2, (_n_models - 1) / 2, _n_models) * _width

    _pal = sns.color_palette("muted", _n_models)
    fig_corr, _ax = plt.subplots(figsize=(10, 4))
    for _i, (_m, _off) in enumerate(zip(model_order, _offsets)):
        _vals = _data[_m]
        _colors = [_pal[_i] if v >= 0 else "#aaaaaa" for v in _vals]
        _bars = _ax.bar(
            _x + _off, _vals, _width,
            color=_colors, label=_m,
            edgecolor="white", linewidth=0.5,
        )
        for _bar, _v in zip(_bars, _vals):
            if not np.isnan(_v):
                _ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _v + (0.02 if _v >= 0 else -0.06),
                    f"{_v:.2f}",
                    ha="center",
                    va="bottom" if _v >= 0 else "top",
                    fontsize=7,
                )

    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(ct_order, fontsize=8)
    _ax.set_ylabel("Circular correlation (post-alignment)", fontsize=10)
    _ax.set_ylim(-0.9, 0.9)
    _ax.set_title("Circular correlation by cell type and EMA momentum", fontsize=11)
    _ax.legend(title="Model", fontsize=8, title_fontsize=8)
    sns.despine(ax=_ax)
    plt.tight_layout()
    fig_corr
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2D latent space — z_0 / z_1 coloured by ZT phase
    Each subplot is one (celltype, ema_mom) pair.
    A clean ring coloured smoothly by phase = healthy geometry.
    Collapse, k=2 doubling, or noisy clouds indicate failure modes.
    """)
    return


@app.cell
def _(np, plt, results):
    _celltypes = sorted({v["celltype"] for v in results.values()})
    _emas      = sorted({v["ema_mom"]  for v in results.values()})

    _n_rows = len(_celltypes)
    _n_cols = len(_emas)
    fig_latent, _axes = plt.subplots(
        _n_rows, _n_cols,
        figsize=(4 * _n_cols, 4 * _n_rows),
        squeeze=False,
    )

    _theta = np.linspace(0, 2 * np.pi, 300)

    for _r, _ct in enumerate(_celltypes):
        for _c, _ema in enumerate(_emas):
            _ax = _axes[_r][_c]

            _match = [
                v for v in results.values()
                if v["celltype"] == _ct and abs(v["ema_mom"] - _ema) < 1e-6
            ]
            if not _match or "z_0" not in _match[0]:
                _ax.text(0.5, 0.5, "z_0/z_1\nnot in JSON\n(re-run script)",
                         ha="center", va="center", transform=_ax.transAxes, fontsize=8)
                _ax.set_title(f"{_ct.split('_',1)[1]}  ema={_ema}", fontsize=8)
                continue

            _exp = _match[0]
            _z0    = np.array(_exp["z_0"])
            _z1    = np.array(_exp["z_1"])
            _phase = np.array(_exp["true_phase"])

            _sc = _ax.scatter(
                _z0, _z1,
                c=_phase, cmap="twilight",
                vmin=0, vmax=2 * np.pi,
                s=4, alpha=0.5, linewidths=0,
            )
            _ax.plot(np.cos(_theta), np.sin(_theta),
                     color="black", lw=0.8, linestyle="--", alpha=0.5)
            _ax.set_aspect("equal")
            _ax.axhline(0, color="grey", lw=0.4, alpha=0.4)
            _ax.axvline(0, color="grey", lw=0.4, alpha=0.4)
            _ax.set_title(f"{_ct.split('_',1)[1]}  ema={_ema}", fontsize=8)
            _ax.set_xlabel("z_0", fontsize=7)
            _ax.set_ylabel("z_1", fontsize=7)
            plt.colorbar(_sc, ax=_ax, label="ZT (rad)", fraction=0.046)

    fig_latent.suptitle(
        "2D latent space per (celltype, EMA momentum)\nUnit circle = dashed; colour = true ZT phase",
        fontsize=11,
    )
    plt.tight_layout()
    fig_latent
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary table — median MAE and circular correlation
    """)
    return


@app.cell
def _(pd, results):
    _rows = []
    for _v in results.values():
        _rows.append({
            "celltype": _v["celltype"].split("_", 1)[1],
            "organ": _v["organ"],
            "ema_mom": _v["ema_mom"],
            "n_cells": _v["n_cells"],
            "n_genes": _v["n_genes"],
            "align_corr": round(_v["align_corr"], 4),
            "median_mae_h": round(_v["median_mae_h"], 2),
        })
    df_summary = pd.DataFrame(_rows).sort_values(["celltype", "ema_mom"])
    df_summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## Training curves — prediction, collapse and amplitude losses
    """)
    return


@app.cell
def _(np, plt, results, sns):
    _jepa = list(results.items())
    _loss_keys = ["predict", "collapse", "amplitude"]
    _loss_titles = ["Prediction loss", "Collapse loss", "Amplitude loss"]

    _celltypes = sorted({v["celltype"] for v in results.values()})
    _emas = sorted({v["ema_mom"] for v in results.values()})
    _ct_pal = dict(zip(_celltypes, sns.color_palette("tab10", len(_celltypes))))
    _ls_map = {e: s for e, s in zip(_emas, ["-", "--", ":", "-."])}

    fig_loss, _axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    _handles_seen = {}
    for _ax, _lk, _lt in zip(_axes, _loss_keys, _loss_titles):
        for _key, _exp in _jepa:
            _hist = _exp.get("history", [])
            if not _hist:
                continue
            _vals = [_h[_lk] for _h in _hist if _lk in _h]
            _ep = np.arange(1, len(_vals) + 1)
            _ct = _exp["celltype"]
            _ema = _exp["ema_mom"]
            _lbl = f"{_ct.split('_', 1)[1]}  ema={_ema}"
            _line, = _ax.plot(
                _ep, _vals,
                color=_ct_pal[_ct],
                linestyle=_ls_map[_ema],
                linewidth=1.4,
                alpha=0.85,
                label=_lbl,
            )
            if _lbl not in _handles_seen:
                _handles_seen[_lbl] = _line
        _ax.set_title(_lt, fontsize=10)
        _ax.set_xlabel("Epoch")
        _ax.set_ylabel("Loss")

    _handles = list(_handles_seen.values())
    _labels = list(_handles_seen.keys())
    fig_loss.legend(
        _handles, _labels,
        loc="upper right", fontsize=7, ncol=1, frameon=False,
        bbox_to_anchor=(1.0, 1.0),
    )
    fig_loss.suptitle("JEPA training curves — per cell type", fontsize=12)
    plt.tight_layout()
    fig_loss
    return


if __name__ == "__main__":
    app.run()
