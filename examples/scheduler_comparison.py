#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a single, publication-ready 6-panel figure (one page, two-column spread)
AND save all data needed to reproduce each subplot as one CSV per subplot.

Change vs. previous version:
  - Subplot (f) is now the Overlap factor plot (O vs f), replacing constraint errors.
  - Overlap data are exported in subplot_f_overlap.csv.

Outputs:
  - scheduler_comparison_6panel.pdf
  - scheduler_comparison_6panel.png
  - figure_data/subplot_a_frequency_distribution.csv
  - figure_data/subplot_b_resolution_bandwidth.csv
  - figure_data/subplot_c_fractional_bin_number.csv
  - figure_data/subplot_d_segment_length.csv
  - figure_data/subplot_e_number_of_averages.csv
  - figure_data/subplot_f_overlap.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from speckit import SpectrumAnalyzer


# =============================================================================
# 1) ANALYSIS CONFIGURATION
# =============================================================================
ANALYSIS_PARAMS = {
    "olap": 0.75,
    "bmin": 3.5,
    "Lmin": 1000,
    "Jdes": 1000,
    "Kdes": 100,
    "force_target_nf": True,
}

SCHEDULERS = {
    "lpsd": {"name": "LPSD", "color": "deepskyblue", "ls": "-"},
    "ltf": {"name": "LTF", "color": "lime", "ls": "--"},
    "vectorized_ltf": {"name": "Vectorized LTF", "color": "gray", "ls": ":"},
    "new_ltf": {"name": "New LTF", "color": "tomato", "ls": "-."},
}

# Publication defaults (tweak as needed)
PUB = {
    "figsize": (7.2, 9.2),  # ~two-column width, near full page height
    "dpi_pdf": 300,
    "dpi_png": 300,
    "fontsize": 8,
    "lw": 1.6,
    "grid_lw": 0.45,
    "grid_ls": "--",
}

DATA_DIR = "figure_data"


# =============================================================================
# 2) HELPERS
# =============================================================================
def set_pub_rcparams():
    mpl.rcParams.update(
        {
            "font.size": PUB["fontsize"],
            "axes.titlesize": PUB["fontsize"],
            "axes.labelsize": PUB["fontsize"],
            "xtick.labelsize": PUB["fontsize"],
            "ytick.labelsize": PUB["fontsize"],
            "legend.fontsize": PUB["fontsize"] - 1,
            "lines.linewidth": PUB["lw"],
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax):
    ax.grid(True, which="both", linestyle=PUB["grid_ls"], linewidth=PUB["grid_lw"])
    ax.tick_params(direction="in", which="both", top=True, right=True)
    return ax


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
    )


def plot_schedulers(ax, x_data, y_data):
    for key, props in SCHEDULERS.items():
        if (key in x_data) and (key in y_data):
            ax.plot(
                x_data[key],
                y_data[key],
                color=props["color"],
                ls=props["ls"],
                label=props["name"],
            )
    return ax


def _pad_to_length(arr, n):
    a = np.asarray(arr, dtype=float)
    if a.shape[0] == n:
        return a
    out = np.full(n, np.nan, dtype=float)
    out[: a.shape[0]] = a
    return out


def save_subplot_csv(path, columns):
    """
    Save a dict of column_name -> 1D array-like to CSV, padding shorter arrays with NaNs.
    """
    maxlen = max(len(v) for v in columns.values()) if columns else 0
    df = pd.DataFrame({k: _pad_to_length(v, maxlen) for k, v in columns.items()})
    df.to_csv(path, index=False, float_format="%.18e")


# =============================================================================
# 3) DATA + PLANS
# =============================================================================
def build_plans():
    N = int(2e6)
    fs = 2.0
    random_data = np.random.randn(N)

    print("--- Generating Scheduler Plans ---")
    plans = {}
    for key, props in SCHEDULERS.items():
        print(f"Running {props['name']}...")
        analyzer = SpectrumAnalyzer(random_data, fs=fs, scheduler=key, **ANALYSIS_PARAMS)
        plans[key] = analyzer.plan()
        plans[key]["final_Jdes"] = analyzer.config["Jdes"]

    # Sanity checks
    print("\n--- Sanity Checks ---")
    nfs = [p["nf"] for p in plans.values()]
    assert len(set(nfs)) == 1, f"Mismatch in frequency counts: {nfs}"
    nf = nfs[0]
    print(f"Desired frequencies (Jdes): {ANALYSIS_PARAMS['Jdes']}")
    print(f"Actual frequencies produced (nf): {nf}\n")

    for key, p in plans.items():
        print(f"Scheduler: {SCHEDULERS[key]['name']}")
        print(f"  Final Jdes used: {p['final_Jdes']}")
        print(
            f"  Frequency range: {p['f'][0]:.4g} Hz to {p['f'][-1]:.4g} Hz (f_max={fs/2:.4g} Hz)"
        )
        print(
            f"  Segment length range (L): {np.min(p['L'])} to {np.max(p['L'])} (L_min={ANALYSIS_PARAMS['Lmin']})"
        )
        print("-" * 20)

    # Optional interpolation grid (kept for completeness / reproducibility)
    fmin = fs / N
    fmax = fs / 2
    f_logspace = np.logspace(np.log10(fmin), np.log10(fmax), ANALYSIS_PARAMS["Jdes"])
    _ = f_logspace  # silence linters if unused below

    # (Not used for plotting in this combined figure, but retained if you extend later)
    interp_results = {}
    for sched_key, plan in plans.items():
        interp_results[sched_key] = {}
        for qty in ["r", "b", "L", "K", "O"]:
            interp_func = interp1d(
                plan["f"],
                plan[qty],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            interp_results[sched_key][qty] = interp_func(f_logspace)

    return plans, fs, N


# =============================================================================
# 4) CSV EXPORT (one file per subplot)
# =============================================================================
def export_all_subplot_data(plans, fs, N, outdir=DATA_DIR):
    os.makedirs(outdir, exist_ok=True)

    fmin = fs / N
    fmax = fs / 2

    nf = next(iter(plans.values()))["nf"]
    j_nf = np.arange(nf, dtype=float)

    # ---- (a) Frequency point distribution ----
    Jdes = int(ANALYSIS_PARAMS["Jdes"])
    i_ref = np.arange(1, Jdes + 1, dtype=float)
    f_ideal_lin = np.linspace(fmin, fmax, Jdes)
    f_ideal_log = np.logspace(np.log10(fmin), np.log10(fmax), Jdes)

    cols_a = {"j_nf": j_nf}
    for key in SCHEDULERS.keys():
        cols_a[f"f_{key}_Hz"] = plans[key]["f"]
    cols_a.update(
        {
            "j_Jdes": i_ref,
            "f_ideal_linspace_Hz": f_ideal_lin,
            "f_ideal_logspace_Hz": f_ideal_log,
            "fmin_Hz": np.array([fmin]),
            "fmax_Hz": np.array([fmax]),
        }
    )
    save_subplot_csv(os.path.join(outdir, "subplot_a_frequency_distribution.csv"), cols_a)

    # ---- (b) Resolution bandwidth r(f) ----
    fresmin = fs / N
    freslim = fresmin * (1 + (1 - ANALYSIS_PARAMS["olap"]) * (ANALYSIS_PARAMS["Kdes"] - 1))
    cols_b = {
        "fres_min_Hz": np.array([fresmin]),
        "fres_lim_Hz": np.array([freslim]),
    }
    for key in SCHEDULERS.keys():
        cols_b[f"f_{key}_Hz"] = plans[key]["f"]
        cols_b[f"r_{key}_Hz"] = plans[key]["r"]
    save_subplot_csv(os.path.join(outdir, "subplot_b_resolution_bandwidth.csv"), cols_b)

    # ---- (c) Fractional bin number b(f) ----
    cols_c = {"bmin_constraint": np.array([ANALYSIS_PARAMS["bmin"]])}
    for key in SCHEDULERS.keys():
        cols_c[f"f_{key}_Hz"] = plans[key]["f"]
        cols_c[f"b_{key}"] = plans[key]["b"]
    save_subplot_csv(os.path.join(outdir, "subplot_c_fractional_bin_number.csv"), cols_c)

    # ---- (d) Segment length L(f) ----
    cols_d = {"Lmin_constraint": np.array([ANALYSIS_PARAMS["Lmin"]])}
    for key in SCHEDULERS.keys():
        cols_d[f"f_{key}_Hz"] = plans[key]["f"]
        cols_d[f"L_{key}"] = plans[key]["L"]
    save_subplot_csv(os.path.join(outdir, "subplot_d_segment_length.csv"), cols_d)

    # ---- (e) Number of averages K(f) ----
    cols_e = {"Kdes_target": np.array([ANALYSIS_PARAMS["Kdes"]])}
    for key in SCHEDULERS.keys():
        cols_e[f"f_{key}_Hz"] = plans[key]["f"]
        cols_e[f"K_{key}"] = plans[key]["K"]
    save_subplot_csv(os.path.join(outdir, "subplot_e_number_of_averages.csv"), cols_e)

    # ---- (f) Overlap factors O(f) ----
    cols_f = {"overlap_desired": np.array([ANALYSIS_PARAMS["olap"]])}
    for key in SCHEDULERS.keys():
        cols_f[f"f_{key}_Hz"] = plans[key]["f"]
        cols_f[f"overlap_{key}"] = plans[key]["O"]
    save_subplot_csv(os.path.join(outdir, "subplot_f_overlap.csv"), cols_f)

    print(f"\nSaved subplot CSVs under: {outdir}/")


# =============================================================================
# 5) 6-PANEL FIGURE
# =============================================================================
def make_6panel_figure(
    plans,
    fs,
    N,
    out_pdf="scheduler_comparison_6panel.pdf",
    out_png="scheduler_comparison_6panel.png",
):
    fmin = fs / N
    fmax = fs / 2

    set_pub_rcparams()

    fig, axes = plt.subplots(
        3,
        2,
        figsize=PUB["figsize"],
        dpi=PUB["dpi_pdf"],
        constrained_layout=True,
    )
    axes = axes.ravel()

    # Global legend (scheduler styles only)
    sched_handles = [
        Line2D([0], [0], color=props["color"], ls=props["ls"], label=props["name"])
        for props in SCHEDULERS.values()
    ]
    fig.legend(
        handles=sched_handles,
        loc="upper center",
        ncol=len(SCHEDULERS),
        frameon=True,
        edgecolor="black",
        fancybox=False,
        bbox_to_anchor=(0.5, 1.01),
    )

    # (a) Frequency point distribution
    ax = axes[0]
    x_data = {k: np.arange(p["nf"]) for k, p in plans.items()}
    y_data = {k: p["f"] for k, p in plans.items()}
    plot_schedulers(ax, x_data, y_data)

    i_ref = np.arange(1, ANALYSIS_PARAMS["Jdes"] + 1)
    ax.plot(i_ref, np.linspace(fmin, fmax, ANALYSIS_PARAMS["Jdes"]), c="black", ls="-", lw=1.0)
    ax.plot(
        i_ref,
        np.logspace(np.log10(fmin), np.log10(fmax), ANALYSIS_PARAMS["Jdes"]),
        c="0.4",
        ls="-",
        lw=1.0,
    )

    ax.set_title("Frequency point distribution")
    ax.set_xlabel("Frequency index, j")
    ax.set_ylabel("Frequency, f(j) [Hz]")
    ax.set_yscale("log")
    style_axis(ax)
    add_panel_label(ax, "(a)")
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", ls="-", lw=1.0, label="Ideal linspace"),
            Line2D([0], [0], color="0.4", ls="-", lw=1.0, label="Ideal logspace"),
        ],
        loc="lower right",
        frameon=True,
        edgecolor="black",
    )

    # (b) Resolution bandwidth
    ax = axes[1]
    x_data = {k: p["f"] for k, p in plans.items()}
    y_data = {k: p["r"] for k, p in plans.items()}
    plot_schedulers(ax, x_data, y_data)

    fresmin = fs / N
    freslim = fresmin * (1 + (1 - ANALYSIS_PARAMS["olap"]) * (ANALYSIS_PARAMS["Kdes"] - 1))
    ax.axhline(y=fresmin, color="0.35", ls="--", lw=1.0)
    ax.axhline(y=freslim, color="0.35", ls="--", lw=1.0)
    ax.text(0.02, 0.06, "Min resolution $f_{\\rm res,min}$", transform=ax.transAxes, color="0.25")
    ax.text(0.02, 0.14, "Avg target $f_{\\rm res,lim}$", transform=ax.transAxes, color="0.25")

    ax.set_title("Resolution bandwidth")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Resolution bandwidth, r [Hz]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axis(ax)
    add_panel_label(ax, "(b)")

    # (c) Fractional bin number
    ax = axes[2]
    x_data = {k: p["f"] for k, p in plans.items()}
    y_data = {k: p["b"] for k, p in plans.items()}
    plot_schedulers(ax, x_data, y_data)

    ax.axhline(y=ANALYSIS_PARAMS["bmin"], color="0.35", ls="--", lw=1.0)
    ax.text(0.02, 0.06, "$b_{\\min}$ constraint", transform=ax.transAxes, color="0.25")

    ax.set_title("Fractional bin number")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("b = f / r")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axis(ax)
    add_panel_label(ax, "(c)")

    # (d) Segment length
    ax = axes[3]
    x_data = {k: p["f"] for k, p in plans.items()}
    y_data = {k: p["L"] for k, p in plans.items()}
    plot_schedulers(ax, x_data, y_data)

    ax.axhline(y=ANALYSIS_PARAMS["Lmin"], color="0.35", ls="--", lw=1.0)
    ax.text(0.02, 0.06, "$L_{\\min}$ constraint", transform=ax.transAxes, color="0.25")

    ax.set_title("Segment length")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Segment length, L")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axis(ax)
    add_panel_label(ax, "(d)")

    # (e) Number of averages
    ax = axes[4]
    x_data = {k: p["f"] for k, p in plans.items()}
    y_data = {k: p["K"] for k, p in plans.items()}
    plot_schedulers(ax, x_data, y_data)

    ax.axhline(y=ANALYSIS_PARAMS["Kdes"], color="0.35", ls="--", lw=1.0)
    ax.text(0.02, 0.06, "$K_{\\rm des}$ target", transform=ax.transAxes, color="0.25")

    ax.set_title("Number of averages")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("K")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axis(ax)
    add_panel_label(ax, "(e)")

    # (f) Overlap factors (replaces constraint plot)
    ax = axes[5]
    x_data = {k: p["f"] for k, p in plans.items()}
    y_data = {k: p["O"] for k, p in plans.items()}
    plot_schedulers(ax, x_data, y_data)

    ax.axhline(y=ANALYSIS_PARAMS["olap"], color="0.35", ls="--", lw=1.0)
    ax.text(0.02, 0.06, "Desired overlap", transform=ax.transAxes, color="0.25")

    ax.set_title("Overlap factors")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Fractional overlap, O")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    style_axis(ax)
    add_panel_label(ax, "(f)")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=PUB["dpi_png"])
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")


def main():
    plans, fs, N = build_plans()
    export_all_subplot_data(plans, fs, N, outdir=DATA_DIR)
    make_6panel_figure(plans, fs, N)


if __name__ == "__main__":
    main()