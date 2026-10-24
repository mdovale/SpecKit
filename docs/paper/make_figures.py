# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale (University of Arizona),
# Gerhard Heinzel (Albert Einstein Institute).

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
"""Regenerate the figures of the SpecKit manuscript (``docs/paper/main.tex``).

The original figures (``docs/paper/figures``) were laid out in DataGraph from
data computed in the notebooks under ``notebooks/``. This script recomputes the
data with SpecKit and redraws each figure with Matplotlib, reproducing the
DataGraph layout in PDF points: page size, axes frames, tick and label
placement, legend geometry, line styles, and colors.

Usage
-----
::

    python docs/paper/make_figures.py                  # every figure
    python docs/paper/make_figures.py fig01 fig03
    python docs/paper/make_figures.py --outdir build/paper-figures

Figures are written as ``figNN.pdf`` next to ``main.tex``, numbered in the
order the manuscript includes them.

Figures
-------
``fig01``
    Welch (SciPy) and LTF (SpecKit) ASD of uniform white noise with the
    LTF error band (notebook 0.0).
``fig02``
    Numba and NumPy computation times (notebook 0.3). The recorded timings are
    embedded below; ``--rerun-benchmark`` measures them on this machine.
``fig03``
    LPSD, LTF, and vectorized-LTF frequency plans (notebook 0.6).
``fig04``
    Window comparison on GRACE-FO LRI phase data (notebook 0.7). The 56 MB
    data file is downloaded on first use into ``build/paper-data`` or read
    from ``--grace-data``.
``fig05``
    Time-delay bias of ``|H_xy|`` and the coherence (notebook 0.1).
``fig06``, ``fig07``
    Synthetic three-input MISO system in the time domain, and its output ASD
    before and after optimal subtraction (notebook 1.0).
"""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from functools import cache, partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ft2font import FT2Font, Kerning, LoadFlags
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib.transforms import Affine2D

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = Path(__file__).resolve().parent
DEFAULT_GRACE = REPO / "build" / "paper-data" / "grace_fo_rawphase.mat"
GRACE_DRIVE_ID = "15IdylJxn79FEp-8If3rYTYOHRRknMKA-"

# ---------------------------------------------------------------------------
# DataGraph style, measured from the published PDFs (all lengths in points)
# ---------------------------------------------------------------------------

FONT_FAMILY = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
TEXT_SIZE = 8.0
SCRIPT_SIZE = 6.0
SUP_RISE = 2.95
SUB_DROP = 2.2356
STACKED_SUB_DROP = 3.065

MAJOR_TICK = 5.3333
MINOR_TICK = 2.6667
FRAME_WIDTH = 0.5
GRID_COLOR = "#797979"
GRID_WIDTH = 0.25
GRID_DASH = (3.0, 1.0)

XTICK_LABEL_DROP = 12.478  # tick-label baseline below the x axis
YTICK_LABEL_GAP = 6.933  # tick-label right edge left of the y axis
YTICK_LABEL_DROP = 2.565  # tick-label baseline below the tick
XLABEL_DROP = 24.466  # axis-label baseline below the x axis
YLABEL_GAP = 30.475  # axis-label baseline left of the y axis

LEGEND_PITCH = 9.6
LEGEND_TOP = 6.0
LEGEND_TEXT_DROP = 2.565

# Set1 and Apple crayon colors, converted from the Generic RGB profile to sRGB.
RED = "#ec3424"
BLUE = "#4492c5"
GREEN = "#5bb95c"
PURPLE = "#aa67b2"
ORANGE = "#ff9200"
LAVENDER = "#a8b9ee"
SALMON = "#ff7e79"
BLUEBERRY = "#0433ff"
BAND = "#98a9ea"
BLACK = "#000000"

MINUS = "\u2212"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "font.size": TEXT_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": FRAME_WIDTH,
    "axes.unicode_minus": True,
    "lines.scale_dashes": False,
    "lines.solid_capstyle": "butt",
    "lines.dash_capstyle": "butt",
    "lines.solid_joinstyle": "round",
    "lines.dash_joinstyle": "round",
    "path.simplify": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": MAJOR_TICK,
    "ytick.major.size": MAJOR_TICK,
    "xtick.minor.size": MINOR_TICK,
    "ytick.minor.size": MINOR_TICK,
    "xtick.major.width": FRAME_WIDTH,
    "ytick.major.width": FRAME_WIDTH,
    "xtick.minor.width": FRAME_WIDTH,
    "ytick.minor.width": FRAME_WIDTH,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "savefig.transparent": True,
    "figure.dpi": 72,
}


# ---------------------------------------------------------------------------
# Text: advance-width layout with sub- and superscripts
# ---------------------------------------------------------------------------


@cache
def _font(italic):
    prop = FontProperties(family=FONT_FAMILY, style="italic" if italic else "normal")
    return FT2Font(findfont(prop))


def text_width(s, size=TEXT_SIZE, italic=False):
    """Advance width of `s` in points, kerning included."""
    font = _font(italic)
    font.set_size(size, 72)
    width, prev = 0.0, None
    for ch in s:
        glyph = font.get_char_index(ord(ch))
        if prev is not None:
            width += font.get_kerning(prev, glyph, Kerning.UNFITTED) / 64
        width += font.load_glyph(glyph, LoadFlags.NO_HINTING).linearHoriAdvance / 65536
        prev = glyph
    return width


_MARKUP = re.compile(r"(\^\{[^}]*\}|_\{[^}]*\}|\\it\{[^}]*\})")


def _runs(s):
    """Split ``^{sup}``, ``_{sub}`` and ``\\it{italic}`` markup into runs."""
    out = []
    for tok in _MARKUP.split(s):
        if not tok:
            continue
        if tok.startswith("^{"):
            out.append((tok[2:-1], "sup"))
        elif tok.startswith("_{"):
            out.append((tok[2:-1], "sub"))
        elif tok.startswith("\\it{"):
            out.append((tok[4:-1], "it"))
        else:
            out.append((tok, "text"))
    return out


def layout_text(s):
    """Return ``(pieces, width)``; a piece is ``(text, size, italic, dx, dy)``.

    A subscript next to a superscript is stacked under it, as DataGraph does
    for ``S_xx^1/2`` and ``gamma_xy^2``.
    """
    runs = _runs(s)
    pieces, x, i = [], 0.0, 0
    while i < len(runs):
        txt, role = runs[i]
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        if role in ("sup", "sub") and nxt and {role, nxt[1]} == {"sup", "sub"}:
            sup = txt if role == "sup" else nxt[0]
            sub = nxt[0] if role == "sup" else txt
            pieces.append((sup, SCRIPT_SIZE, False, x, SUP_RISE))
            pieces.append((sub, SCRIPT_SIZE, False, x, -STACKED_SUB_DROP))
            x += max(text_width(sup, SCRIPT_SIZE), text_width(sub, SCRIPT_SIZE))
            i += 2
            continue
        if role == "sup":
            pieces.append((txt, SCRIPT_SIZE, False, x, SUP_RISE))
            x += text_width(txt, SCRIPT_SIZE)
        elif role == "sub":
            pieces.append((txt, SCRIPT_SIZE, False, x, -SUB_DROP))
            x += text_width(txt, SCRIPT_SIZE)
        else:
            italic = role == "it"
            pieces.append((txt, TEXT_SIZE, italic, x, 0.0))
            x += text_width(txt, TEXT_SIZE, italic)
        i += 1
    return pieces, x


def pow10_label(k):
    return "1" if k == 0 else "10^{" + str(k).replace("-", MINUS) + "}"


def num_label(s):
    return s.replace("-", MINUS)


# ---------------------------------------------------------------------------
# Page, panels, and legends positioned in PDF points
# ---------------------------------------------------------------------------


class Page:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width / 72, height / 72))
        self.pt = Affine2D().scale(1 / 72) + self.fig.dpi_scale_trans

    def text(self, x, y, s, ha="left", rotation=0, right_limit=None, zorder=5, ax=None):
        """Draw markup `s` with its baseline origin at (x, y).

        With `ax`, the text belongs to those axes and is drawn under their frame.
        """
        pieces, width = layout_text(s)
        start = {"left": 0.0, "center": -width / 2, "right": -width}[ha]
        if rotation == 0 and right_limit is not None:
            start = min(start, right_limit - x - width)
        target = self.fig if ax is None else ax
        for txt, size, italic, dx, dy in pieces:
            adv = start + dx
            px, py = (x + adv, y + dy) if rotation == 0 else (x - dy, y + adv)
            target.text(
                px,
                py,
                txt,
                transform=self.pt,
                fontsize=size,
                fontstyle="italic" if italic else "normal",
                ha="left",
                va="baseline",
                rotation=rotation,
                rotation_mode="anchor",
                zorder=zorder,
                clip_on=False,
            )

    def line(self, xs, ys, **kw):
        kw.setdefault("zorder", 6)
        art = Line2D(xs, ys, transform=self.pt, **kw)
        self.fig.add_artist(art)
        return art

    def rect(self, x, y, w, h, ax=None, **kw):
        kw.setdefault("zorder", 6)
        art = Rectangle((x, y), w, h, transform=self.pt, clip_on=False, **kw)
        (self.fig if ax is None else ax).add_artist(art)
        return art

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path)
        plt.close(self.fig)
        print(f"  wrote {path}")


class Panel:
    """Axes whose frame, ticks, and labels sit at DataGraph point positions."""

    def __init__(
        self,
        page,
        x0,
        y0,
        x1,
        y1,
        xlim,
        ylim,
        xlog=False,
        ylog=False,
        label_right=None,
        xlabel_drop=XLABEL_DROP,
        xlabel_center=None,
    ):
        self.page = page
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.xlim, self.ylim = xlim, ylim
        self.xlog, self.ylog = xlog, ylog
        self.label_right = page.w if label_right is None else label_right
        self.xlabel_drop = xlabel_drop
        self.xlabel_center = 0.5 * (x0 + x1) if xlabel_center is None else xlabel_center
        W, H = page.w, page.h
        ax = page.fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])
        ax.set_xscale("log" if xlog else "linear")
        ax.set_yscale("log" if ylog else "linear")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.patch.set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(FRAME_WIDTH)
            spine.set_capstyle("projecting")
            spine.set_zorder(4)
        ax.tick_params(which="both", top=False, right=False)
        ax.tick_params(labelbottom=False, labelleft=False)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([]))
            axis.set_minor_locator(FixedLocator([]))
            axis.set_major_formatter(NullFormatter())
            axis.set_minor_formatter(NullFormatter())
        self.ax = ax

    def _map(self, v, lim, log, p0, p1):
        if log:
            lo, hi, v = np.log10(lim[0]), np.log10(lim[1]), np.log10(v)
        else:
            lo, hi = lim
        return p0 + (v - lo) / (hi - lo) * (p1 - p0)

    def xpt(self, v):
        return self._map(v, self.xlim, self.xlog, self.x0, self.x1)

    def ypt(self, v):
        return self._map(v, self.ylim, self.ylog, self.y0, self.y1)

    def _inner(self, values, to_pt, p0, p1):
        return [v for v in values if p0 + 0.01 < to_pt(v) < p1 - 0.01]

    def xticks(self, major, labels=None, minor=(), grid=True, ticks=True):
        if ticks:
            self.ax.xaxis.set_major_locator(FixedLocator(list(major)))
            self.ax.xaxis.set_minor_locator(FixedLocator(list(minor)))
        else:
            self.ax.tick_params(axis="x", which="both", length=0)
        if grid:
            for v in self._inner(major, self.xpt, self.x0, self.x1):
                self.ax.axvline(v, **_grid_style(self.y1 - self.y0))
        if labels is not None:
            y = self.y0 - XTICK_LABEL_DROP
            for v, s in zip(major, labels):
                self.page.text(
                    self.xpt(v), y, s, ha="center", right_limit=self.label_right
                )

    def yticks(self, major, labels=None, minor=(), grid=True):
        self.ax.yaxis.set_major_locator(FixedLocator(list(major)))
        self.ax.yaxis.set_minor_locator(FixedLocator(list(minor)))
        if grid:
            for v in self._inner(major, self.ypt, self.y0, self.y1):
                self.ax.axhline(v, **_grid_style(self.x1 - self.x0))
        if labels is not None:
            x = self.x0 - YTICK_LABEL_GAP
            for v, s in zip(major, labels):
                self.page.text(x, self.ypt(v) - YTICK_LABEL_DROP, s, ha="right")

    def xlabel(self, s):
        self.page.text(self.xlabel_center, self.y0 - self.xlabel_drop, s, ha="center")

    def ylabel(self, s):
        y = 0.5 * (self.y0 + self.y1)
        self.page.text(self.x0 - YLABEL_GAP, y, s, ha="center", rotation=90)

    def annotate(self, dx, dy, s):
        """Label on a white box, placed `dx`, `dy` points from the axes origin."""
        x, y = self.x0 + dx, self.y0 + dy
        width = math.floor(layout_text(s)[1] + 6.0)
        self.page.rect(
            x - 3.0,
            y - 3.146,
            width,
            11.891,
            fc="white",
            ec="none",
            ax=self.ax,
            zorder=3,
        )
        self.page.text(x, y, s, zorder=3.5, ax=self.ax)


def _grid_style(length):
    """Dashes stretched to a whole number of periods, half a dash at each end."""
    on, off = GRID_DASH
    period = length / max(1, round(length / (on + off)))
    on, off = period * on / (on + off), period * off / (on + off)
    return {
        "color": GRID_COLOR,
        "lw": GRID_WIDTH,
        "ls": (on / 2, (on, off)),
        "dash_capstyle": "butt",
        "zorder": 0.5,
    }


def log_majors(lo_exp, hi_exp, step=1):
    exps = list(range(lo_exp, hi_exp + 1, step))
    return [10.0**k for k in exps], [pow10_label(k) for k in exps]


def log_minors(lim, subs=range(2, 10)):
    lo, hi = lim
    out = []
    for k in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for s in subs:
            v = s * 10.0**k
            if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                out.append(v)
    return out


def legend(page, x, y, w, h, columns, handle_rows=None):
    """DataGraph legend: white box, 0.5 pt frame, handle then text per row.

    Each column is ``(handle_offset, handle_length, text_offset, entries)`` with
    entries ``(draw_handle, label)``; ``draw_handle(page, x0, x1, y)`` draws the
    sample at row height `y`. `handle_rows` overrides the handle heights where
    DataGraph snapped thick handles to the half-point grid.
    """
    page.rect(x, y, w, h, fc="white", ec=BLACK, lw=FRAME_WIDTH, joinstyle="miter")
    for hx, hlen, tx, entries in columns:
        rows = [y + h - LEGEND_TOP - k * LEGEND_PITCH for k in range(len(entries))]
        hrows = handle_rows or rows
        for (draw, label), ry, hy in zip(entries, rows, hrows):
            draw(page, x + hx, x + hx + hlen, hy)
            page.text(x + tx, ry - LEGEND_TEXT_DROP, label, zorder=7)


def line_handle(color, lw=1.0, dashes=None, capstyle="butt", marker=None, **mk):
    def draw(page, x0, x1, y):
        kw = {
            "color": color,
            "lw": lw,
            "solid_capstyle": capstyle,
            "dash_capstyle": capstyle,
        }
        if dashes is not None:
            kw["ls"] = dashes
        if lw > 0:
            page.line([x0, x1], [y, y], **kw)
        if marker is not None:
            mk.setdefault("color", color)
            page.line([0.5 * (x0 + x1)], [y], ls="none", marker=marker, **mk)

    return draw


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SMALL_PAGE = (243.36, 162.0)
LEFT, RIGHT, WIDE_RIGHT, BOTTOM = 39.685, 237.6907, 240.5254, 28.3465


def welch_comparison(path, **_):
    """Welch vs LTF on uniform white noise (notebook 0.0)."""
    from scipy.signal import welch
    from scipy.signal.windows import kaiser

    from speckit import compute_spectrum

    np.random.seed(123)
    fs = 2.0
    x = np.random.rand(int(2e6))
    res = compute_spectrum(x, fs)
    f_w, p_w = welch(x, fs, window=kaiser(len(x) // 4, 30), nperseg=len(x) // 4)

    # The published band: asd +/- sqrt(3 Gxx / sqrt(navg)).
    asd = res.asd
    half = np.sqrt(3.0 * res.Gxx / np.sqrt(res.navg))
    lower, upper = asd - half, asd + half

    page = Page(*SMALL_PAGE)
    p = Panel(
        page,
        LEFT,
        BOTTOM,
        WIDE_RIGHT,
        159.1654,
        (1e-6, 1.0),
        (1e-2, 1.0),
        xlog=True,
        ylog=True,
    )
    p.xticks(*log_majors(-6, 0), minor=log_minors(p.xlim))
    p.yticks(*log_majors(-2, 0), minor=log_minors(p.ylim))

    ax = p.ax
    # DataGraph draws the DC bin of a log axis at the left edge.
    f_w = f_w.copy()
    f_w[0] = p.xlim[0]
    ax.plot(f_w, np.sqrt(p_w), color=SALMON, lw=1.0)
    ax.plot(res.f, lower, color=BLUE, lw=0.5)
    ax.fill_between(
        res.f, np.clip(lower, 1e-300, None), upper, color=BAND, alpha=0.5, lw=0
    )
    ax.plot(res.f, upper, color=BLUE, lw=0.5)
    ax.plot(res.f, asd, color=BLUEBERRY, lw=1.0)

    p.xlabel("Frequency, f(j) (Hz)")
    p.ylabel("Amplitude spectral density, S_{xx}^{1/2}")

    def band_handle(page, x0, x1, y):
        page.rect(x0, y - 4.08, x1 - x0, 4.08, fc=BAND, ec="none", alpha=0.5)
        page.line([x0, x1], [y, y], color=BLUE, lw=0.5)

    legend(
        page,
        165.525,
        31.965,
        71.0,
        31.2,
        [
            (
                3.2,
                9.6,
                16.0,
                [
                    (line_handle(SALMON), "Welch (SciPy)"),
                    (line_handle(BLUEBERRY), "LTF (SpecKit)"),
                    (band_handle, "\u00b13\u03c3"),
                ],
            )
        ],
    )
    page.save(path)


# Wall-clock seconds of compute_spectrum(np.random.rand(N), fs=2, order=k),
# measured with notebook 0.3 on an Apple M4 Max and read back from the
# published performance.pdf (N = 1e3, 1e4, 1e5, 1e6).
RECORDED_TIMINGS = {
    "numba": {
        -1: [0.01589, 0.04666, 0.1127, 0.6948],
        0: [0.01548, 0.05731, 0.2145, 0.8244],
        1: [0.01824, 0.07474, 0.2822, 1.130],
        2: [0.01751, 0.07789, 0.3183, 1.296],
    },
    "numpy": {
        -1: [0.01065, 0.1110, 0.8370, 6.726],
        0: [0.01175, 0.1450, 1.099, 9.276],
        1: [0.01468, 0.1743, 1.247, 10.97],
        2: [0.01623, 0.1945, 1.286, 10.84],
    },
}
BENCH_N = [1000, 10000, 100000, 1000000]
ORDER_COLORS = {-1: RED, 0: BLUE, 1: GREEN, 2: PURPLE}


def run_benchmark():
    """Repeat the notebook 0.3 timing loop on this machine."""
    from speckit import compute_spectrum, core

    np.random.seed(123)
    inputs = [np.random.rand(n) for n in BENCH_N]
    saved = core._NUMBA_ENABLED
    times = {"numba": {}, "numpy": {}}
    try:
        core._NUMBA_ENABLED = True
        compute_spectrum(inputs[0], fs=2, order=0)
        for backend, flag in (("numba", True), ("numpy", False)):
            core._NUMBA_ENABLED = flag
            for order in ORDER_COLORS:
                times[backend][order] = []
                for x in inputs:
                    t0 = time.time()
                    compute_spectrum(x, fs=2, order=order)
                    times[backend][order].append(time.time() - t0)
                print(
                    f"  {backend} order={order}: "
                    + ", ".join(f"{t:.4g}" for t in times[backend][order])
                )
    finally:
        core._NUMBA_ENABLED = saved
    return times


def performance(path, rerun_benchmark=False, **_):
    """Numba vs NumPy computation time (notebook 0.3)."""
    times = run_benchmark() if rerun_benchmark else RECORDED_TIMINGS

    page = Page(*SMALL_PAGE)
    p = Panel(
        page,
        LEFT,
        BOTTOM,
        RIGHT,
        116.6457,
        (1e3, 1e6),
        (3.7586e-3, 30.0),
        xlog=True,
        ylog=True,
    )
    p.xticks(*log_majors(3, 6), minor=log_minors(p.xlim))
    p.yticks(*log_majors(-2, 1), minor=log_minors(p.ylim))
    styles = {
        "numba": {"ls": "-", "marker": "o"},
        "numpy": {"ls": (0, (5, 3)), "marker": "s"},
    }
    for backend in ("numba", "numpy"):
        for order, color in ORDER_COLORS.items():
            p.ax.plot(
                BENCH_N,
                times[backend][order],
                color=color,
                lw=1.0,
                ms=5.0,
                mew=0,
                clip_on=False,
                **styles[backend],
            )
    p.xlabel("Length of the time series")
    p.ylabel("Computation time (s)")

    numba = [
        (line_handle(c, marker="o", ms=5.0, mew=0), f"Numba (order={k})")
        for k, c in ORDER_COLORS.items()
    ]
    numpy_ = [
        (
            line_handle(c, dashes=(0.9, (5, 3)), marker="s", ms=5.0, mew=0),
            f"NumPy (order={k})",
        )
        for k, c in ORDER_COLORS.items()
    ]
    legend(
        page,
        LEFT,
        119.346,
        197.0,
        40.8,
        [(3.2, 19.2, 28.8, numba), (101.705, 19.2, 127.305, numpy_)],
    )
    page.save(path)


SCHEDULER_PARAMS = {
    "olap": 0.75,
    "bmin": 3.5,
    "Lmin": 1000,
    "Jdes": 1000,
    "Kdes": 100,
    "force_target_nf": True,
}


def scheduler_comparison(path, **_):
    """Frequency plans of the three schedulers (notebook 0.6)."""
    from speckit import SpectrumAnalyzer

    N, fs = int(2e6), 2.0
    plans = {}
    for key in ("lpsd", "ltf", "vectorized_ltf"):
        plans[key] = SpectrumAnalyzer(
            np.zeros(N), fs=fs, scheduler=key, **SCHEDULER_PARAMS
        ).plan()
    style = {
        "lpsd": {"color": BLUE},
        "ltf": {"color": RED},
        "vectorized_ltf": {"color": LAVENDER, "ls": (0, (0.015, 4, 8, 4))},
    }
    fmin, fmax = fs / N, fs / 2
    dashed = {"color": BLACK, "lw": 1.5, "ls": (0, (5, 3))}

    page = Page(453.6, 468.0)
    origins = [
        (5.0, 288.0),
        (237.6, 288.0),
        (5.0, 146.5),
        (237.6, 146.5),
        (5.0, 5.0),
        (237.6, 5.0),
    ]
    ylims = [
        (10**-7.03580, 10**0.90276),
        (10**-6.58286, 10**-1.60227),
        (10**-0.40492, 1e3),
        (10**1.83378, 10**6.69877),
        (10**-0.79541, 10**5.30078),
        (0.61, 0.85),
    ]

    def panel(i, xlim, xlog=True):
        ox, oy = origins[i]
        # Each panel is a 216 pt wide DataGraph subgraph: tick labels stay
        # inside it and the x label is centered on it, not on the frame.
        return Panel(
            page,
            ox + 36.5,
            oy + 28.5,
            ox + 213.1654,
            oy + 141.5,
            xlim,
            ylims[i],
            xlog=xlog,
            ylog=True,
            label_right=ox + 216.0,
            xlabel_drop=25.195,
            xlabel_center=ox + 125.0,
        )

    def freq_axis(p, label="Frequency, f(j) (Hz)"):
        p.xticks(*log_majors(-6, 0), minor=log_minors(p.xlim))
        p.xlabel(label)

    def curves(p, x, y, mask_zero=False):
        for key, plan in plans.items():
            yy = np.asarray(plan[y], float)
            if mask_zero:
                yy = np.where(yy > 0, yy, np.nan)
            xx = np.arange(plan["nf"]) if x == "j" else plan[x]
            p.ax.plot(xx, yy, lw=1.5, **style[key])

    # (a) f(j) against the frequency index
    p = panel(0, (-20.0, 1000.0), xlog=False)
    p.xticks(
        range(0, 1001, 200),
        [str(v) for v in range(0, 1001, 200)],
        minor=range(0, 1001, 100),
    )
    p.yticks(*log_majors(-6, 0, 2), minor=[1e-7, 1e-5, 1e-3, 1e-1])
    p.ax.axhline(fmin, **dashed)
    p.ax.axhline(fmax, **dashed)
    j = np.arange(SCHEDULER_PARAMS["Jdes"])
    p.ax.plot(j, np.linspace(fmin, fmax, j.size), color=GREEN, lw=1.5)
    p.ax.plot(
        j, np.logspace(np.log10(fmin), np.log10(fmax), j.size), color=ORANGE, lw=1.5
    )
    curves(p, "j", "f")
    p.annotate(120.0, 20.255, "f_{min}")
    p.annotate(120.0, 105.255, "f_{max}")
    p.xlabel("Frequency index, j")
    p.ylabel("Frequency, f(j) (Hz)")

    # (b) r(j)
    p = panel(1, (1e-6, 1.0))
    freq_axis(p)
    p.yticks(*log_majors(-6, -2), minor=log_minors(p.ylim))
    p.ax.axhline(fs / N, **dashed)
    curves(p, "f", "r")
    p.annotate(68.0, 18.255, "r_{min}")
    p.ylabel("Frequency resolution, r(j) (Hz)")

    # (c) b(j)
    p = panel(2, (1e-6, 1.0))
    freq_axis(p, "Frequency \\it{f(j)} (Hz)")
    p.yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"], minor=log_minors(p.ylim))
    p.ax.axhline(SCHEDULER_PARAMS["bmin"], **dashed)
    curves(p, "f", "b")
    p.annotate(66.0, 37.255, "b_{min}")
    p.ylabel("Bin number, b(j)")

    # (d) L(j)
    p = panel(3, (1e-6, 1.0))
    freq_axis(p)
    exps = range(2, 7)
    p.yticks(
        [10.0**k for k in exps],
        [pow10_label(k) for k in exps],
        minor=log_minors(p.ylim),
    )
    p.ax.axhline(SCHEDULER_PARAMS["Lmin"], **dashed)
    curves(p, "f", "L")
    p.annotate(96.0, 33.255, "L_{min}")
    p.ylabel("Segment lengths, L(j)")

    # (e) K(j)
    p = panel(4, (1e-6, 1.0))
    freq_axis(p)
    exps = range(6)
    p.yticks([10.0**k for k in exps], [pow10_label(k) for k in exps])
    p.ax.axhline(SCHEDULER_PARAMS["Kdes"], **dashed)
    curves(p, "f", "K")
    p.annotate(65.0, 58.255, "K_{des}")
    p.ylabel("Number of averages, K(j)")

    # (f) overlap; single-segment bins (overlap 0) are left out of the log axis
    p = panel(5, (1e-6, 1.0))
    freq_axis(p)
    ticks = [0.65, 0.70, 0.75, 0.80, 0.85]
    p.yticks(
        ticks,
        [f"{v:.2f}" for v in ticks],
        minor=[round(0.61 + 0.01 * k, 2) for k in range(25)],
    )
    curves(p, "f", "O", mask_zero=True)
    p.ax.axhline(SCHEDULER_PARAMS["olap"], **dashed)
    p.annotate(112.0, 58.255, "Desired overlap")
    p.ylabel("Fractional overlap, \u03be(j)")

    legend(
        page,
        41.3,
        434.3,
        175.0,
        31.2,
        [
            (
                3.0,
                19.0,
                28.0,
                [
                    (line_handle(BLUE, 1.5), "LPSD"),
                    (line_handle(RED, 1.5), "LTF"),
                    (line_handle(LAVENDER, 1.5, (4.5, (8, 2))), "Vectorized LTF"),
                ],
            ),
            (
                90.5,
                19.0,
                115.5,
                [
                    (line_handle(GREEN, 1.5), "Linear spacing"),
                    (line_handle(ORANGE, 1.5), "Log spacing"),
                ],
            ),
        ],
        handle_rows=[459.5, 449.5, 440.5],
    )
    page.save(path)


def fetch_grace(path):
    """Download the GRACE-FO LRI phase file used in notebook 0.7."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    base = "https://drive.usercontent.google.com/download"
    query = {"id": GRACE_DRIVE_ID, "export": "download"}
    print(f"  downloading GRACE-FO data to {path} ...")
    with urllib.request.urlopen(f"{base}?{urllib.parse.urlencode(query)}") as r:
        body = r.read()
    if body[:6] == b"<html>" or b"<!DOCTYPE html>" in body[:200]:
        # Large files return a virus-scan warning page; resubmit its form.
        form = dict(re.findall(rb'name="([^"]+)" value="([^"]*)"', body))
        query = {k.decode(): html.unescape(v.decode()) for k, v in form.items()}
        with urllib.request.urlopen(f"{base}?{urllib.parse.urlencode(query)}") as r:
            body = r.read()
    if body[:6] == b"<html>" or b"<!DOCTYPE" in body[:200]:
        raise RuntimeError(
            "Google Drive did not return the data file; pass "
            "--grace-data with a local copy of data.mat"
        )
    tmp = path.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.rename(path)
    return path


def window_comparison(path, grace_data=None, **_):
    """Window functions on GRACE-FO LRI phase data (notebook 0.7)."""
    import scipy.signal.windows as win
    from scipy.io import loadmat

    import speckit.flattop as ft
    from speckit import compute_spectrum

    x = np.array(loadmat(fetch_grace(grace_data or DEFAULT_GRACE))["RawPhase"][:, 0])
    fs = 9.664
    windows = [
        ("boxcar", win.boxcar, BLACK, "boxcar (SciPy)"),
        ("nuttall", partial(win.nuttall, sym=False), RED, "nuttall (SciPy)"),
        ("hann", win.hann, BLUE, "hann (SciPy)"),
        ("kaiser", win.kaiser, PURPLE, "kaiser (NumPy)"),
        ("HFT248D", ft.HFT248D, GREEN, "HFT248D (SpecKit)"),
    ]
    res = {}
    for name, fun, _, _ in windows:
        print(f"  {name} ...")
        res[name] = compute_spectrum(
            x,
            fs=fs,
            win=fun,
            olap=0.75,
            Jdes=1000,
            force_target_nf=True,
            scheduler="vectorized_ltf",
            num_patch_pts=100,
        )
    f = res["boxcar"].f

    page = Page(*SMALL_PAGE)
    p = Panel(
        page,
        LEFT,
        BOTTOM,
        RIGHT,
        156.3307,
        (f[0], f[-1]),
        (1e-10, 1e6),
        xlog=True,
        ylog=True,
    )
    p.xticks(*log_majors(-5, 0), minor=log_minors(p.xlim))
    p.yticks(*log_majors(-9, 6, 3), minor=[10.0**k for k in range(-10, 7)])
    for name, _, color, _ in windows:
        p.ax.plot(res[name].f, res[name].asd, color=color, lw=1.0)
    p.xlabel("Frequency, f(j) (Hz)")
    p.ylabel("Amplitude spectral density, S_{xx}^{1/2}")
    legend(
        page,
        43.441,
        31.681,
        89.75,
        50.4,
        [
            (3.2, 9.6, 16.0, [(line_handle(c), s) for _, _, c, s in windows]),
        ],
    )
    page.save(path)


def delay_bias_error(path, **_):
    """Bias of |H_xy| and coherence from a time delay (notebook 0.1)."""
    import scipy.signal.windows as win

    from speckit import SpectrumAnalyzer
    from speckit.lagrange import timeshift

    N, fs, delay = int(1e6), 1e6, 1e-4
    np.random.seed(42)
    y0 = np.random.normal(0, 1, N)
    y1 = timeshift(y0, delay * fs)
    n_trunc = int(2 * delay * fs)
    y0, y1 = y0[n_trunc:-n_trunc], y1[n_trunc:-n_trunc]
    ref = SpectrumAnalyzer(data=[y0, y0], fs=fs, win=win.boxcar, olap=0.95)
    dly = SpectrumAnalyzer(data=[y0, y1], fs=fs, win=win.boxcar, olap=0.95)

    xs = np.linspace(0.01, 0.99, 40)
    coh_nbe, tf_nbe = [], []
    for x in xs:
        L = int(delay / x * fs)
        r = ref.compute_single_bin(freq=fs / 2.0, L=L)
        d = dly.compute_single_bin(freq=fs / 2.0, L=L)
        coh_nbe.append((d.coh[0] - r.coh[0]) / r.coh[0])
        tf_nbe.append((d.cf[0] - r.cf[0]) / r.cf[0])

    page = Page(*SMALL_PAGE)
    p = Panel(page, LEFT, BOTTOM, RIGHT, 156.3307, (0.0, 1.0), (-1.1, 0.0))
    major = [round(0.1 * k, 1) for k in range(11)]
    p.xticks(
        major,
        ["0"] + [f"{v:.1f}" for v in major[1:]],
        minor=[round(0.05 * k, 2) for k in range(21)],
    )
    ymaj = [round(-0.2 * k, 1) for k in range(6)][::-1]
    p.yticks(
        ymaj,
        [num_label(f"{v:.1f}") for v in ymaj[:-1]] + ["0"],
        minor=[round(-0.1 * k, 1) for k in range(12)],
    )
    ax = p.ax
    ax.plot(xs, (1 - xs) - 1, color=BLUE, lw=5.0)
    ax.plot(xs, (1 - xs) ** 2 - 1, color=GREEN, lw=5.0)
    marker = {
        "ls": "none",
        "ms": 5.0,
        "mew": 1.0,
        "mfc": "none",
        "mec": BLACK,
        "clip_on": False,
    }
    ax.plot(xs, tf_nbe, marker="s", **marker)
    ax.plot(xs, coh_nbe, marker="o", **marker)
    p.xlabel("Delay / Segment duration (\u0394t / T)")
    p.ylabel("Normalized bias error, NBE")

    def open_marker(m):
        return line_handle(BLACK, lw=0, marker=m, ms=5.0, mew=1.0, mfc="none")

    legend(
        page,
        136.941,
        113.531,
        98.0,
        40.8,
        [
            (
                3.2,
                9.6,
                16.0,
                [
                    (line_handle(BLUE, 5.0), "Theoretical NBE{|H_{xy}|}"),
                    (line_handle(GREEN, 5.0), "Theoretical NBE{\u03b3^{2}_{xy}}"),
                    (open_marker("s"), "Empirical NBE{|H_{xy}|}"),
                    (open_marker("o"), "Empirical NBE{\u03b3^{2}_{xy}}"),
                ],
            )
        ],
    )
    page.save(path)


def create_miso_data(n=100000, fs=1.0, noise_level=0.1, n_inputs=3, seed=42):
    """Synthetic MISO system of notebook 1.0: output = sum(g_i x_i) + noise."""
    np.random.seed(seed)
    t = np.arange(n) / fs
    gains = [1.5, -0.8, 2.0, 0.5, -1.2][:n_inputs]
    inputs = []
    for i in range(n_inputs):
        f = 0.01 * (i + 1)
        inputs.append(np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2 * t))
    output = np.zeros(n)
    for inp, gain in zip(inputs, gains):
        output += gain * inp
    output += noise_level * np.random.randn(n)
    return inputs, output, fs


def _miso_data():
    return create_miso_data(n=200000, fs=1.0, noise_level=0.2, n_inputs=3)


def miso_time_domain(path, **_):
    """First 1000 samples of the MISO inputs and output (notebook 1.0)."""
    inputs, output, _ = _miso_data()
    page = Page(*SMALL_PAGE)
    # DataGraph plots against the 1-based row number.
    n = 1300
    sample = np.arange(1, n + 1)
    rows = [
        ("Input 1", inputs[0], 131.0846, 156.3307),
        ("Input 2", inputs[1], 96.8386, 122.0846),
        ("Input 3", inputs[2], 62.5925, 87.8386),
        ("Output", output, BOTTOM, 53.5925),
    ]
    for label, y, y0, y1 in rows:
        is_output = label == "Output"
        ylim = (-5.0, 5.0) if is_output else (-1.4, 1.4)
        p = Panel(page, LEFT, y0, RIGHT, y1, (0.0, 1000.0), ylim)
        if is_output:
            p.xticks(
                range(0, 1001, 200),
                [str(v) for v in range(0, 1001, 200)],
                minor=range(0, 1001, 100),
            )
            p.yticks([-5, 0, 5], [num_label("-5"), "0", "5"], minor=range(-5, 6))
            p.xlabel("Sample")
        else:
            p.xticks(range(0, 1001, 200), ticks=False)
            p.yticks(
                [-1, 0, 1], [num_label("-1"), "0", "1"], minor=[-1, -0.5, 0, 0.5, 1]
            )
        p.ax.plot(sample, y[:n], color=BLACK, lw=0.5)
        p.ylabel(label)
    page.save(path)


def miso_frequency_domain(path, **_):
    """Output ASD with and without the inputs' contribution (notebook 1.0)."""
    from speckit import compute_spectrum
    from speckit.systems import (
        MISO_analytic_optimal_spectral_analysis,
        MISO_numeric_optimal_spectral_analysis,
    )

    inputs, output, fs = _miso_data()
    kw = {"order": -1, "win": "hann", "olap": 0.5}
    f_a, asd_a = MISO_analytic_optimal_spectral_analysis(inputs, output, fs, **kw)
    f_n, asd_n = MISO_numeric_optimal_spectral_analysis(inputs, output, fs, **kw)
    out = compute_spectrum(output, fs, **kw)

    page = Page(*SMALL_PAGE)
    p = Panel(
        page,
        LEFT,
        BOTTOM,
        RIGHT,
        127.9843,
        (out.f[0], out.f[-1]),
        (0.1, 100.0),
        xlog=True,
        ylog=True,
    )
    p.xticks(*log_majors(-5, -1), minor=log_minors(p.xlim))
    p.yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"], minor=log_minors(p.ylim))
    dotted = {"ls": (0, (0.015, 2.25)), "dash_capstyle": "round"}
    p.ax.plot(out.f, np.sqrt(out.Gxx), color=RED, lw=1.5)
    p.ax.plot(f_a, asd_a, color=BLACK, lw=1.5)
    p.ax.plot(f_n, asd_n, color=BLUE, lw=1.5, **dotted)
    p.xlabel("Frequency, f(j) (Hz)")
    p.ylabel("Amplitude spectral density, S_{xx}^{1/2}")
    legend(
        page,
        40.441,
        130.284,
        197.5,
        31.2,
        [
            (
                3.2,
                9.6,
                16.0,
                [
                    (line_handle(RED, 1.5), "Output ASD (with inputs influence)"),
                    (
                        line_handle(BLACK, 1.5),
                        "Optimal ASD (inputs influence removed, analytic)",
                    ),
                    (
                        line_handle(
                            BLUE, 1.5, (2.0025, (0.015, 2.25)), capstyle="round"
                        ),
                        "Optimal ASD (inputs influence removed, numeric)",
                    ),
                ],
            )
        ],
    )
    page.save(path)


FIGURES = {
    "fig01": welch_comparison,
    "fig02": performance,
    "fig03": scheduler_comparison,
    "fig04": window_comparison,
    "fig05": delay_bias_error,
    "fig06": miso_time_domain,
    "fig07": miso_frequency_domain,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "figures", nargs="*", metavar="FIGURE", help=f"subset of: {', '.join(FIGURES)}"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"output directory (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--grace-data",
        type=Path,
        default=None,
        help="local copy of the GRACE-FO data.mat file",
    )
    parser.add_argument(
        "--rerun-benchmark",
        action="store_true",
        help="time compute_spectrum here instead of using the recorded M4 Max timings",
    )
    args = parser.parse_args(argv)
    unknown = [name for name in args.figures if name not in FIGURES]
    if unknown:
        parser.error(f"unknown figure(s): {', '.join(unknown)}")

    plt.rcParams.update(RC)
    for name in args.figures or FIGURES:
        print(f"{name}:")
        t0 = time.time()
        FIGURES[name](
            args.outdir / f"{name}.pdf",
            grace_data=args.grace_data,
            rerun_benchmark=args.rerun_benchmark,
        )
        print(f"  done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
