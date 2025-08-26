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
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from typing import Any, List, Optional, Tuple, Union

# Define custom colormaps for consistency and aesthetic quality
cm_data = [
    [0.2422, 0.1504, 0.6603],
    [0.2444, 0.1534, 0.6728],
    [0.2464, 0.1569, 0.6847],
    [0.2484, 0.1607, 0.6961],
    [0.2503, 0.1648, 0.7071],
    [0.2522, 0.1689, 0.7179],
    [0.2540, 0.1732, 0.7286],
    [0.2558, 0.1773, 0.7393],
    [0.2576, 0.1814, 0.7501],
    [0.2594, 0.1854, 0.7610],
    [0.2611, 0.1893, 0.7719],
    [0.2628, 0.1932, 0.7828],
    [0.2645, 0.1972, 0.7937],
    [0.2661, 0.2011, 0.8043],
    [0.2676, 0.2052, 0.8148],
    [0.2691, 0.2094, 0.8249],
    [0.2704, 0.2138, 0.8346],
    [0.2717, 0.2184, 0.8439],
    [0.2729, 0.2231, 0.8528],
    [0.2740, 0.2280, 0.8612],
    [0.2749, 0.2330, 0.8692],
    [0.2758, 0.2382, 0.8767],
    [0.2766, 0.2435, 0.8840],
    [0.2774, 0.2489, 0.8908],
    [0.2781, 0.2543, 0.8973],
    [0.2788, 0.2598, 0.9035],
    [0.2794, 0.2653, 0.9094],
    [0.2798, 0.2708, 0.9150],
    [0.2802, 0.2764, 0.9204],
    [0.2806, 0.2819, 0.9255],
    [0.2809, 0.2875, 0.9305],
    [0.2811, 0.2930, 0.9352],
    [0.2813, 0.2985, 0.9397],
    [0.2814, 0.3040, 0.9441],
    [0.2814, 0.3095, 0.9483],
    [0.2813, 0.3150, 0.9524],
    [0.2811, 0.3204, 0.9563],
    [0.2809, 0.3259, 0.9600],
    [0.2807, 0.3313, 0.9636],
    [0.2803, 0.3367, 0.9670],
    [0.2798, 0.3421, 0.9702],
    [0.2791, 0.3475, 0.9733],
    [0.2784, 0.3529, 0.9763],
    [0.2776, 0.3583, 0.9791],
    [0.2766, 0.3638, 0.9817],
    [0.2754, 0.3693, 0.9840],
    [0.2741, 0.3748, 0.9862],
    [0.2726, 0.3804, 0.9881],
    [0.2710, 0.3860, 0.9898],
    [0.2691, 0.3916, 0.9912],
    [0.2670, 0.3973, 0.9924],
    [0.2647, 0.4030, 0.9935],
    [0.2621, 0.4088, 0.9946],
    [0.2591, 0.4145, 0.9955],
    [0.2556, 0.4203, 0.9965],
    [0.2517, 0.4261, 0.9974],
    [0.2473, 0.4319, 0.9983],
    [0.2424, 0.4378, 0.9991],
    [0.2369, 0.4437, 0.9996],
    [0.2311, 0.4497, 0.9995],
    [0.2250, 0.4559, 0.9985],
    [0.2189, 0.4620, 0.9968],
    [0.2128, 0.4682, 0.9948],
    [0.2066, 0.4743, 0.9926],
    [0.2006, 0.4803, 0.9906],
    [0.1950, 0.4861, 0.9887],
    [0.1903, 0.4919, 0.9867],
    [0.1869, 0.4975, 0.9844],
    [0.1847, 0.5030, 0.9819],
    [0.1831, 0.5084, 0.9793],
    [0.1818, 0.5138, 0.9766],
    [0.1806, 0.5191, 0.9738],
    [0.1795, 0.5244, 0.9677],
    [0.1778, 0.5349, 0.9641],
    [0.1773, 0.5401, 0.9602],
    [0.1768, 0.5452, 0.9560],
    [0.1764, 0.5504, 0.9516],
    [0.1755, 0.5554, 0.9473],
    [0.1740, 0.5605, 0.9432],
    [0.1716, 0.5655, 0.9393],
    [0.1686, 0.5705, 0.9357],
    [0.1649, 0.5755, 0.9323],
    [0.1610, 0.5805, 0.9289],
    [0.1573, 0.5854, 0.9254],
    [0.1540, 0.5902, 0.9218],
    [0.1513, 0.5950, 0.9182],
    [0.1492, 0.5997, 0.9147],
    [0.1475, 0.6043, 0.9113],
    [0.1461, 0.6089, 0.9080],
    [0.1446, 0.6135, 0.9050],
    [0.1429, 0.6180, 0.9022],
    [0.1408, 0.6226, 0.8998],
    [0.1383, 0.6272, 0.8975],
    [0.1354, 0.6317, 0.8953],
    [0.1321, 0.6363, 0.8932],
    [0.1288, 0.6408, 0.8910],
    [0.1253, 0.6453, 0.8887],
    [0.1219, 0.6497, 0.8862],
    [0.1185, 0.6541, 0.8834],
    [0.1152, 0.6584, 0.8804],
    [0.1119, 0.6627, 0.8770],
    [0.1085, 0.6669, 0.8734],
    [0.1048, 0.6710, 0.8695],
    [0.1009, 0.6750, 0.8653],
    [0.0964, 0.6789, 0.8609],
    [0.0914, 0.6828, 0.8562],
    [0.0855, 0.6865, 0.8513],
    [0.0789, 0.6902, 0.8462],
    [0.0713, 0.6938, 0.8409],
    [0.0628, 0.6972, 0.8355],
    [0.0535, 0.7006, 0.8299],
    [0.0433, 0.7039, 0.8242],
    [0.0328, 0.7071, 0.8183],
    [0.0234, 0.7103, 0.8124],
    [0.0155, 0.7133, 0.8064],
    [0.0091, 0.7163, 0.8003],
    [0.0046, 0.7192, 0.7941],
    [0.0019, 0.7220, 0.7878],
    [0.0009, 0.7248, 0.7815],
    [0.0018, 0.7275, 0.7752],
    [0.0046, 0.7301, 0.7688],
    [0.0094, 0.7327, 0.7623],
    [0.0162, 0.7352, 0.7558],
    [0.0253, 0.7376, 0.7492],
    [0.0369, 0.7400, 0.7426],
    [0.0504, 0.7423, 0.7359],
    [0.0638, 0.7446, 0.7292],
    [0.0770, 0.7468, 0.7224],
    [0.0899, 0.7489, 0.7156],
    [0.1023, 0.7510, 0.7088],
    [0.1141, 0.7531, 0.7019],
    [0.1252, 0.7552, 0.6950],
    [0.1354, 0.7572, 0.6881],
    [0.1448, 0.7593, 0.6812],
    [0.1532, 0.7614, 0.6741],
    [0.1609, 0.7635, 0.6671],
    [0.1678, 0.7656, 0.6599],
    [0.1741, 0.7678, 0.6527],
    [0.1799, 0.7699, 0.6454],
    [0.1853, 0.7721, 0.6379],
    [0.1905, 0.7743, 0.6303],
    [0.1954, 0.7765, 0.6225],
    [0.2003, 0.7787, 0.6146],
    [0.2061, 0.7808, 0.6065],
    [0.2118, 0.7828, 0.5983],
    [0.2178, 0.7849, 0.5899],
    [0.2244, 0.7869, 0.5813],
    [0.2318, 0.7887, 0.5725],
    [0.2401, 0.7905, 0.5636],
    [0.2491, 0.7922, 0.5546],
    [0.2589, 0.7937, 0.5454],
    [0.2695, 0.7951, 0.5360],
    [0.2809, 0.7964, 0.5266],
    [0.2929, 0.7975, 0.5170],
    [0.3052, 0.7985, 0.5074],
    [0.3176, 0.7994, 0.4975],
    [0.3301, 0.8002, 0.4876],
    [0.3424, 0.8009, 0.4774],
    [0.3548, 0.8016, 0.4669],
    [0.3671, 0.8021, 0.4563],
    [0.3795, 0.8026, 0.4454],
    [0.3921, 0.8029, 0.4344],
    [0.4050, 0.8031, 0.4233],
    [0.4184, 0.8030, 0.4122],
    [0.4322, 0.8028, 0.4013],
    [0.4463, 0.8024, 0.3904],
    [0.4608, 0.8018, 0.3797],
    [0.4753, 0.8011, 0.3691],
    [0.4899, 0.8002, 0.3586],
    [0.5044, 0.7993, 0.3480],
    [0.5187, 0.7982, 0.3374],
    [0.5329, 0.7970, 0.3267],
    [0.5470, 0.7957, 0.3159],
    [0.5609, 0.7943, 0.3050],
    [0.5748, 0.7929, 0.2941],
    [0.5886, 0.7913, 0.2833],
    [0.6024, 0.7896, 0.2726],
    [0.6161, 0.7878, 0.2622],
    [0.6297, 0.7859, 0.2521],
    [0.6433, 0.7839, 0.2423],
    [0.6567, 0.7818, 0.2329],
    [0.6701, 0.7796, 0.2239],
    [0.6833, 0.7773, 0.2155],
    [0.6963, 0.7750, 0.2075],
    [0.7091, 0.7727, 0.1998],
    [0.7218, 0.7703, 0.1924],
    [0.7344, 0.7679, 0.1852],
    [0.7468, 0.7654, 0.1782],
    [0.7590, 0.7629, 0.1717],
    [0.7710, 0.7604, 0.1658],
    [0.7829, 0.7579, 0.1608],
    [0.7945, 0.7554, 0.1570],
    [0.8060, 0.7529, 0.1546],
    [0.8172, 0.7505, 0.1535],
    [0.8281, 0.7481, 0.1536],
    [0.8389, 0.7457, 0.1546],
    [0.8495, 0.7435, 0.1564],
    [0.8600, 0.7413, 0.1587],
    [0.8703, 0.7392, 0.1615],
    [0.8804, 0.7372, 0.1650],
    [0.8903, 0.7353, 0.1695],
    [0.9000, 0.7336, 0.1749],
    [0.9093, 0.7321, 0.1815],
    [0.9184, 0.7308, 0.1890],
    [0.9272, 0.7298, 0.1973],
    [0.9357, 0.7290, 0.2061],
    [0.9440, 0.7285, 0.2151],
    [0.9523, 0.7284, 0.2237],
    [0.9606, 0.7285, 0.2312],
    [0.9689, 0.7292, 0.2373],
    [0.9770, 0.7304, 0.2418],
    [0.9842, 0.7330, 0.2446],
    [0.9900, 0.7365, 0.2429],
    [0.9946, 0.7407, 0.2394],
    [0.9966, 0.7458, 0.2351],
    [0.9971, 0.7513, 0.2309],
    [0.9972, 0.7569, 0.2267],
    [0.9971, 0.7626, 0.2224],
    [0.9969, 0.7683, 0.2181],
    [0.9966, 0.7740, 0.2138],
    [0.9962, 0.7798, 0.2095],
    [0.9957, 0.7856, 0.2053],
    [0.9949, 0.7915, 0.2012],
    [0.9938, 0.7974, 0.1974],
    [0.9923, 0.8034, 0.1939],
    [0.9906, 0.8095, 0.1906],
    [0.9885, 0.8156, 0.1875],
    [0.9861, 0.8218, 0.1846],
    [0.9835, 0.8280, 0.1817],
    [0.9807, 0.8342, 0.1787],
    [0.9778, 0.8404, 0.1757],
    [0.9748, 0.8467, 0.1726],
    [0.9720, 0.8529, 0.1695],
    [0.9694, 0.8591, 0.1665],
    [0.9671, 0.8654, 0.1636],
    [0.9651, 0.8716, 0.1608],
    [0.9634, 0.8778, 0.1582],
    [0.9619, 0.8840, 0.1557],
    [0.9608, 0.8902, 0.1532],
    [0.9601, 0.8963, 0.1507],
    [0.9596, 0.9023, 0.1480],
    [0.9595, 0.9084, 0.1450],
    [0.9597, 0.9143, 0.1418],
    [0.9601, 0.9203, 0.1382],
    [0.9608, 0.9262, 0.1344],
    [0.9618, 0.9320, 0.1304],
    [0.9629, 0.9379, 0.1261],
    [0.9642, 0.9437, 0.1216],
    [0.9657, 0.9494, 0.1168],
    [0.9674, 0.9552, 0.1116],
    [0.9692, 0.9609, 0.1061],
    [0.9711, 0.9667, 0.1001],
    [0.9730, 0.9724, 0.0938],
    [0.9749, 0.9782, 0.0872],
    [0.9769, 0.9839, 0.0805],
]

cmap_parula = LinearSegmentedColormap.from_list("parula", cm_data)
cmap_parula_r = LinearSegmentedColormap.from_list("parula_r", cm_data[::-1])

# Default Matplotlib rcParams for consistent plotting style
default_rc = {
    "figure.dpi": 150,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.prop_cycle": plt.cycler(
        "color",
        [
            "#000000",
            "#DC143C",
            "#00BFFF",
            "#FFD700",
            "#32CD32",
            "#FF69B4",
            "#FF4500",
            "#1E90FF",
            "#8A2BE2",
            "#FFA07A",
            "#8B0000",
        ],
    ),
}
plt.rcParams.update(default_rc)  # Apply default styles

legend_params = {
    "loc": "best",
    "fontsize": 8,
    "frameon": True,
}


def apply_legend(ax: Union[Axes, List[Axes]]) -> Legend:
    """Applies a consistent legend style to a Matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to which the legend will be applied.

    Returns
    -------
    matplotlib.legend.Legend
        The created Legend object.
    """
    if isinstance(ax, list):
        ax = ax[-1]

    legend = ax.legend(**legend_params)
    if legend:  # Check if a legend was actually created (e.g., if labels exist)
        frame = legend.get_frame()
        frame.set_alpha(1.0)
        frame.set_edgecolor("black")
        frame.set_linewidth(0.7)
        try:
            frame.set_boxstyle("Square")
        except AttributeError:
            pass  # Safe fallback for older matplotlib
    return legend


def figsize(scale: float) -> List[float]:
    """Calculates figure size in inches for publication.

    Based on LaTeX's `\the\textwidth` (typically 390pt for a single column).

    Parameters
    ----------
    scale : float
        Scaling factor relative to the standard text width.

    Returns
    -------
    list[float]
        A list `[width, height]` in inches, maintaining the golden ratio.
    """
    fig_width_pt = 390
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    return [fig_width, fig_height]


def lin_plot(
    x: Optional[np.ndarray],
    y: np.ndarray,
    ax: Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Axes:
    """Creates a linear plot on a given Axes object.

    Parameters
    ----------
    x : np.ndarray, optional
        The x-axis data. If None, y is plotted against its index.
    y : np.ndarray
        The y-axis data.
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    title : str
        The title for the plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    label : str
        The label for the plotted line, used in the legend.
    *args, **kwargs :
        Additional arguments passed directly to `matplotlib.pyplot.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot.
    """
    if x is not None:
        ax.plot(x, y, label=label, *args, **kwargs)
    else:
        ax.plot(y, label=label, *args, **kwargs)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return ax


def log_plot(
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Axes:
    """Creates a log-log plot on a given Axes object.

    Parameters
    ----------
    x : np.ndarray
        The x-axis data.
    y : np.ndarray
        The y-axis data.
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    title : str
        The title for the plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    label : str
        The label for the plotted line, used in the legend.
    *args, **kwargs :
        Additional arguments passed directly to `matplotlib.pyplot.loglog`.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot.
    """
    ax.loglog(x, y, label=label, *args, **kwargs)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return ax


def stem_plot(
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> Axes:
    """Creates a stem plot (vertical lines from baseline) on a given Axes object.

    Parameters
    ----------
    x : np.ndarray
        The x-axis data for the stems.
    y : np.ndarray
        The y-axis data for the height of the stems.
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    title : str
        The title for the plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    label : str
        The label for the plotted stems, used in the legend.
    *args, **kwargs :
        Additional arguments passed directly to `matplotlib.pyplot.vlines`.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot.
    """
    ax.vlines(x, 0, y, color="b", label=label, *args, **kwargs)
    # Ensure y-limits accommodate both positive and negative stems
    min_val = np.min(y) if y.size > 0 else 0
    max_val = np.max(y) if y.size > 0 else 0
    ax.set_ylim(
        [min_val * 1.05 if min_val < 0 else 0, max_val * 1.05 if max_val > 0 else 0]
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return ax


def autoscale_y(ax: Axes, margin: float = 0.1) -> None:
    """Rescales the y-axis of a plot based on the currently visible x-data.

    This function adjusts the y-limits of the specified Axes object to fit
    only the data that is within the current x-axis limits, adding a margin.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to autoscale.
    margin : float, optional
        The fraction of the total height of the y-data to pad the upper and
        lower y-limits. Defaults to 0.1.
    """

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        # Find data within current x-limits
        y_displayed = yd[((xd >= lo) & (xd <= hi))]  # Use >= and <= for inclusive range
        if y_displayed.size == 0:
            return np.inf, -np.inf  # No data in view, return sentinel values

        y_min = np.min(y_displayed)
        y_max = np.max(y_displayed)
        h = y_max - y_min
        bot = y_min - margin * h
        top = y_max + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot:
            bot = new_bot
        if new_top > top:
            top = new_top

    if np.isinf(bot) and np.isinf(top):  # Handle case where no data is visible
        return  # Do not change limits if no data is present

    ax.set_ylim(bot, top)


def time_plot(
    t_list: List[Union[pd.Series, np.ndarray, List[float]]],
    y_list: List[Union[pd.DataFrame, np.ndarray, List[float]]],
    label_list: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    xrange: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
    remove_y_offsets: bool = False,
    remove_time_offsets: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Axes:
    """Plots multiple time series on a single Axes object.

    Parameters
    ----------
    t_list : list of (pd.Series or np.ndarray or list of float)
        List of time values for each series.
    y_list : list of (pd.DataFrame or np.ndarray or list of float)
        List of y-values for each series. If pd.DataFrame, assumes single column.
    label_list : list of str, optional
        List of labels for each series, used in the legend. If None, no labels.
    ax : matplotlib.axes.Axes, optional
        An existing Axes object to plot on. If None, a new figure and axes are created.
    xrange : tuple of (float, float), optional
        A tuple (xmin, xmax) specifying the time range to display.
    title : str, optional
        The title for the plot.
    y_label : str, optional
        The label for the y-axis.
    figsize : tuple of (float, float), optional
        The size of the figure to create if `ax` is not provided.
    remove_y_offsets : bool, optional
        If True, the mean of each y-series is subtracted before plotting.
        Defaults to False.
    remove_time_offsets : bool, optional
        If True, the start time of each series is set to 0. Defaults to False.
    *args, **kwargs :
        Additional arguments passed directly to `matplotlib.pyplot.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.
    """
    if len(t_list) != len(y_list):
        raise ValueError("Length of `y_list` must be equal to `t_list`.")
    if label_list is not None and len(t_list) != len(label_list):
        raise ValueError("Length of `label_list` must be equal to `t_list`.")

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_xlabel("Time (s)")
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)

    for i in range(len(t_list)):
        current_y = np.asarray(y_list[i]).flatten()  # Ensure numpy array and flatten
        current_t = np.asarray(t_list[i]).flatten()  # Ensure numpy array and flatten

        if remove_y_offsets:
            current_y = current_y - np.average(current_y)

        if remove_time_offsets:
            if current_t.size > 0:
                current_t = current_t - current_t[0]

        plot_label = label_list[i] if label_list is not None else None
        ax.plot(current_t, current_y, label=plot_label, *args, **kwargs)

    if xrange is not None:
        ax.set_xlim(xrange)
        autoscale_y(ax)  # Uses the autoscale_y helper

    _, labels = ax.get_legend_handles_labels()
    if any(label.strip() for label in labels):  # Only activate legend if labels exist
        ax.legend()
    fig.tight_layout()
    return ax


def time_histogram(
    df: pd.DataFrame,
    time_key: str,
    y_key: str,
    time_floor: str = "h",
    nbins: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    format_str: str = "%d/%m/%Y-%H:%M",
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[Figure, Axes]:
    """Plots a 2D histogram of time series data.

    The width of the time bins is adjusted by the `time_floor` parameter,
    which accepts pandas frequency strings (e.g., 'h' for 1 hour, '6h' for
    6 hours, '10t' for 10 minutes, 'd' for 1 day).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time series data.
    time_key : str
        Name of the column containing the timestamp (x-axis data).
    y_key : str
        Name of the column containing the time series data (y-axis data).
    time_floor : str, optional
        Width of the time bins using pandas frequency strings. Defaults to 'h'.
    nbins : int, optional
        Number of bins for the y-axis. If None, the number of bins is taken
        as the number of unique values of `df[y_key]`. Defaults to None.
    start : str, optional
        A start date for filtering the data (e.g., 'YYYY-MM-DD'). Defaults to None.
    end : str, optional
        An end date for filtering the data (e.g., 'YYYY-MM-DD'). Defaults to None.
    format_str : str, optional
        Format string for the time axis labels. Defaults to '%d/%m/%Y-%H:%M'.
    figsize : tuple of (float, float), optional
        Figure size in inches. Defaults to (12,5).

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the Figure and Axes objects of the plot.

    Raises
    ------
    ValueError
        If the timestamp column cannot be converted to datetime.
        If the filtered time axis is null.
    """
    try:
        datetime_series = pd.to_datetime(df[time_key])
    except Exception as e:
        raise ValueError(
            f"The timestamp column '{time_key}' cannot be converted to datetime: {e}"
        )

    dates_of_interest = pd.Series(True, index=datetime_series.index)

    if start is not None:
        dates_of_interest &= datetime_series >= pd.to_datetime(start)
    if end is not None:
        dates_of_interest &= datetime_series <= pd.to_datetime(end)

    filtered_df = df[dates_of_interest]
    if filtered_df.empty:
        raise ValueError("No data found within the specified time range.")

    time_axis_floored = datetime_series[dates_of_interest].dt.floor(time_floor)

    if time_axis_floored.empty:
        raise ValueError(
            "Null time axis after flooring. Check `time_floor` or data range."
        )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    cmap = copy.copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))  # Set color for cells with no data

    if nbins is None:
        unique_y_values = filtered_df[y_key].unique()
        if unique_y_values.size == 0:
            raise ValueError(
                f"No unique values for '{y_key}' in the filtered data to determine bins."
            )
        nbins = unique_y_values.size

    h, xedges, yedges = np.histogram2d(
        mdates.date2num(
            time_axis_floored.values
        ),  # Convert datetime to float for histogram2d
        filtered_df[y_key],
        bins=[time_axis_floored.unique().shape[0], nbins],
    )

    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(), rasterized=True)

    fig.colorbar(pcm, ax=ax, label="Number of events", pad=0)

    formatter = mdates.DateFormatter(format_str)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Time")
    ax.set_ylabel(y_key)
    ax.set_title(f"2D Histogram of {y_key} vs. Time")

    return fig, ax


def asd_plot(
    f_list: List[np.ndarray],
    asd_list: List[np.ndarray],
    label_list: Optional[List[str]] = None,
    title: Optional[str] = None,
    unit: Optional[str] = None,
    psd: bool = False,
) -> Axes:
    """Plots Amplitude Spectral Densities (ASDs) or Power Spectral Densities (PSDs).

    Plots one or more spectral density curves on a log-log scale.

    Parameters
    ----------
    f_list : list of np.ndarray
        A list of frequency arrays (Hz) for each spectrum.
    asd_list : list of np.ndarray
        A list of ASD arrays for each spectrum. If `psd` is True, these are
        interpreted as PSDs and converted to ASDs for plotting.
    label_list : list of str, optional
        A list of labels for each spectrum, used in the legend. If None,
        labels are auto-generated.
    title : str, optional
        The title for the plot.
    unit : str, optional
        The unit of the signal (e.g., 'V', 'm', 'Hz'). Used to construct the
        y-axis label. Defaults to 'A' (Amperes).
    psd : bool, optional
        If True, the `asd_list` values are treated as PSDs and their square
        root is taken to plot as ASDs. Defaults to False.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.
    """
    if label_list is None:
        label_list = [f"Spectrum {i + 1}" for i in range(len(f_list))]
    elif len(f_list) != len(label_list):
        raise ValueError("Length of `f_list` must match `label_list`.")

    if title is None:
        title = "Amplitude Spectral Density"

    if unit is None:
        unit = "A"

    xlabel = r"Frequency$\,({\mathrm{Hz}})$"
    ylabel = r"ASD$\,(\mathrm{" + unit + r"}/\sqrt{\mathrm{Hz}})$"

    fig, ax = plt.subplots(
        1, figsize=figsize(1.2), dpi=300
    )  # Use 300 DPI for high quality

    for i in range(len(asd_list)):
        current_asd = np.sqrt(asd_list[i]) if psd else asd_list[i]
        log_plot(f_list[i], current_asd, ax, title, xlabel, ylabel, label_list[i])

    apply_legend(ax)

    fig.tight_layout()
    return ax
