"""Colour maps shared with the author's R packages.

``parula_colors`` reproduces MATLAB's R2014b Parula colormap from its 64 RGB
stops; ``"Parula"`` is the default colour scale of every heatmap in the package.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

PARULA_STOPS = np.array([
    [0.2422, 0.1504, 0.6603], [0.2504, 0.1650, 0.7076], [0.2578, 0.1818, 0.7511],
    [0.2647, 0.1978, 0.7952], [0.2706, 0.2147, 0.8364], [0.2751, 0.2342, 0.8710],
    [0.2783, 0.2559, 0.8991], [0.2803, 0.2782, 0.9221], [0.2813, 0.3006, 0.9414],
    [0.2810, 0.3228, 0.9579], [0.2795, 0.3447, 0.9717], [0.2760, 0.3667, 0.9829],
    [0.2699, 0.3892, 0.9906], [0.2602, 0.4123, 0.9952], [0.2440, 0.4358, 0.9988],
    [0.2206, 0.4603, 0.9973], [0.1963, 0.4847, 0.9892], [0.1834, 0.5074, 0.9798],
    [0.1786, 0.5289, 0.9682], [0.1764, 0.5499, 0.9520], [0.1687, 0.5703, 0.9359],
    [0.1540, 0.5902, 0.9218], [0.1460, 0.6091, 0.9079], [0.1380, 0.6276, 0.8973],
    [0.1248, 0.6459, 0.8883], [0.1113, 0.6635, 0.8763], [0.0952, 0.6798, 0.8598],
    [0.0689, 0.6948, 0.8394], [0.0297, 0.7082, 0.8163], [0.0036, 0.7203, 0.7917],
    [0.0067, 0.7312, 0.7660], [0.0433, 0.7411, 0.7394], [0.0964, 0.7500, 0.7120],
    [0.1408, 0.7584, 0.6842], [0.1717, 0.7670, 0.6554], [0.1938, 0.7758, 0.6251],
    [0.2161, 0.7843, 0.5923], [0.2470, 0.7918, 0.5567], [0.2906, 0.7973, 0.5188],
    [0.3406, 0.8008, 0.4789], [0.3909, 0.8029, 0.4354], [0.4456, 0.8024, 0.3909],
    [0.5044, 0.7993, 0.3480], [0.5616, 0.7942, 0.3045], [0.6174, 0.7876, 0.2612],
    [0.6720, 0.7793, 0.2227], [0.7242, 0.7698, 0.1910], [0.7738, 0.7598, 0.1646],
    [0.8203, 0.7498, 0.1535], [0.8634, 0.7406, 0.1596], [0.9035, 0.7330, 0.1774],
    [0.9393, 0.7288, 0.2100], [0.9728, 0.7298, 0.2394], [0.9956, 0.7434, 0.2371],
    [0.9970, 0.7659, 0.2199], [0.9952, 0.7893, 0.2028], [0.9892, 0.8129, 0.1885],
    [0.9786, 0.8360, 0.1766], [0.9676, 0.8587, 0.1643], [0.9610, 0.8806, 0.1537],
    [0.9597, 0.9023, 0.1423], [0.9628, 0.9234, 0.1330], [0.9691, 0.9438, 0.1241],
    [0.9769, 0.9839, 0.0805],
])

_JET = ["#000090", "#0000FF", "#00FFFF", "#FFFF00", "#FF0000", "#800000"]
_TURBO = ["#30123B", "#4145AB", "#4675ED", "#39A2FC", "#1BCFD4", "#24EAA7",
          "#61FC6C", "#A4FC3C", "#D1E834", "#F3C63A", "#FE9B2D", "#F36315",
          "#D93806", "#A91201", "#7A0403"]
_BLUERED = ["#053061", "#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B", "#67001F"]
_SINHA = ["#000000", "#4A0E00", "#B22222", "#FF8C00", "#FFFF66"]

# Line colours and styles for forecasting methods (colour is always paired with
# a line style so figures survive greyscale printing).
METHOD_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30",
                 "#4DBEEE", "#A2142F", "#000000", "#8C564B", "#E377C2",
                 "#17BECF", "#7F7F7F", "#BCBD22", "#1F77B4"]
METHOD_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)),
                     (0, (1, 1)), "-", "--", "-.", ":", (0, (5, 2)),
                     (0, (3, 2, 1, 2)), "-"]


def _hex(rgb: np.ndarray) -> List[str]:
    rgb = np.clip(np.round(np.asarray(rgb) * 255), 0, 255).astype(int)
    return ["#{:02X}{:02X}{:02X}".format(*c) for c in rgb]


def _ramp(hex_stops: Sequence[str], n: int) -> List[str]:
    """Linear interpolation between hex stops (as R's ``colorRampPalette``)."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    stops = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in hex_stops])
    if n == 1:
        return _hex(stops[:1])
    pos = np.linspace(0, 1, len(stops))
    grid = np.linspace(0, 1, n)
    out = np.column_stack([np.interp(grid, pos, stops[:, j]) for j in range(3)])
    return _hex(out)


def parula_colors(n: int = 256) -> List[str]:
    """MATLAB R2014b Parula colormap as ``n`` hex colours.

    Parameters
    ----------
    n : int, default 256
        Number of colours.

    Returns
    -------
    list of str

    Examples
    --------
    >>> from forecastpanel import parula_colors
    >>> parula_colors(3)
    ['#3E26A8', '#12BEBA', '#F9FB15']
    """
    hexes = _hex(PARULA_STOPS)
    if n == len(hexes):
        return hexes
    return _ramp(hexes, n)


def matlab_jet_colors(n: int = 256) -> List[str]:
    """MATLAB Jet colormap as ``n`` hex colours."""
    return _ramp(_JET, n)


def turbo_colors(n: int = 256) -> List[str]:
    """Google Turbo colormap as ``n`` hex colours."""
    return _ramp(_TURBO, n)


def bluered_colors(n: int = 256) -> List[str]:
    """Diverging blue-white-red colormap as ``n`` hex colours."""
    return _ramp(_BLUERED, n)


def sinha_colors(n: int = 256) -> List[str]:
    """Sinha black-red-yellow colormap as ``n`` hex colours."""
    return _ramp(_SINHA, n)


_NAMED = {"parula": parula_colors, "jet": matlab_jet_colors, "turbo": turbo_colors,
          "bluered": bluered_colors, "blue_red": bluered_colors,
          "sinha": sinha_colors, "red_yellow_black": sinha_colors}


def resolve_cmap(name: str = "Parula", n: int = 256):
    """Return a matplotlib colormap for a named scale (default ``"Parula"``)."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib

    key = str(name).lower()
    if key in _NAMED:
        return LinearSegmentedColormap.from_list(key, _NAMED[key](n), N=n)
    return matplotlib.colormaps[name]


def resolve_colorscale(name: str = "Parula", n_breaks: int = 32):
    """Return a plotly colorscale ``[[0.0, '#..'], ...]`` for a named scale.

    Native plotly names (``"Viridis"``, ``"RdBu"``, ...) are passed through.
    """
    if isinstance(name, (list, tuple)):
        return list(name)
    key = str(name).lower()
    if key in _NAMED:
        cols = _NAMED[key](256)
        n_breaks = max(2, int(n_breaks))
        idx = np.round(np.linspace(0, len(cols) - 1, n_breaks)).astype(int)
        vals = np.linspace(0, 1, n_breaks)
        return [[float(v), cols[i]] for v, i in zip(vals, idx)]
    return name.capitalize() if key.isalpha() else name
