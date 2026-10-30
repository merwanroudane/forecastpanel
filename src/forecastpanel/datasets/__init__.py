"""Real datasets of the two empirical applications of Pesaran, Pick and Timmermann (2026).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

__all__ = ["load_house_prices", "load_house_weights", "load_cpi_subindices",
           "load_macro_predictors", "load_cpi_panel"]


def _read(name: str, **kw) -> pd.DataFrame:
    with resources.files(__name__).joinpath(name).open("rb") as fh:
        return pd.read_csv(fh, compression="gzip", **kw)


def load_house_prices(with_neighbors_only: bool = False) -> pd.DataFrame:
    """Real house price inflation in 377 US metropolitan areas, 1976Q1-2023Q1.

    Long format, one row per (msa, quarter):

    ============  ==============================================================
    msa           CBSA code of the metropolitan statistical area
    msa_name      MSA name (Freddie Mac FMHPI)
    region        census region code (1-8) used for regional averages
    n_neighbors   number of MSAs within 100 miles (W100 > 0)
    quarter       '1976Q1' ... '2023Q1' (189 quarters)
    dhp           100 x 4-quarter log change of the FMHPI deflated by CPI-U (%)
    dhp_spatial   spatial average sum_k w_ik dhp_kt, w from W100 (Yang, 2021)
    dhp_region    average of dhp over the MSAs of the same region
    dhp_us        average of dhp over all 377 MSAs
    ============  ==============================================================

    The paper's application (Section 6.2) uses the 362 MSAs with at least one
    neighbour: pass ``with_neighbors_only=True``. Sources: Freddie Mac House
    Price Index (FMHPI), FRED CPIAUCSL, spatial weights of Yang (2021),
    processed exactly as ``ppt_house.m`` of the replication archive.

    Examples
    --------
    >>> from forecastpanel import load_house_prices
    >>> df = load_house_prices(with_neighbors_only=True)
    >>> df["msa"].nunique(), df["quarter"].nunique()
    (362, 189)
    """
    df = _read("house_prices.csv.gz")
    if with_neighbors_only:
        df = df[df["n_neighbors"] > 0].reset_index(drop=True)
    return df


def load_house_weights(as_matrix: bool = True):
    """Spatial weights W100 (row-normalised, MSAs within 100 miles), Yang (2021).

    Returns a (377 x 377) DataFrame indexed by MSA code, or the edge list.
    """
    edges = _read("house_weights.csv.gz")
    if not as_matrix:
        return edges
    codes = load_house_prices()["msa"].drop_duplicates().tolist()
    mat = edges.pivot(index="msa", columns="neighbor", values="weight").reindex(index=codes, columns=codes)
    return mat.fillna(0.0)


def load_cpi_subindices() -> pd.DataFrame:
    """Monthly inflation of 187 US CPI sub-indices, 1967M1-2023M5 (FRED/BLS).

    Long format with columns ``series`` (FRED code), ``month`` ('YYYY-MM') and
    ``infl`` = 100 x (P_t / P_{t-1} - 1). Series enter the panel when their
    index starts, so the panel is unbalanced in the early years.
    """
    return _read("cpi_subindices.csv.gz")


def load_macro_predictors() -> pd.DataFrame:
    """Default spread ``dfy`` and term spread ``tms`` (in %), monthly 1947M1-2022M12.

    Source: Welch and Goyal (2008) data, as in the replication archive.
    """
    return _read("macro_predictors.csv.gz")


def load_cpi_panel(start: str = "1967-01", end: str = "2022-12") -> pd.DataFrame:
    """CPI sub-index inflation merged with the macro predictors (long format)."""
    c = load_cpi_subindices()
    m = load_macro_predictors()
    df = c.merge(m, on="month", how="left")
    return df[(df["month"] >= start) & (df["month"] <= end)].reset_index(drop=True)
