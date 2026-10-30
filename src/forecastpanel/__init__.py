"""forecastpanel: forecasting with panel data.

Individual, pooled, fixed- and random-effects, empirical Bayes and optimally
combined panel forecasts of Pesaran, Pick and Timmermann (2026), Quantitative
Economics 17(2), 342-393, doi:10.3982/QE2589, with evaluation tables, panel
Diebold-Mariano tests, Monte Carlo designs and multi-step scenario forecasts.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane/forecastpanel
"""
from ._version import __version__
from ._colors import (bluered_colors, matlab_jet_colors, parula_colors, resolve_cmap,
                      resolve_colorscale, sinha_colors, turbo_colors)
from ._core import (empirical_bayes, fixed_effects, half_jackknife, hierarchical_bayes,
                    individual_ols, iterative_bayes, mean_group, oracle_weights, pooled_ols,
                    random_effects_ar1_blup, random_effects_blup, weight_individual_fe,
                    weight_individual_pooled, weight_mallows, weight_stein, weights_unit_specific)
from ._dm import DMResult, dm_test, panel_dm_test
from ._engine import (ALL_METHODS, DEFAULT_METHODS, METHODS, PAPER_METHODS, PanelFit,
                      panel_forecasts)
from ._evaluation import EvaluationResults, evaluate_forecasts
from ._forecaster import ForecastResults, PanelForecaster, forecast_panel
from ._panel import PanelData, wide_to_long
from ._pca import principal_components
from ._plots import JOURNAL_RC, save_figure
from ._simulate import (DGPParameters, delta_ar_numerical, draw_parameters, monte_carlo,
                        panel_pr2, simulate_panel, table1_designs)
from .datasets import (load_cpi_panel, load_cpi_subindices, load_house_prices,
                       load_house_weights, load_macro_predictors)

__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

__all__ = [
    # high-level
    "PanelForecaster", "ForecastResults", "forecast_panel", "PanelData", "wide_to_long",
    "evaluate_forecasts", "EvaluationResults", "panel_forecasts", "PanelFit",
    "METHODS", "PAPER_METHODS", "DEFAULT_METHODS", "ALL_METHODS",
    # estimators and weights
    "individual_ols", "pooled_ols", "fixed_effects", "random_effects_blup",
    "random_effects_ar1_blup", "empirical_bayes", "iterative_bayes", "mean_group",
    "hierarchical_bayes", "half_jackknife", "weight_individual_pooled", "weight_individual_fe",
    "weights_unit_specific", "weight_mallows", "weight_stein", "oracle_weights",
    # tests
    "dm_test", "panel_dm_test", "DMResult",
    # simulation
    "monte_carlo", "draw_parameters", "simulate_panel", "table1_designs", "panel_pr2",
    "delta_ar_numerical", "DGPParameters", "principal_components",
    # data
    "load_house_prices", "load_house_weights", "load_cpi_subindices", "load_macro_predictors",
    "load_cpi_panel",
    # graphics
    "parula_colors", "matlab_jet_colors", "turbo_colors", "bluered_colors", "sinha_colors",
    "resolve_colorscale", "resolve_cmap", "JOURNAL_RC", "save_figure",
]
