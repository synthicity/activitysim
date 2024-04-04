# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import warnings
from typing import Optional

import pandas as pd

from activitysim.core import config, expressions, simulate, workflow
from activitysim.core.configuration.logit import TourModeComponentSettings
from activitysim.core.estimation import Estimator

"""
At this time, these utilities are mostly for transforming the mode choice
spec, which is more complicated than the other specs, into something that
looks like the other specs.
"""

logger = logging.getLogger(__name__)


def mode_choice_simulate(
    state: workflow.State,
    choosers: pd.DataFrame,
    spec: pd.DataFrame,
    nest_spec,
    skims,
    locals_d,
    mode_column_name,
    logsum_column_name,
    trace_label: str,
    trace_choice_name,
    trace_column_names=None,
    estimator: Optional[Estimator] = None,
    overflow_protection: bool = False,
):
    """
    common method for  both tour_mode_choice and trip_mode_choice

    Parameters
    ----------
    choosers
    spec
    nest_spec
    skims
    locals_d
    chunk_size
    mode_column_name
    logsum_column_name
    trace_label
    trace_choice_name
    estimator

    Returns
    -------

    """
    want_logsums = logsum_column_name is not None

    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=spec,
        nest_spec=nest_spec,
        skims=skims,
        locals_d=locals_d,
        want_logsums=want_logsums,
        trace_label=trace_label,
        trace_choice_name=trace_choice_name,
        estimator=estimator,
        trace_column_names=trace_column_names,
        overflow_protection=overflow_protection,
    )

    # for consistency, always return dataframe, whether or not logsums were requested
    if isinstance(choices, pd.Series):
        choices = choices.to_frame("choice")

    choices.rename(
        columns={"logsum": logsum_column_name, "choice": mode_column_name}, inplace=True
    )

    alts = spec.columns
    choices[mode_column_name] = choices[mode_column_name].map(
        dict(list(zip(list(range(len(alts))), alts)))
    )
    cat_type = pd.api.types.CategoricalDtype([""] + alts.tolist(), ordered=True)
    choices[mode_column_name] = choices[mode_column_name].astype(cat_type)

    return choices


def run_tour_mode_choice_simulate(
    state: workflow.State,
    choosers,
    tour_purpose,
    model_settings: TourModeComponentSettings,
    mode_column_name,
    logsum_column_name,
    network_los,
    skims,
    constants,
    estimator,
    trace_label=None,
    trace_choice_name=None,
):
    """
    This is a utility to run a mode choice model for each segment (usually
    segments are tour/trip purposes).  Pass in the tours/trip that need a mode,
    the Skim object, the spec to evaluate with, and any additional expressions
    you want to use in the evaluation of variables.
    """

    spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients = state.filesystem.get_segment_coefficients(
        model_settings, tour_purpose
    )

    spec = simulate.eval_coefficients(state, spec, coefficients, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(skims)

    # coefficients can appear in expressions
    locals_keys = locals_dict.keys()
    if any([coeff in locals_keys for coeff in coefficients.keys()]):
        logger.warning("coefficients are obscuring locals_dict values")
    locals_dict.update(coefficients)

    assert ("in_period" not in choosers) and ("out_period" not in choosers)
    in_time = skims["in_time_col_name"]
    out_time = skims["out_time_col_name"]
    choosers["in_period"] = network_los.skim_time_period_label(
        choosers[in_time], as_cat=True
    )
    choosers["out_period"] = network_los.skim_time_period_label(
        choosers[out_time], as_cat=True
    )

    expressions.annotate_preprocessors(
        state, choosers, locals_dict, skims, model_settings, trace_label
    )

    trace_column_names = choosers.index.name
    if trace_column_names != "tour_id":
        # TODO suppress this warning?  It should not be relevant in regular
        #      activitysim models, but could be annoying in extensions.
        warnings.warn(f"trace_column_names is {trace_column_names!r} not 'tour_id'")
    if trace_column_names not in choosers:
        choosers[trace_column_names] = choosers.index

    if estimator:
        # write choosers after annotation
        estimator.write_choosers(choosers)

    choices = mode_choice_simulate(
        state,
        choosers=choosers,
        spec=spec,
        nest_spec=nest_spec,
        skims=skims,
        locals_d=locals_dict,
        mode_column_name=mode_column_name,
        logsum_column_name=logsum_column_name,
        trace_label=trace_label,
        trace_choice_name=trace_choice_name,
        trace_column_names=trace_column_names,
        estimator=estimator,
        overflow_protection=model_settings.overflow_protection,
    )

    return choices
