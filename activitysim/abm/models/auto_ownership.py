# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd
from pydantic import validator

from activitysim.core import (
    config,
    expressions,
    estimation,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.configuration.logit import LogitComponentSettings
from .util import annotate

logger = logging.getLogger(__name__)


class AutoOwnershipSettings(LogitComponentSettings):
    """
    Settings for the `auto_ownership` component.
    """

    preprocessor: PreprocessorSettings | None = None
    annotate_households: PreprocessorSettings | None = None


@workflow.step
def auto_ownership_simulate(
    state: workflow.State,
    households: pd.DataFrame,
    households_merged: pd.DataFrame,
    model_settings: AutoOwnershipSettings | None = None,
    model_settings_file_name: str = "auto_ownership.yaml",
    trace_label: str = "auto_ownership_simulate",
    trace_hh_id: bool = False,
) -> None:
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """

    if model_settings is None:
        model_settings = AutoOwnershipSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    estimator = estimation.manager.begin_estimation(state, "auto_ownership")
    model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    choosers = households_merged

    logger.info("Running %s with %d households", trace_label, len(choosers))

    # - preprocessor
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    log_alt_losers = state.settings.log_alt_losers

    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="auto_ownership",
        log_alt_losers=log_alt_losers,
        estimator=estimator,
        overflow_protection=model_settings.overflow_protection,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "households", "auto_ownership")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # no need to reindex as we used all households
    households["auto_ownership"] = choices

    state.add_table("households", households)

    tracing.print_summary(
        "auto_ownership", households.auto_ownership, value_counts=True
    )

    if model_settings.annotate_households:
        annotate.annotate_households(model_settings, trace_label)

    if trace_hh_id:
        state.tracing.trace_df(households, label="auto_ownership", warn_if_empty=True)
