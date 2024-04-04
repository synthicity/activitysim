# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.configuration.logit import LogitComponentSettings

logger = logging.getLogger("activitysim")


class TransitPassSubsidySettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `transit_pass_subsidy` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""


@workflow.step
def transit_pass_subsidy(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: TransitPassSubsidySettings | None = None,
    model_settings_file_name: str = "transit_pass_subsidy.yaml",
    trace_label: str = "transit_pass_subsidy",
) -> None:
    """
    Transit pass subsidy model.
    """
    if model_settings is None:
        model_settings = TransitPassSubsidySettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    choosers = persons_merged
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    estimator = estimation.manager.begin_estimation(state, "transit_pass_subsidy")

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            state,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    model_spec = state.filesystem.read_model_spec(model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="transit_pass_subsidy",
        estimator=estimator,
        overflow_protection=model_settings.overflow_protection,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "persons", "transit_pass_subsidy"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["transit_pass_subsidy"] = choices.reindex(persons.index)

    state.add_table("persons", persons)

    tracing.print_summary(
        "transit_pass_subsidy", persons.transit_pass_subsidy, value_counts=True
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)
