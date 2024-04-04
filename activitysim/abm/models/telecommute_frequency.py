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
from activitysim.core.configuration.base import PreprocessorSettings
from activitysim.core.configuration.logit import LogitComponentSettings

logger = logging.getLogger("activitysim")


class TelecommuteFrequencySettings(LogitComponentSettings):
    """
    Settings for the `telecommute_frequency` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""


@workflow.step
def telecommute_frequency(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: TelecommuteFrequencySettings | None = None,
    model_settings_file_name: str = "telecommute_frequency.yaml",
    trace_label: str = "telecommute_frequency",
) -> None:
    """
    This model predicts the frequency of telecommute for a person (worker) who
    does not works from home. The alternatives of this model are 'No Telecommute',
    '1 day per week', '2 to 3 days per week' and '4 days per week'. This model
    reflects the choices of people who prefer a combination of working from home and
    office during a week.
    """

    if model_settings is None:
        model_settings = TelecommuteFrequencySettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    choosers = persons_merged
    choosers = choosers[choosers.workplace_zone_id > -1]

    logger.info("Running %s with %d persons", trace_label, len(choosers))

    estimator = estimation.manager.begin_estimation(state, "telecommute_frequency")

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

    model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
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
        trace_choice_name="telecommute_frequency",
        estimator=estimator,
        overflow_protection=model_settings.overflow_protection,
    )

    choices = pd.Series(model_spec.columns[choices.values], index=choices.index)
    telecommute_frequency_cat = pd.api.types.CategoricalDtype(
        model_spec.columns.tolist() + [""],
        ordered=False,
    )
    choices = choices.astype(telecommute_frequency_cat)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "persons", "telecommute_frequency"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["telecommute_frequency"] = choices.reindex(persons.index).fillna("")

    state.add_table("persons", persons)

    tracing.print_summary(
        "telecommute_frequency", persons.telecommute_frequency, value_counts=True
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)
