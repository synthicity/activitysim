# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
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


class WorkFromHomeSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `work_from_home` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    WORK_FROM_HOME_ALT: int
    """Value that specify if the person is working from home"""  # TODO

    WORK_FROM_HOME_ITERATIONS: int = 1
    """Setting to specify the number of iterations."""

    CHOOSER_FILTER_COLUMN_NAME: str = "is_worker"
    """Column name in the dataframe to represent worker."""

    WORK_FROM_HOME_CHOOSER_FILTER: str = None
    """Setting to filter work from home chooser."""

    WORK_FROM_HOME_COEFFICIENT_CONSTANT: float = None
    """Setting to set the work from home coefficient."""

    WORK_FROM_HOME_TARGET_PERCENT: float = None
    """Setting to set work from target percent."""

    WORK_FROM_HOME_TARGET_PERCENT_TOLERANCE: float = None
    """Setting to set work from home target percent tolerance."""

    sharrow_skip: bool = False
    """Setting to skip sharrow."""

    DEST_CHOICE_COLUMN_NAME: str = "workplace_zone_id"
    """Column name in persons dataframe to specify the workplace zone id. """

    SPEC: str = "work_from_home.csv"
    """Filename for the accessibility specification (csv) file."""


@workflow.step
def work_from_home(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: WorkFromHomeSettings | None = None,
    model_settings_file_name: str = "work_from_home.yaml",
    trace_label: str = "work_from_home",
) -> None:
    """
    This model predicts whether a person (worker) works from home. The output
    from this model is TRUE (if works from home) or FALSE (works away from home).
    The workplace location choice is overridden for workers who work from home
    and set to -1.
    """
    if model_settings is None:
        model_settings = WorkFromHomeSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    choosers = persons_merged
    chooser_filter_column_name = model_settings.CHOOSER_FILTER_COLUMN_NAME
    choosers = choosers[choosers[chooser_filter_column_name]]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    estimator = estimation.manager.begin_estimation(state, "work_from_home")

    constants = config.get_model_constants(model_settings)
    work_from_home_alt = model_settings.WORK_FROM_HOME_ALT

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

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    # - iterative single process what-if adjustment if specified
    iterations = model_settings.WORK_FROM_HOME_ITERATIONS
    iterations_chooser_filter = model_settings.WORK_FROM_HOME_CHOOSER_FILTER
    iterations_coefficient_constant = model_settings.WORK_FROM_HOME_COEFFICIENT_CONSTANT
    iterations_target_percent = model_settings.WORK_FROM_HOME_TARGET_PERCENT
    iterations_target_percent_tolerance = (
        model_settings.WORK_FROM_HOME_TARGET_PERCENT_TOLERANCE
    )

    for iteration in range(iterations):
        logger.info(
            "Running %s with %d persons iteration %d",
            trace_label,
            len(choosers),
            iteration,
        )

        # re-read spec to reset substitution
        model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
        model_spec = simulate.eval_coefficients(
            state, model_spec, coefficients_df, estimator
        )

        if model_settings.sharrow_skip:
            constants["disable_sharrow"] = True

        choices = simulate.simple_simulate(
            state,
            choosers=choosers,
            spec=model_spec,
            nest_spec=nest_spec,
            locals_d=constants,
            trace_label=trace_label,
            trace_choice_name="work_from_home",
            estimator=estimator,
            overflow_protection=model_settings.overflow_protection,
        )

        if iterations_target_percent is not None:
            choices_for_filter = choices[choosers[iterations_chooser_filter]]

            current_percent = (choices_for_filter == work_from_home_alt).sum() / len(
                choices_for_filter
            )
            logger.info(
                "Running %s iteration %i choosers %i current percent %f target percent %f",
                trace_label,
                iteration,
                len(choices_for_filter),
                current_percent,
                iterations_target_percent,
            )

            if current_percent <= (
                iterations_target_percent + iterations_target_percent_tolerance
            ) and current_percent >= (
                iterations_target_percent - iterations_target_percent_tolerance
            ):
                logger.info(
                    "Running %s iteration %i converged with coefficient %f",
                    trace_label,
                    iteration,
                    coefficients_df.value[iterations_coefficient_constant],
                )
                break

            else:
                new_value = (
                    np.log(
                        iterations_target_percent / np.maximum(current_percent, 0.0001)
                    )
                    + coefficients_df.value[iterations_coefficient_constant]
                )
                coefficients_df.value[iterations_coefficient_constant] = new_value
                logger.info(
                    "Running %s iteration %i new coefficient for next iteration %f",
                    trace_label,
                    iteration,
                    new_value,
                )
                iteration = iteration + 1

    choices = choices == work_from_home_alt

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "persons", "work_from_home")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["work_from_home"] = choices.reindex(persons.index).fillna(0).astype(bool)
    persons["is_out_of_home_worker"] = (
        persons[chooser_filter_column_name] & ~persons["work_from_home"]
    )

    # setting workplace_zone_id to -1 if person works from home
    # this will exclude them from the telecommute frequency model choosers
    # See https://github.com/ActivitySim/activitysim/issues/627
    dest_choice_column_name = model_settings.DEST_CHOICE_COLUMN_NAME
    if dest_choice_column_name in persons.columns:
        persons[dest_choice_column_name] = np.where(
            persons.work_from_home == True, -1, persons[dest_choice_column_name]
        )

    state.add_table("persons", persons)

    tracing.print_summary("work_from_home", persons.work_from_home, value_counts=True)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)
