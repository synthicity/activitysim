# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd
from pydantic import validator

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

logger = logging.getLogger(__name__)


class FreeParkingSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `free_parking` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    FREE_PARKING_ALT: int
    """The code for free parking."""


@workflow.step
def free_parking(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: FreeParkingSettings | None = None,
    model_settings_file_name: str = "free_parking.yaml",
    trace_label: str = "free_parking",
) -> None:
    """
    Determine for each person whether they have free parking available at work.

    Parameters
    ----------
    state : workflow.State
    persons_merged : DataFrame
        This represents the 'choosers' table for this component.
    persons : DataFrame
        The original persons table is referenced so the free parking column
        can be appended to it.
    model_settings : FreeParkingSettings, optional
        The settings used in this model component.  If not provided, they are
        loaded out of the configs directory YAML file referenced by
        the `model_settings_file_name` argument.
    model_settings_file_name : str, default "free_parking.yaml"
        This is where model setting are found if `model_settings` is not given
        explicitly.  The same filename is also used to write settings files to
        the estimation data bundle in estimation mode.
    trace_label : str, default "free_parking"
        This label is used for various tracing purposes.
    """
    if model_settings is None:
        model_settings = FreeParkingSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    choosers = pd.DataFrame(persons_merged)
    choosers = choosers[choosers.workplace_zone_id > -1]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    estimator = estimation.manager.begin_estimation(state, "free_parking")

    constants = model_settings.CONSTANTS or {}

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
        estimator.write_spec(file_name=model_settings.SPEC)
        estimator.write_coefficients(
            coefficients_df, file_name=model_settings.COEFFICIENTS
        )
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="free_parking_at_work",
        estimator=estimator,
        overflow_protection=model_settings.overflow_protection,
    )

    free_parking_alt = model_settings.FREE_PARKING_ALT
    choices = choices == free_parking_alt

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "persons", "free_parking_at_work"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["free_parking_at_work"] = (
        choices.reindex(persons.index).fillna(0).astype(bool)
    )

    state.add_table("persons", persons)

    tracing.print_summary(
        "free_parking", persons.free_parking_at_work, value_counts=True
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)
