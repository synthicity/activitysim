# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from activitysim.abm.models.util.tour_frequency import process_mandatory_tours
from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.configuration.logit import (
    BaseLogitComponentSettings,
    LogitComponentSettings,
    PreprocessorSettings,
)

logger = logging.getLogger(__name__)


def add_null_results(state, trace_label, mandatory_tour_frequency_settings):
    logger.info("Skipping %s: add_null_results", trace_label)

    persons = state.get_dataframe("persons")
    persons["mandatory_tour_frequency"] = pd.categorical(
        "",
        categories=["", "work1", "work2", "school1", "school2", "work_and_school"],
        ordered=False,
    )

    tours = pd.DataFrame()
    tours["tour_category"] = None
    tours["tour_type"] = None
    tours["person_id"] = None
    tours.index.name = "tour_id"
    state.add_table("tours", tours)

    expressions.assign_columns(
        state,
        df=persons,
        model_settings=mandatory_tour_frequency_settings.get("annotate_persons"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
    )

    state.add_table("persons", persons)


class MandatoryTourFrequencySettings(LogitComponentSettings):
    """
    Settings for the `mandatory_tour_frequency` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    annotate_persons: PreprocessorSettings | None = None


@workflow.step
def mandatory_tour_frequency(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    model_settings: MandatoryTourFrequencySettings | None = None,
    model_settings_file_name: str = "mandatory_tour_frequency.yaml",
    trace_label: str = "mandatory_tour_frequency",
) -> None:
    """
    This model predicts the frequency of making mandatory trips (see the
    alternatives above) - these trips include work and school in some combination.
    """

    trace_hh_id = state.settings.trace_hh_id

    if model_settings is None:
        model_settings = MandatoryTourFrequencySettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    choosers = persons_merged
    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity == "M"]
    logger.info("Running mandatory_tour_frequency with %d persons", len(choosers))

    # - if no mandatory tours
    if choosers.shape[0] == 0:
        add_null_results(state, trace_label, model_settings)
        return

    # - preprocessor
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        locals_dict = {}

        expressions.assign_columns(
            state,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    estimator = estimation.manager.begin_estimation(state, "mandatory_tour_frequency")

    model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    if estimator:
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="mandatory_tour_frequency",
        estimator=estimator,
        overflow_protection=model_settings.overflow_protection,
    )

    # convert indexes to alternative names
    choices = pd.Series(model_spec.columns[choices.values], index=choices.index)
    cat_type = pd.api.types.CategoricalDtype(
        model_spec.columns.tolist() + [""], ordered=False
    )
    choices = choices.astype(cat_type)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "persons", "mandatory_tour_frequency"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # - create mandatory tours
    """
    This reprocesses the choice of index of the mandatory tour frequency
    alternatives into an actual dataframe of tours.  Ending format is
    the same as got non_mandatory_tours except trip types are "work" and "school"
    """
    alternatives = simulate.read_model_alts(
        state, "mandatory_tour_frequency_alternatives.csv", set_index="alt"
    )
    choosers["mandatory_tour_frequency"] = choices.reindex(choosers.index)

    mandatory_tours = process_mandatory_tours(
        state, persons=choosers, mandatory_tour_frequency_alts=alternatives
    )

    # convert purpose to pandas categoricals
    purpose_type = pd.api.types.CategoricalDtype(
        alternatives.columns.tolist() + ["univ", "home", "escort"], ordered=False
    )
    mandatory_tours["tour_type"] = mandatory_tours["tour_type"].astype(purpose_type)

    tours = state.extend_table("tours", mandatory_tours)
    state.tracing.register_traceable_table("tours", mandatory_tours)
    state.get_rn_generator().add_channel("tours", mandatory_tours)

    # - annotate persons
    persons = state.get_dataframe("persons")

    # need to reindex as we only handled persons with cdap_activity == 'M'
    persons["mandatory_tour_frequency"] = choices.reindex(persons.index).fillna("")

    expressions.assign_columns(
        state,
        df=persons,
        model_settings=model_settings.annotate_persons,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
    )

    state.add_table("persons", persons)

    tracing.print_summary(
        "mandatory_tour_frequency", persons.mandatory_tour_frequency, value_counts=True
    )

    if trace_hh_id:
        state.tracing.trace_df(
            mandatory_tours,
            label="mandatory_tour_frequency.mandatory_tours",
            warn_if_empty=True,
        )

        state.tracing.trace_df(
            persons, label="mandatory_tour_frequency.persons", warn_if_empty=True
        )
