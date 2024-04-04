# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.models.util.overlap import hh_time_window_overlap
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

logger = logging.getLogger(__name__)


def add_null_results(state, trace_label, tours):
    logger.info("Skipping %s: add_null_results" % trace_label)
    tours["composition"] = ""
    cat_type = pd.api.types.CategoricalDtype(
        ["", "adults", "children", "mixed"], ordered=False
    )
    tours["composition"] = tours["composition"].astype(cat_type)
    state.add_table("tours", tours)


class JointTourCompositionSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `joint_tour_composition` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""


@workflow.step
def joint_tour_composition(
    state: workflow.State,
    tours: pd.DataFrame,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    model_settings: JointTourCompositionSettings | None = None,
    model_settings_file_name: str = "joint_tour_composition.yaml",
    trace_label: str = "joint_tour_composition",
) -> None:
    """
    This model predicts the makeup of the travel party (adults, children, or mixed).
    """

    joint_tours = tours[tours.tour_category == "joint"]

    if model_settings is None:
        model_settings = JointTourCompositionSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        add_null_results(state, trace_label, tours)
        return

    estimator = estimation.manager.begin_estimation(state, "joint_tour_composition")

    # - only interested in households with joint_tours
    households = households[households.num_hh_joint_tours > 0]

    persons = persons[persons.household_id.isin(households.index)]

    logger.info(
        "Running joint_tour_composition with %d joint tours" % joint_tours.shape[0]
    )

    # - run preprocessor
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        locals_dict = {
            "persons": persons,
            "hh_time_window_overlap": lambda *x: hh_time_window_overlap(state, *x),
        }

        expressions.assign_columns(
            state,
            df=households,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    joint_tours_merged = pd.merge(
        joint_tours, households, left_on="household_id", right_index=True, how="left"
    )

    # - simple_simulate
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
        estimator.write_choosers(joint_tours_merged)

    choices = simulate.simple_simulate(
        state,
        choosers=joint_tours_merged,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="composition",
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
        choices = estimator.get_survey_values(choices, "tours", "composition")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # add composition column to tours for tracing
    joint_tours["composition"] = choices

    # reindex since we ran model on a subset of households
    tours["composition"] = choices.reindex(tours.index).fillna("")
    state.add_table("tours", tours)

    tracing.print_summary(
        "joint_tour_composition", joint_tours.composition, value_counts=True
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(
            joint_tours,
            label="joint_tour_composition.joint_tours",
            slicer="household_id",
        )
