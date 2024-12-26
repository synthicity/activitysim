# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from activitysim.abm.models.util import cdap
from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import (
    ComputeSettings,
    PreprocessorSettings,
    PydanticReadable,
)
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


class CdapSettings(PydanticReadable, extra="forbid"):
    PERSON_TYPE_MAP: dict[str, list[int]]
    INDIV_AND_HHSIZE1_SPEC: str
    INTERACTION_COEFFICIENTS: str = "cdap_interaction_coefficients.csv"
    FIXED_RELATIVE_PROPORTIONS_SPEC: str = "cdap_fixed_relative_proportions.csv"
    ADD_JOINT_TOUR_UTILITY: bool = False
    JOINT_TOUR_COEFFICIENTS: str = "cdap_joint_tour_coefficients.csv"
    JOINT_TOUR_USEFUL_COLUMNS: list[str] | None = None
    """Columns to include from the persons table that will be need to calculate household joint tour utility."""
    annotate_persons: PreprocessorSettings | None = None
    annotate_households: PreprocessorSettings | None = None
    COEFFICIENTS: Path
    CONSTANTS: dict[str, Any] = {}
    compute_settings: ComputeSettings | None = None


@workflow.step
def cdap_simulate(
    state: workflow.State,
    persons_merged: pd.DataFrame,
    persons: pd.DataFrame,
    households: pd.DataFrame,
    model_settings: CdapSettings | None = None,
    model_settings_file_name: str = "cdap.yaml",
    trace_label: str = "cdap",
) -> None:
    """
    CDAP stands for Coordinated Daily Activity Pattern, which is a choice of
    high-level activity pattern for each person, in a coordinated way with other
    members of a person's household.

    Because Python requires vectorization of computation, there are some specialized
    routines in the cdap directory of activitysim for this purpose.  This module
    simply applies those utilities using the simulation framework.
    """
    if model_settings is None:
        model_settings = CdapSettings.read_settings_file(
            state.filesystem, model_settings_file_name
        )
    trace_hh_id = state.settings.trace_hh_id
    person_type_map = model_settings.PERSON_TYPE_MAP
    estimator = estimation.manager.begin_estimation(state, "cdap")

    cdap_indiv_spec = state.filesystem.read_model_spec(
        file_name=model_settings.INDIV_AND_HHSIZE1_SPEC
    )

    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    cdap_indiv_spec = simulate.eval_coefficients(
        state, cdap_indiv_spec, coefficients_df, estimator
    )

    # Rules and coefficients for generating interaction specs for different household sizes
    interaction_coefficients_file_name = model_settings.INTERACTION_COEFFICIENTS
    cdap_interaction_coefficients = pd.read_csv(
        state.filesystem.get_config_file_path(interaction_coefficients_file_name),
        comment="#",
    )

    # replace cdap_interaction_coefficients coefficient labels with numeric values
    # for backward compatibility, use where() to allow hard-coded coefficients and dummy (empty) coefficients_file
    coefficients = cdap_interaction_coefficients.coefficient.map(
        coefficients_df.value.to_dict()
    )
    coefficients = cdap_interaction_coefficients.coefficient.where(
        coefficients.isnull(), coefficients
    )
    coefficients = pd.to_numeric(coefficients, errors="coerce").astype(float)
    if coefficients.isnull().any():
        # show them the offending lines from interaction_coefficients_file
        logger.warning(
            f"bad coefficients in INTERACTION_COEFFICIENTS {interaction_coefficients_file_name}\n"
            f"{cdap_interaction_coefficients[coefficients.isnull()]}"
        )
        assert not coefficients.isnull().any()
    cdap_interaction_coefficients.coefficient = coefficients

    """
    spec to compute/specify the relative proportions of each activity (M, N, H)
    that should be used to choose activities for additional household members not handled by CDAP
    This spec is handled much like an activitysim logit utility spec,
    EXCEPT that the values computed are relative proportions, not utilities
    (i.e. values are not exponentiated before being normalized to probabilities summing to 1.0)
    """
    cdap_fixed_relative_proportions = state.filesystem.read_model_spec(
        file_name=model_settings.FIXED_RELATIVE_PROPORTIONS_SPEC
    )

    add_joint_tour_utility = model_settings.ADD_JOINT_TOUR_UTILITY

    if add_joint_tour_utility:
        # Rules and coefficients for generating cdap joint tour specs for different household sizes
        joint_tour_coefficients_file_name = model_settings.JOINT_TOUR_COEFFICIENTS
        cdap_joint_tour_coefficients = pd.read_csv(
            state.filesystem.get_config_file_path(joint_tour_coefficients_file_name),
            comment="#",
        )

    # add tour-based chunk_id so we can chunk all trips in tour together
    assert "chunk_id" not in persons_merged.columns
    unique_household_ids = persons_merged.household_id.unique()
    household_chunk_ids = pd.Series(
        range(len(unique_household_ids)), index=unique_household_ids
    )
    persons_merged["chunk_id"] = reindex(
        household_chunk_ids, persons_merged.household_id
    )

    constants = config.get_model_constants(model_settings)

    cdap_interaction_coefficients = cdap.preprocess_interaction_coefficients(
        cdap_interaction_coefficients
    )

    # specs are built just-in-time on demand and cached as injectables
    # prebuilding here allows us to write them to the output directory
    # (also when multiprocessing locutor might not see all household sizes)
    logger.info("Pre-building cdap specs")
    for hhsize in range(2, cdap.MAX_HHSIZE + 1):
        spec = cdap.build_cdap_spec(
            state,
            cdap_interaction_coefficients,
            hhsize,
            cache=True,
            joint_tour_alt=add_joint_tour_utility,
        )
        if state.get_injectable("locutor", False):
            spec.to_csv(
                state.get_output_file_path(f"cdap_spec_{hhsize}.csv"), index=True
            )
        if add_joint_tour_utility:
            # build cdap joint tour spec
            # joint_spec_dependency = spec.loc[[c for c in spec.index if c.startswith(('M_p', 'N_p', 'H_p'))]]
            joint_spec = cdap.build_cdap_joint_spec(
                state, cdap_joint_tour_coefficients, hhsize, cache=True
            )
            if state.get_injectable("locutor", False):
                joint_spec.to_csv(
                    state.get_output_file_path(
                        f"cdap_joint_spec_{hhsize}.csv",
                    ),
                    index=True,
                )

    if estimator:
        estimator.write_model_settings(model_settings, "cdap.yaml")
        estimator.write_spec(model_settings, tag="INDIV_AND_HHSIZE1_SPEC")
        estimator.write_spec(
            model_settings=model_settings, tag="FIXED_RELATIVE_PROPORTIONS_SPEC"
        )
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_table(
            cdap_interaction_coefficients,
            "cdap_interaction_coefficients",
            index=False,
            append=False,
        )
        estimator.write_choosers(persons_merged)
        for hhsize in range(2, cdap.MAX_HHSIZE + 1):
            spec = cdap.get_cached_spec(state, hhsize)
            estimator.write_table(spec, "spec_%s" % hhsize, append=False)
            if add_joint_tour_utility:
                joint_spec = cdap.get_cached_joint_spec(state, hhsize)
                estimator.write_table(
                    joint_spec, "joint_spec_%s" % hhsize, append=False
                )

    logger.info("Running cdap_simulate with %d persons", len(persons_merged.index))

    if add_joint_tour_utility:
        choices, hh_joint = cdap.run_cdap(
            state,
            persons=persons_merged,
            person_type_map=person_type_map,
            cdap_indiv_spec=cdap_indiv_spec,
            cdap_interaction_coefficients=cdap_interaction_coefficients,
            cdap_fixed_relative_proportions=cdap_fixed_relative_proportions,
            locals_d=constants,
            chunk_size=state.settings.chunk_size,
            trace_hh_id=trace_hh_id,
            trace_label=trace_label,
            add_joint_tour_utility=add_joint_tour_utility,
            compute_settings=model_settings.compute_settings,
        )
    else:
        choices = cdap.run_cdap(
            state,
            persons=persons_merged,
            person_type_map=person_type_map,
            cdap_indiv_spec=cdap_indiv_spec,
            cdap_interaction_coefficients=cdap_interaction_coefficients,
            cdap_fixed_relative_proportions=cdap_fixed_relative_proportions,
            locals_d=constants,
            chunk_size=state.settings.chunk_size,
            trace_hh_id=trace_hh_id,
            trace_label=trace_label,
            compute_settings=model_settings.compute_settings,
        )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "persons", "cdap_activity")
        if add_joint_tour_utility:
            hh_joint.index.name = "household_id"
            hh_joint = estimator.get_survey_values(
                hh_joint, "households", "has_joint_tour"
            )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    choices = choices.reindex(persons.index)
    cap_cat_type = pd.api.types.CategoricalDtype(["", "M", "N", "H"], ordered=False)
    choices = choices.astype(cap_cat_type)
    persons["cdap_activity"] = choices

    expressions.assign_columns(
        state,
        df=persons,
        model_settings=model_settings.annotate_persons,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
    )

    state.add_table("persons", persons)

    # - annotate households table
    if add_joint_tour_utility:
        hh_joint = hh_joint.reindex(households.index)
        households["has_joint_tour"] = hh_joint

    expressions.assign_columns(
        state,
        df=households,
        model_settings=model_settings.annotate_households,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_households"),
    )
    state.add_table("households", households)

    tracing.print_summary("cdap_activity", persons.cdap_activity, value_counts=True)
    logger.info(
        "cdap crosstabs:\n%s"
        % pd.crosstab(persons.ptype, persons.cdap_activity, margins=True)
    )
