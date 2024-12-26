# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

from activitysim.abm.models.util import school_escort_tours_trips
from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings
from activitysim.core.configuration.logit import BaseLogitComponentSettings
from activitysim.core.interaction_simulate import interaction_simulate
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)

# setting global defaults for max number of escortees and escortees in model
NUM_ESCORTEES = 3
NUM_CHAPERONES = 2


def determine_escorting_participants(
    choosers: pd.DataFrame, persons: pd.DataFrame, model_settings: SchoolEscortSettings
):
    """
    Determining which persons correspond to chauffer 1..n and escortee 1..n.
    Chauffers are those with the highest weight given by:
    weight = 100 * person type +  10 * gender + 1*(age > 25)
    and escortees are selected youngest to oldest.

    Choosers are only those households with escortees.
    """
    global NUM_ESCORTEES
    global NUM_CHAPERONES
    NUM_ESCORTEES = model_settings.NUM_ESCORTEES
    NUM_CHAPERONES = model_settings.NUM_CHAPERONES

    ptype_col = model_settings.PERSONTYPE_COLUMN
    sex_col = model_settings.GENDER_COLUMN
    age_col = model_settings.AGE_COLUMN

    escortee_age_cutoff = model_settings.ESCORTEE_AGE_CUTOFF
    chaperone_age_cutoff = model_settings.CHAPERONE_AGE_CUTOFF

    escortees = persons[
        persons.is_student
        & (persons[age_col] < escortee_age_cutoff)
        & (persons.cdap_activity == "M")
    ]
    households_with_escortees = escortees["household_id"]
    if len(households_with_escortees) == 0:
        logger.warning("No households with escortees found!")
    else:
        tot_households = len(choosers)
        choosers = choosers[choosers.index.isin(households_with_escortees)]
        logger.info(
            f"Proceeding with {len(choosers)} households with escortees out of {tot_households} total households"
        )

    # can specify different weights to determine chaperones
    persontype_weight = model_settings.PERSON_WEIGHT
    gender_weight = model_settings.GENDER_WEIGHT
    age_weight = model_settings.AGE_WEIGHT

    # can we move all of these to a config file?
    chaperones = persons[
        (persons[age_col] > chaperone_age_cutoff)
        & persons.household_id.isin(households_with_escortees)
    ]

    chaperones["chaperone_weight"] = (
        (persontype_weight * chaperones[ptype_col].astype("int64"))
        + (gender_weight * np.where(chaperones[sex_col].astype("int64") == 1, 1, 2))
        + (age_weight * np.where(chaperones[age_col].astype("int64") > 25, 1, 0))
    )

    chaperones["chaperone_num"] = (
        chaperones.sort_values("chaperone_weight", ascending=False)
        .groupby("household_id")
        .cumcount()
        + 1
    )
    escortees["escortee_num"] = (
        escortees.sort_values("age", ascending=True).groupby("household_id").cumcount()
        + 1
    )

    participant_columns = []
    for i in range(1, NUM_CHAPERONES + 1):
        choosers["chauf_id" + str(i)] = (
            chaperones[chaperones["chaperone_num"] == i]
            .reset_index()
            .set_index("household_id")
            .reindex(choosers.index)["person_id"]
        )
        participant_columns.append("chauf_id" + str(i))
    for i in range(1, NUM_ESCORTEES + 1):
        choosers["child_id" + str(i)] = (
            escortees[escortees["escortee_num"] == i]
            .reset_index()
            .set_index("household_id")
            .reindex(choosers.index)["person_id"]
        )
        participant_columns.append("child_id" + str(i))

    return choosers, participant_columns


def check_alts_consistency(alts: pd.DataFrame):
    """
    Checking to ensure that the alternatives file is consistent with
    the number of chaperones and escortees set in the model settings.
    """
    for i in range(1, NUM_ESCORTEES + 1):
        chauf_col = f"chauf{i}"
        # The number of chauf columns should equal the number of escortees
        assert chauf_col in alts.columns, f"Missing {chauf_col} in alternatives file"

        # Each escortee should be able to be escorted by each chaperone with ride hail or pure escort
        assert alts[chauf_col].max() == (NUM_CHAPERONES * 2)
    return


def add_prev_choices_to_choosers(
    choosers: pd.DataFrame, choices: pd.Series, alts: pd.DataFrame, stage: str
) -> pd.DataFrame:
    # adding choice details to chooser table
    escorting_choice = "school_escorting_" + stage
    choosers[escorting_choice] = choices

    stage_alts = alts.copy()
    stage_alts.columns = stage_alts.columns + "_" + stage

    choosers = (
        choosers.reset_index()
        .merge(
            stage_alts,
            how="left",
            left_on=escorting_choice,
            right_index=True,
        )
        .set_index("household_id")
    )

    return choosers


def create_school_escorting_bundles_table(choosers, tours, stage):
    """
    Creates a table that has one row for every school escorting bundle.
    Additional calculations are performed to help facilitate tour and
    trip creation including escortee order, times, etc.

    Parameters
    ----------
    choosers : pd.DataFrame
        households pre-processed for the school escorting model
    tours : pd.Dataframe
        mandatory tours
    stage : str
        inbound or outbound_cond

    Returns
    -------
    bundles : pd.DataFrame
        one school escorting bundle per row
    """
    # want to keep household_id in columns, which is already there if running in estimation mode
    if "household_id" in choosers.columns:
        choosers = choosers.reset_index(drop=True)
    else:
        choosers = choosers.reset_index()
    # creating a row for every school escorting bundle
    choosers = choosers.loc[choosers.index.repeat(choosers["nbundles"])]

    bundles = pd.DataFrame()
    # bundles.index = choosers.index
    bundles["household_id"] = choosers["household_id"]
    bundles["home_zone_id"] = choosers["home_zone_id"]
    bundles["school_escort_direction"] = (
        "outbound" if "outbound" in stage else "inbound"
    )
    bundles["bundle_num"] = bundles.groupby("household_id").cumcount() + 1

    # school escorting direction category
    escort_direction_cat = pd.api.types.CategoricalDtype(
        ["outbound", "inbound"], ordered=False
    )
    bundles["school_escort_direction"] = bundles["school_escort_direction"].astype(
        escort_direction_cat
    )

    # initialize values
    bundles["chauf_type_num"] = 0

    # getting bundle school start times and locations
    school_tours = tours[(tours.tour_type == "school") & (tours.tour_num == 1)]

    school_starts = school_tours.set_index("person_id").start
    school_ends = school_tours.set_index("person_id").end
    school_destinations = school_tours.set_index("person_id").destination
    school_origins = school_tours.set_index("person_id").origin
    school_tour_ids = school_tours.reset_index().set_index("person_id").tour_id

    for child_num in range(1, NUM_ESCORTEES + 1):
        i = str(child_num)
        bundles["bundle_child" + i] = np.where(
            choosers["bundle" + i] == bundles["bundle_num"],
            choosers["child_id" + i],
            -1,
        )
        bundles["chauf_type_num"] = np.where(
            (choosers["bundle" + i] == bundles["bundle_num"]),
            choosers["chauf" + i],
            bundles["chauf_type_num"],
        )
        bundles["time_home_to_school" + i] = np.where(
            (choosers["bundle" + i] == bundles["bundle_num"]),
            choosers["time_home_to_school" + i],
            np.NaN,
        )

        bundles["school_destination_child" + i] = reindex(
            school_destinations, bundles["bundle_child" + i]
        )
        bundles["school_origin_child" + i] = reindex(
            school_origins, bundles["bundle_child" + i]
        )
        bundles["school_start_child" + i] = reindex(
            school_starts, bundles["bundle_child" + i]
        )
        bundles["school_end_child" + i] = reindex(
            school_ends, bundles["bundle_child" + i]
        )
        bundles["school_tour_id_child" + i] = reindex(
            school_tour_ids, bundles["bundle_child" + i]
        )

    # each chauffeur option has ride share or pure escort
    bundles["chauf_num"] = np.ceil(bundles["chauf_type_num"].div(2)).astype(int)

    # getting bundle chauffeur id based on the chauffeur num
    bundles["chauf_id"] = -1
    for i in range(1, NUM_CHAPERONES + 1):
        bundles["chauf_id"] = np.where(
            bundles["chauf_num"] == i,
            choosers["chauf_id" + str(i)],
            bundles["chauf_id"],
        )
    bundles["chauf_id"] = bundles["chauf_id"].astype(int)
    assert (
        bundles["chauf_id"] > 0
    ).all(), "Invalid chauf_id's for school escort bundles!"

    # odd chauf_type_num means ride share, even means pure escort
    # this comes from the way the alternatives file is constructed where chauf_id is
    # incremented for each possible chauffeur and for each tour type
    escort_type_cat = pd.api.types.CategoricalDtype(
        ["pure_escort", "ride_share"], ordered=False
    )
    bundles["escort_type"] = np.where(
        bundles["chauf_type_num"].mod(2) == 1, "ride_share", "pure_escort"
    )
    bundles["escort_type"] = bundles["escort_type"].astype(escort_type_cat)

    # This is just pulled from the pre-processor. Will break if removed or renamed in pre-processor
    # I think this is still a better implmentation than re-calculating here...
    school_time_cols = [
        "time_home_to_school" + str(i) for i in range(1, NUM_ESCORTEES + 1)
    ]
    bundles["outbound_order"] = list(bundles[school_time_cols].values.argsort() + 1)
    bundles["inbound_order"] = list(
        (-1 * bundles[school_time_cols]).values.argsort() + 1
    )  # inbound gets reverse order
    bundles["child_order"] = np.where(
        bundles["school_escort_direction"] == "outbound",
        bundles["outbound_order"],
        bundles["inbound_order"],
    )

    # putting the bundle attributes in order of child pickup/dropoff
    bundles = school_escort_tours_trips.create_bundle_attributes(bundles)

    # getting chauffer mandatory times
    mandatory_escort_tours = tours[
        (tours.tour_category == "mandatory") & (tours.tour_num == 1)
    ]
    bundles["first_mand_tour_id"] = reindex(
        mandatory_escort_tours.reset_index().set_index("person_id").tour_id,
        bundles["chauf_id"],
    )
    bundles["first_mand_tour_dest"] = reindex(
        mandatory_escort_tours.reset_index().set_index("person_id").destination,
        bundles["chauf_id"],
    )
    bundles["first_mand_tour_purpose"] = reindex(
        mandatory_escort_tours.reset_index().set_index("person_id").tour_type,
        bundles["chauf_id"],
    )

    bundles["Alt"] = choosers["Alt"]
    bundles["Description"] = choosers["Description"]

    return bundles


class SchoolEscortSettings(BaseLogitComponentSettings, extra="forbid"):
    """
    Settings for the `telecommute_frequency` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    ALTS: Any

    NUM_ESCORTEES: int = 3
    NUM_CHAPERONES: int = 2

    PERSONTYPE_COLUMN: str = "ptype"
    GENDER_COLUMN: str = "sex"
    AGE_COLUMN: str = "age"

    ESCORTEE_AGE_CUTOFF: int = 16
    CHAPERONE_AGE_CUTOFF: int = 18

    PERSON_WEIGHT: float = 100.0
    GENDER_WEIGHT: float = 10.0
    AGE_WEIGHT: float = 1.0

    SIMULATE_CHOOSER_COLUMNS: list[str] | None = None

    SPEC: None = None
    """The school escort model does not use this setting."""

    OUTBOUND_SPEC: str = "school_escorting_outbound.csv"
    OUTBOUND_COEFFICIENTS: str = "school_escorting_coefficients_outbound.csv"
    INBOUND_SPEC: str = "school_escorting_inbound.csv"
    INBOUND_COEFFICIENTS: str = "school_escorting_coefficients_inbound.csv"
    OUTBOUND_COND_SPEC: str = "school_escorting_outbound_cond.csv"
    OUTBOUND_COND_COEFFICIENTS: str = "school_escorting_coefficients_outbound_cond.csv"

    preprocessor_outbound: PreprocessorSettings | None = None
    preprocessor_inbound: PreprocessorSettings | None = None
    preprocessor_outbound_cond: PreprocessorSettings | None = None

    no_escorting_alterative: int = 1
    """The alternative number for no escorting. Used to set the choice for households with no escortees."""

    explicit_chunk: float = 0
    """
    If > 0, use this chunk size instead of adaptive chunking.
    If less than 1, use this fraction of the total number of rows.
    """

    LOGIT_TYPE: Literal["MNL"] = "MNL"
    """Logit model mathematical form.

    * "MNL"
        Multinomial logit model.
    """


@workflow.step
def school_escorting(
    state: workflow.State,
    households: pd.DataFrame,
    households_merged: pd.DataFrame,
    persons: pd.DataFrame,
    tours: pd.DataFrame,
    model_settings: SchoolEscortSettings | None = None,
    model_settings_file_name: str = "school_escorting.yaml",
    trace_label: str = "school_escorting_simulate",
) -> None:
    """
    school escorting model

    The school escorting model determines whether children are dropped-off at or
    picked-up from school, simultaneously with the driver responsible for
    chauffeuring the children, which children are bundled together on half-tours,
    and the type of tour (pure escort versus rideshare).

    Run iteratively for an outbound choice, an inbound choice, and an outbound choice
    conditional on the inbound choice. The choices for inbound and outbound conditional
    are used to create school escort tours and trips.

    Updates / adds the following tables to the pipeline:

    ::

        - households with school escorting choice
        - tours including pure school escorting
        - school_escort_tours which contains only pure school escort tours
        - school_escort_trips
        - timetable to avoid joint tours scheduled over school escort tours

    """
    if model_settings is None:
        model_settings = SchoolEscortSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    trace_hh_id = state.settings.trace_hh_id

    # FIXME setting index as "Alt" causes crash in estimation mode...
    # happens in joint_tour_frequency_composition too!
    # alts = simulate.read_model_alts(state, model_settings.ALTS, set_index="Alt")
    alts = simulate.read_model_alts(state, model_settings.ALTS, set_index=None)
    alts.index = alts["Alt"].values

    choosers, participant_columns = determine_escorting_participants(
        households_merged, persons, model_settings
    )

    check_alts_consistency(alts)

    constants = config.get_model_constants(model_settings)
    locals_dict = {}
    locals_dict.update(constants)

    school_escorting_stages = ["outbound", "inbound", "outbound_cond"]
    escort_bundles = []
    choices = None
    for stage_num, stage in enumerate(school_escorting_stages):
        stage_trace_label = trace_label + "_" + stage
        estimator = estimation.manager.begin_estimation(
            state,
            model_name="school_escorting_" + stage,
            bundle_name="school_escorting",
        )

        model_spec_raw = state.filesystem.read_model_spec(
            file_name=getattr(model_settings, stage.upper() + "_SPEC")
        )
        coefficients_df = state.filesystem.read_model_coefficients(
            file_name=getattr(model_settings, stage.upper() + "_COEFFICIENTS")
        )
        model_spec = simulate.eval_coefficients(
            state, model_spec_raw, coefficients_df, estimator
        )

        # allow for skipping sharrow entirely in this model with `compute_settings.sharrow_skip: true`
        # or skipping stages selectively with a mapping of the stages to skip
        stage_compute_settings = model_settings.compute_settings.subcomponent_settings(
            stage.upper()
        )
        # if stage_sharrow_skip:
        #     locals_dict["_sharrow_skip"] = True
        # else:
        #     locals_dict.pop("_sharrow_skip", None)

        # reduce memory by limiting columns if selected columns are supplied
        chooser_columns = model_settings.SIMULATE_CHOOSER_COLUMNS
        if chooser_columns is not None:
            chooser_columns = chooser_columns + participant_columns
            choosers = choosers[chooser_columns]

        # add previous data to stage
        if stage_num >= 1:
            choosers = add_prev_choices_to_choosers(
                choosers, choices, alts, school_escorting_stages[stage_num - 1]
            )

        locals_dict.update(coefficients_df)

        logger.info("Running %s with %d households", stage_trace_label, len(choosers))

        preprocessor_settings = getattr(model_settings, "preprocessor_" + stage, None)
        if preprocessor_settings:
            expressions.assign_columns(
                state,
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=stage_trace_label,
            )

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings, tag=stage.upper() + "_SPEC")
            estimator.write_coefficients(
                coefficients_df, file_name=stage.upper() + "_COEFFICIENTS"
            )
            estimator.write_choosers(choosers)
            if state.settings.multiprocess:
                estimator.write_alternatives(alts, bundle_directory=False)
            else:
                estimator.write_alternatives(alts, bundle_directory=True)

            # FIXME #interaction_simulate_estimation_requires_chooser_id_in_df_column
            #  shuold we do it here or have interaction_simulate do it?
            # chooser index must be duplicated in column or it will be omitted from interaction_dataset
            # estimation requires that chooser_id is either in index or a column of interaction_dataset
            # so it can be reformatted (melted) and indexed by chooser_id and alt_id
            assert choosers.index.name == "household_id"
            assert "household_id" not in choosers.columns
            choosers["household_id"] = choosers.index

            # FIXME set_alt_id - do we need this for interaction_simulate estimation bundle tables?
            estimator.set_alt_id("alt_id")

            estimator.set_chooser_id(choosers.index.name)

        log_alt_losers = state.settings.log_alt_losers

        choices = interaction_simulate(
            state,
            choosers=choosers,
            alternatives=alts,
            spec=model_spec,
            log_alt_losers=log_alt_losers,
            locals_d=locals_dict,
            trace_label=stage_trace_label,
            trace_choice_name="school_escorting_" + stage,
            estimator=estimator,
            explicit_chunk_size=model_settings.explicit_chunk,
            compute_settings=stage_compute_settings,
        )

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "households", "school_escorting_" + stage
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # choices are merged into households by index (household_id)
        # households that do not have an escortee are assigned the no_escorting_alterative
        escorting_choice = "school_escorting_" + stage
        households[escorting_choice] = choices
        households[escorting_choice].fillna(
            model_settings.no_escorting_alterative, inplace=True
        )

        # tracing each step -- outbound, inbound, outbound_cond
        tracing.print_summary(
            escorting_choice, households[escorting_choice], value_counts=True
        )

        if trace_hh_id:
            state.tracing.trace_df(
                households, label=escorting_choice, warn_if_empty=True
            )

        if stage_num >= 1:
            choosers["Alt"] = choices
            choosers = choosers.join(alts.set_index("Alt"), how="left", on="Alt")
            bundles = create_school_escorting_bundles_table(
                choosers[choosers["Alt"] > 1], tours, stage
            )
            escort_bundles.append(bundles)

    escort_bundles = pd.concat(escort_bundles)

    # Only want to create bundles and tours and trips if at least one household has school escorting
    if len(escort_bundles) > 0:
        escort_bundles["bundle_id"] = (
            escort_bundles["household_id"] * 10
            + escort_bundles.groupby("household_id").cumcount()
            + 1
        )
        escort_bundles.sort_values(
            by=["household_id", "school_escort_direction"],
            ascending=[True, False],
            inplace=True,
        )

        school_escort_tours = school_escort_tours_trips.create_pure_school_escort_tours(
            state, escort_bundles
        )
        chauf_tour_id_map = {
            v: k for k, v in school_escort_tours["bundle_id"].to_dict().items()
        }
        escort_bundles["chauf_tour_id"] = np.where(
            escort_bundles["escort_type"] == "ride_share",
            escort_bundles["first_mand_tour_id"],
            escort_bundles["bundle_id"].map(chauf_tour_id_map),
        )

        assert (
            escort_bundles["chauf_tour_id"].notnull().all()
        ), f"chauf_tour_id is null for {escort_bundles[escort_bundles['chauf_tour_id'].isna()]}. Check availability conditions."

        tours = school_escort_tours_trips.add_pure_escort_tours(
            tours, school_escort_tours
        )
        tours = school_escort_tours_trips.process_tours_after_escorting_model(
            state, escort_bundles, tours
        )
        school_escort_trips = school_escort_tours_trips.create_school_escort_trips(
            escort_bundles
        )

    else:
        # create empty school escort tours & trips tables to be used downstream
        tours["school_esc_outbound"] = pd.NA
        tours["school_esc_inbound"] = pd.NA
        tours["school_escort_direction"] = pd.NA
        tours["next_pure_escort_start"] = pd.NA
        school_escort_tours = pd.DataFrame(columns=tours.columns)
        trip_cols = [
            "household_id",
            "person_id",
            "tour_id",
            "trip_id",
            "outbound",
            "depart",
            "purpose",
            "destination",
            "escort_participants",
            "chauf_tour_id",
            "primary_purpose",
        ]
        school_escort_trips = pd.DataFrame(columns=trip_cols)

    school_escort_trips["primary_purpose"] = school_escort_trips[
        "primary_purpose"
    ].astype(state.get_dataframe("tours")["tour_type"].dtype)
    school_escort_trips["purpose"] = school_escort_trips["purpose"].astype(
        state.get_dataframe("tours")["tour_type"].dtype
    )

    # update pipeline
    state.add_table("households", households)
    state.add_table("tours", tours)
    state.get_rn_generator().drop_channel("tours")
    state.get_rn_generator().add_channel("tours", tours)
    state.add_table("escort_bundles", escort_bundles)
    # save school escorting tours and trips in pipeline so we can overwrite results from downstream models
    state.add_table("school_escort_tours", school_escort_tours)
    state.add_table("school_escort_trips", school_escort_trips)

    # updating timetable object with pure escort tours so joint tours do not schedule ontop
    timetable = state.get_injectable("timetable")

    # Need to do this such that only one person is in nth_tours
    # thus, looping through tour_category and tour_num
    # including mandatory tours because their start / end times may have
    # changed to match the school escort times
    for tour_category in tours.tour_category.unique():
        for _tour_num, nth_tours in tours[tours.tour_category == tour_category].groupby(
            "tour_num", sort=True
        ):
            timetable.assign(
                window_row_ids=nth_tours["person_id"], tdds=nth_tours["tdd"]
            )

    timetable.replace_table(state)
