# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core.interaction_simulate import interaction_simulate
from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import expressions
from activitysim.core import los

import activitysim.abm.tables.tours as tables_tours
from activitysim.core.util import reindex

import pandas as pd
import numpy as np
import warnings

from .util import estimation
from .util import school_escort_tours_trips

logger = logging.getLogger(__name__)

# setting global defaults for max number of escortees and escortees in model
NUM_ESCORTEES = 3
NUM_CHAPERONES = 2


def determine_escorting_paricipants(choosers, persons, model_settings):
    """
    Determining which persons correspond to chauffer 1..n and escortee 1..n.
    Chauffers are those with the highest weight given by:
     weight = 100 * person type +  10*gender + 1*(age > 25)
    and escortees are selected youngest to oldest.
    """

    NUM_ESCORTEES = model_settings["NUM_ESCORTEES"]
    NUM_CHAPERONES = model_settings["NUM_CHAPERONES"]

    # is this cut correct?
    escortees = persons[
        persons.is_student & (persons.age < 16) & (persons.cdap_activity == "M")
    ]
    households_with_escortees = escortees["household_id"]

    persontype_weight = 100
    gender_weight = 10
    age_weight = 1

    # can we move all of these to a config file?
    chaperones = persons[
        (persons.age > 18) & persons.household_id.isin(households_with_escortees)
    ]

    chaperones["chaperone_weight"] = (
        (persontype_weight * chaperones["ptype"])
        + (gender_weight * np.where(chaperones["sex"] == 1, 1, 0))
        + (age_weight * np.where(chaperones["age"] > 25, 1, 0))
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


def add_prev_choices_to_choosers(choosers, choices, alts, stage):
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
            right_on=stage_alts.index.name,
        )
        .set_index("household_id")
    )

    return choosers


def create_bundle_attributes(row):
    escortee_str = ""
    escortee_num_str = ""
    school_dests_str = ""
    school_starts_str = ""
    school_ends_str = ""
    school_tour_ids_str = ""
    num_escortees = 0

    for child_num in row["child_order"]:
        child_num = str(child_num)
        child_id = int(row["bundle_child" + child_num])

        if child_id > 0:
            num_escortees += 1
            school_dest = str(int(row["school_destination_child" + child_num]))
            school_start = str(int(row["school_start_child" + child_num]))
            school_end = str(int(row["school_end_child" + child_num]))
            school_tour_id = str(int(row["school_tour_id_child" + child_num]))

            if escortee_str == "":
                escortee_str = str(child_id)
                escortee_num_str = str(child_num)
                school_dests_str = school_dest
                school_starts_str = school_start
                school_ends_str = school_end
                school_tour_ids_str = school_tour_id
            else:
                escortee_str = escortee_str + "_" + str(child_id)
                escortee_num_str = escortee_num_str + "_" + str(child_num)
                school_dests_str = school_dests_str + "_" + school_dest
                school_starts_str = school_starts_str + "_" + school_start
                school_ends_str = school_ends_str + "_" + school_end
                school_tour_ids_str = school_tour_ids_str + "_" + school_tour_id

    row["escortees"] = escortee_str
    row["escortee_nums"] = escortee_num_str
    row["num_escortees"] = num_escortees
    row["school_destinations"] = school_dests_str
    row["school_starts"] = school_starts_str
    row["school_ends"] = school_ends_str
    row["school_tour_ids"] = school_tour_ids_str
    return row


def create_school_escorting_bundles_table(choosers, tours, stage):
    # making a table of bundles
    choosers = choosers.reset_index()
    choosers = choosers.loc[choosers.index.repeat(choosers["nbundles"])]

    bundles = pd.DataFrame()
    # bundles.index = choosers.index
    bundles["household_id"] = choosers["household_id"]
    bundles["home_zone_id"] = choosers["home_zone_id"]
    bundles["school_escort_direction"] = (
        "outbound" if "outbound" in stage else "inbound"
    )
    bundles["bundle_num"] = bundles.groupby("household_id").cumcount() + 1

    # initialize values
    bundles["chauf_type_num"] = 0

    # getting bundle school start times and locations
    school_tours = tours[(tours.tour_type == "school") & (tours.tour_num == 1)]

    school_starts = school_tours.set_index("person_id").start
    school_ends = school_tours.set_index("person_id").end
    school_destinations = school_tours.set_index("person_id").destination
    school_origins = school_tours.set_index("person_id").origin
    school_tour_ids = school_tours.reset_index().set_index("person_id").tour_id

    for child_num in range(1, 4):
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

    # FIXME assumes only two chauffeurs
    bundles["chauf_id"] = np.where(
        bundles["chauf_type_num"] <= 2, choosers["chauf_id1"], choosers["chauf_id2"]
    ).astype(int)
    bundles["chauf_num"] = np.where(bundles["chauf_type_num"] <= 2, 1, 2)
    bundles["escort_type"] = np.where(
        bundles["chauf_type_num"].isin([1, 3]), "ride_share", "pure_escort"
    )

    # FIXME this is just pulled from the pre-processor... would break if removed or renamed in pre-processor
    school_time_cols = ["time_home_to_school" + str(i) for i in range(1, 4)]
    bundles["outbound_order"] = list(bundles[school_time_cols].values.argsort() + 1)
    bundles["inbound_order"] = list(
        (-1 * bundles[school_time_cols]).values.argsort() + 1
    )  # inbound gets reverse order
    bundles["child_order"] = np.where(
        bundles["school_escort_direction"] == "outbound",
        bundles["outbound_order"],
        bundles["inbound_order"],
    )

    bundles = bundles.apply(lambda row: create_bundle_attributes(row), axis=1)

    # getting chauffer mandatory times
    mandatory_escort_tours = tours[
        (tours.tour_category == "mandatory") & (tours.tour_num == 1)
    ]
    bundles["first_mand_tour_start_time"] = reindex(
        mandatory_escort_tours.set_index("person_id").start, bundles["chauf_id"]
    )
    bundles["first_mand_tour_end_time"] = reindex(
        mandatory_escort_tours.set_index("person_id").end, bundles["chauf_id"]
    )
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
    # bundles.set_index('bundle_id', inplace=True)

    return bundles


@inject.step()
def school_escorting(
    households, households_merged, persons, tours, chunk_size, trace_hh_id
):
    """
    The school escorting model determines whether children are dropped-off at or
    picked-up from school, simultaneously with the driver responsible for
    chauffeuring the children, which children are bundled together on half-tours,
    and the type of tour (pure escort versus rideshare).
    """
    trace_label = "school_escorting_simulate"
    model_settings_file_name = "school_escorting.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    persons = persons.to_frame()
    households = households.to_frame()
    households_merged = households_merged.to_frame()
    tours = tours.to_frame()

    alts = simulate.read_model_alts(model_settings["ALTS"], set_index="Alt")

    households_merged, participant_columns = determine_escorting_paricipants(
        households_merged, persons, model_settings
    )

    constants = config.get_model_constants(model_settings)
    locals_dict = {}
    locals_dict.update(constants)

    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    school_escorting_stages = ["outbound", "inbound", "outbound_cond"]
    # school_escorting_stages = ['outbound', 'inbound']
    escort_bundles = []
    for stage_num, stage in enumerate(school_escorting_stages):
        stage_trace_label = trace_label + "_" + stage
        estimator = estimation.manager.begin_estimation("school_escorting_" + stage)

        model_spec_raw = simulate.read_model_spec(
            file_name=model_settings[stage.upper() + "_SPEC"]
        )
        coefficients_df = simulate.read_model_coefficients(
            file_name=model_settings[stage.upper() + "_COEFFICIENTS"]
        )
        model_spec = simulate.eval_coefficients(
            model_spec_raw, coefficients_df, estimator
        )

        # reduce memory by limiting columns if selected columns are supplied
        chooser_columns = model_settings.get("SIMULATE_CHOOSER_COLUMNS", None)
        if chooser_columns is not None:
            chooser_columns = chooser_columns + participant_columns
            choosers = households_merged[chooser_columns]
        else:
            choosers = households_merged

        # add previous data to stage
        if stage_num >= 1:
            choosers = add_prev_choices_to_choosers(
                choosers, choices, alts, school_escorting_stages[stage_num - 1]
            )

        locals_dict.update(coefficients_df)

        logger.info("Running %s with %d households", stage_trace_label, len(choosers))

        preprocessor_settings = model_settings.get("preprocessor_" + stage, None)
        if preprocessor_settings:
            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=stage_trace_label,
            )

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        log_alt_losers = config.setting("log_alt_losers", False)

        choices = interaction_simulate(
            choosers=choosers,
            alternatives=alts,
            spec=model_spec,
            log_alt_losers=log_alt_losers,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            trace_label=stage_trace_label,
            trace_choice_name="school_escorting_" + "stage",
            estimator=estimator,
        )

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "households", "school_escorting_" + stage
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # no need to reindex as we used all households
        escorting_choice = "school_escorting_" + stage
        households[escorting_choice] = choices

        # should this tracing be done for every step? - I think so...
        tracing.print_summary(
            escorting_choice, households[escorting_choice], value_counts=True
        )

        if trace_hh_id:
            tracing.trace_df(households, label=escorting_choice, warn_if_empty=True)

        if stage_num >= 1:
            choosers["Alt"] = choices
            choosers = choosers.join(alts, how="left", on="Alt")
            bundles = create_school_escorting_bundles_table(
                choosers[choosers["Alt"] > 1], tours, stage
            )
            escort_bundles.append(bundles)

    escort_bundles = pd.concat(escort_bundles)
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
        escort_bundles
    )
    chauf_tour_id_map = {
        v: k for k, v in school_escort_tours["bundle_id"].to_dict().items()
    }
    escort_bundles["chauf_tour_id"] = np.where(
        escort_bundles["escort_type"] == "ride_share",
        escort_bundles["first_mand_tour_id"],
        escort_bundles["bundle_id"].map(chauf_tour_id_map),
    )

    tours = school_escort_tours_trips.add_pure_escort_tours(tours, school_escort_tours)
    tours = school_escort_tours_trips.process_tours_after_escorting_model(
        escort_bundles, tours
    )

    school_escort_trips = school_escort_tours_trips.create_school_escort_trips(
        escort_bundles
    )

    # update pipeline
    pipeline.replace_table("households", households)
    pipeline.replace_table("tours", tours)
    pipeline.get_rn_generator().drop_channel("tours")
    pipeline.get_rn_generator().add_channel("tours", tours)
    pipeline.replace_table("escort_bundles", escort_bundles)
    # save school escorting tours and trips in pipeline so we can overwrite results from downstream models
    pipeline.replace_table("school_escort_tours", school_escort_tours)
    pipeline.replace_table("school_escort_trips", school_escort_trips)

    # updating timetable object with pure escort tours so joint tours do not schedule ontop
    timetable = inject.get_injectable("timetable")

    # Need to do this such that only one person is in nth_tours
    # thus, looping through tour_category and tour_num
    # including mandatory tours because their start / end times may have 
    # changed to match the school escort times
    for tour_category in tours.tour_category.unique():
        for tour_num, nth_tours in tours[tours.tour_category == tour_category].groupby(
            "tour_num", sort=True
        ):
            timetable.assign(
                window_row_ids=nth_tours["person_id"], tdds=nth_tours["tdd"]
            )

    timetable.replace_table()