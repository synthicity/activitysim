# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

from activitysim.abm.models.util import canonical_ids as cid
from activitysim.core import workflow
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(ch)

CONSTANTS = {}

SURVEY_TOUR_ID = "survey_tour_id"
SURVEY_PARENT_TOUR_ID = "survey_parent_tour_id"
SURVEY_PARTICIPANT_ID = "survey_participant_id"
SURVEY_TRIP_ID = "survey_trip_id"
ASIM_TOUR_ID = "tour_id"
ASIM_PARENT_TOUR_ID = "parent_tour_id"
ASIM_TRIP_ID = "trip_id"

ASIM_PARTICIPANT_ID = "participant_id"

survey_tables = {
    "households": {"file_name": "survey_households.csv", "index": "household_id"},
    "persons": {"file_name": "survey_persons.csv", "index": "person_id"},
    "tours": {"file_name": "survey_tours.csv"},
    "joint_tour_participants": {"file_name": "survey_joint_tour_participants.csv"},
    "trips": {"file_name": "survey_trips.csv"},
}

outputs = {
    "households": "override_households.csv",
    "persons": "override_persons.csv",
    "tours": "override_tours.csv",
    "joint_tour_participants": "override_joint_tour_participants.csv",
    "trips": "override_trips.csv",
}

control_tables = {
    "households": {"file_name": "final_households.csv", "index": "household_id"},
    "persons": {"file_name": "final_persons.csv", "index": "person_id"},
    "tours": {"file_name": "final_tours.csv"},
    "joint_tour_participants": {"file_name": "final_joint_tour_participants.csv"},
    "trips": {"file_name": "final_trips.csv"},
}
apply_controls = True
skip_controls = not apply_controls


def mangle_ids(ids):
    return ids * 10


def unmangle_ids(ids):
    return ids // 10


def infer_cdap_activity(persons, tours, joint_tour_participants):
    mandatory_tour_types = ["work", "school"]
    non_mandatory_tour_types = [
        "escort",
        "shopping",
        "othmaint",
        "othdiscr",
        "eatout",
        "social",
    ]

    num_mandatory_tours = (
        tours[tours.tour_type.isin(mandatory_tour_types)]
        .groupby("person_id")
        .size()
        .reindex(persons.index)
        .fillna(0)
        .astype(np.int8)
    )

    num_non_mandatory_tours = (
        tours[tours.tour_type.isin(non_mandatory_tour_types)]
        .groupby("person_id")
        .size()
        .reindex(persons.index)
        .fillna(0)
        .astype(np.int8)
    )

    num_joint_tours = (
        joint_tour_participants.groupby("person_id")
        .size()
        .reindex(persons.index)
        .fillna(0)
        .astype(np.int8)
    )

    num_non_mandatory_tours += num_joint_tours

    cdap_activity = pd.Series("H", index=persons.index)
    cdap_activity = cdap_activity.where(num_mandatory_tours == 0, "M")
    cdap_activity = cdap_activity.where(
        (cdap_activity == "M") | (num_non_mandatory_tours == 0), "N"
    )

    return cdap_activity


def infer_mandatory_tour_frequency(persons, tours):
    num_work_tours = (
        tours[tours.tour_type == "work"]
        .groupby("person_id")
        .size()
        .reindex(persons.index)
        .fillna(0)
        .astype(np.int8)
    )

    num_school_tours = (
        tours[tours.tour_type == "school"]
        .groupby("person_id")
        .size()
        .reindex(persons.index)
        .fillna(0)
        .astype(np.int8)
    )

    mtf = {
        0: "",
        1: "work1",
        2: "work2",
        10: "school1",
        20: "school2",
        11: "work_and_school",
    }

    mandatory_tour_frequency = (num_work_tours + num_school_tours * 10).map(mtf)

    return mandatory_tour_frequency


def infer_non_mandatory_tour_frequency(configs_dir, persons, tours):
    def read_alts():
        # escort,shopping,othmaint,othdiscr,eatout,social
        # 0,0,0,0,0,0
        # 0,0,0,1,0,0, ...
        alts = pd.read_csv(
            os.path.join(configs_dir, "non_mandatory_tour_frequency_alternatives.csv"),
            comment="#",
        )
        alts = alts.astype(np.int8)  # - NARROW
        return alts

    tours = tours[tours.tour_category == "non_mandatory"]

    alts = read_alts()
    tour_types = list(alts.columns.values)
    if "tot_tours" in tour_types:
        tour_types.remove("tot_tours")

    # tour_frequency is index in alts table
    alts["alt_id"] = alts.index

    # actual tour counts (may exceed counts envisioned by alts)
    unconstrained_tour_counts = pd.DataFrame(index=persons.index)
    for tour_type in tour_types:
        unconstrained_tour_counts[tour_type] = (
            tours[tours.tour_type == tour_type]
            .groupby("person_id")
            .size()
            .reindex(persons.index)
            .fillna(0)
            .astype(np.int8)
        )

    # unextend tour counts
    # activitysim extend tours counts based on a probability table
    # counts can only be extended if original count is between 1 and 4
    # and tours can only be extended if their count is at the max possible
    max_tour_counts = alts[tour_types].max(axis=0)
    constrained_tour_counts = pd.DataFrame(index=persons.index)
    for tour_type in tour_types:
        constrained_tour_counts[tour_type] = unconstrained_tour_counts[tour_type].clip(
            upper=max_tour_counts[tour_type]
        )

    # persons whose tours were constrained who aren't eligible for extension becuase they have > 4 constrained tours
    has_constrained_tours = (unconstrained_tour_counts != constrained_tour_counts).any(
        axis=1
    )
    print("%s persons with constrained tours" % (has_constrained_tours.sum()))
    too_many_tours = has_constrained_tours & constrained_tour_counts.sum(axis=1) > 4
    if too_many_tours.any():
        print("%s persons with too many tours" % (too_many_tours.sum()))
        print(constrained_tour_counts[too_many_tours])
        # not sure what to do about this. Throw out some tours? let them through?
        print("not sure what to do about this. Throw out some tours? let them through?")
        assert False

    # determine alt id corresponding to constrained_tour_counts
    # need to do index waltz because pd.merge doesn't preserve index in this case
    alt_id = (
        pd.merge(
            constrained_tour_counts.reset_index(),
            alts,
            left_on=tour_types,
            right_on=tour_types,
            how="left",
        )
        .set_index(persons.index.name)
        .alt_id
    )

    # did we end up with any tour frequencies not in alts?
    if alt_id.isna().any():
        bad_tour_frequencies = alt_id.isna()
        logger.warning("WARNING Bad joint tour frequencies\n\n")
        logger.warning(
            "\nWARNING Bad non_mandatory tour frequencies: num_tours\n%s"
            % constrained_tour_counts[bad_tour_frequencies]
        )
        logger.warning(
            "\nWARNING Bad non_mandatory tour frequencies: num_tours\n%s"
            % tours[
                tours.person_id.isin(persons.index[bad_tour_frequencies])
            ].sort_values("person_id")
        )
        bug

    tf = unconstrained_tour_counts.rename(
        columns={tour_type: "_%s" % tour_type for tour_type in tour_types}
    )
    tf["non_mandatory_tour_frequency"] = alt_id
    return tf


def infer_joint_tour_frequency(configs_dir, households, tours):
    def read_alts():
        # right now this file just contains the start and end hour
        alts = pd.read_csv(
            os.path.join(configs_dir, "joint_tour_frequency_alternatives.csv"),
            comment="#",
            index_col="alt",
        )
        alts = alts.astype(np.int8)  # - NARROW
        return alts

    alts = read_alts()
    tour_types = list(alts.columns.values)

    assert len(alts.index[(alts == 0).all(axis=1)]) == 1  # should be one zero_tours alt
    zero_tours_alt = alts.index[(alts == 0).all(axis=1)].values[0]

    alts["joint_tour_frequency"] = alts.index
    joint_tours = tours[tours.tour_category == "joint"]

    num_tours = pd.DataFrame(index=households.index)
    for tour_type in tour_types:
        joint_tour_is_tour_type = joint_tours.tour_type == tour_type
        if joint_tour_is_tour_type.any():
            num_tours[tour_type] = (
                joint_tours[joint_tour_is_tour_type]
                .groupby("household_id")
                .size()
                .reindex(households.index)
                .fillna(0)
            )
        else:
            logger.warning(
                "WARNING infer_joint_tour_frequency - no tours of type '%s'" % tour_type
            )
            num_tours[tour_type] = 0
    num_tours = num_tours.fillna(0).astype(np.int64)

    # need to do index waltz because pd.merge doesn't preserve index in this case
    jtf = pd.merge(
        num_tours.reset_index(),
        alts,
        left_on=tour_types,
        right_on=tour_types,
        how="left",
    ).set_index(households.index.name)

    if jtf.joint_tour_frequency.isna().any():
        bad_tour_frequencies = jtf.joint_tour_frequency.isna()
        logger.warning("WARNING Bad joint tour frequencies\n\n")
        logger.warning(
            "\nWARNING Bad joint tour frequencies: num_tours\n%s"
            % num_tours[bad_tour_frequencies]
        )
        logger.warning(
            "\nWARNING Bad joint tour frequencies: num_tours\n%s"
            % joint_tours[
                joint_tours.household_id.isin(households.index[bad_tour_frequencies])
            ]
        )
        bug

    logger.info(
        "infer_joint_tour_frequency: %s households with joint tours",
        (jtf.joint_tour_frequency != zero_tours_alt).sum(),
    )

    return jtf.joint_tour_frequency


def infer_joint_tour_composition(persons, tours, joint_tour_participants):
    """
    assign joint_tours a 'composition' column ('adults', 'children', or 'mixed')
    depending on the composition of the joint_tour_participants
    """
    joint_tours = tours[tours.tour_category == "joint"].copy()

    joint_tour_participants = pd.merge(
        joint_tour_participants,
        persons,
        left_on="person_id",
        right_index=True,
        how="left",
    )

    # FIXME - computed by asim annotate persons - not needed if embeded in asim and called just-in-time
    if "adult" not in joint_tour_participants:
        joint_tour_participants["adult"] = joint_tour_participants.age >= 18

    tour_has_adults = (
        joint_tour_participants[joint_tour_participants.adult]
        .groupby(SURVEY_TOUR_ID)
        .size()
        .reindex(joint_tours[SURVEY_TOUR_ID])
        .fillna(0)
        > 0
    )

    tour_has_children = (
        joint_tour_participants[~joint_tour_participants.adult]
        .groupby([SURVEY_TOUR_ID])
        .size()
        .reindex(joint_tours[SURVEY_TOUR_ID])
        .fillna(0)
        > 0
    )

    assert (tour_has_adults | tour_has_children).all()

    joint_tours["composition"] = np.where(
        tour_has_adults, np.where(tour_has_children, "mixed", "adults"), "children"
    )

    return joint_tours.composition.reindex(tours.index).fillna("").astype(str)


def infer_tour_scheduling(configs_dir, tours):
    # given start and end periods, infer tdd

    def read_tdd_alts():
        # right now this file just contains the start and end hour
        tdd_alts = pd.read_csv(
            os.path.join(configs_dir, "tour_departure_and_duration_alternatives.csv")
        )
        tdd_alts["duration"] = tdd_alts.end - tdd_alts.start
        tdd_alts = tdd_alts.astype(np.int8)  # - NARROW

        tdd_alts["tdd"] = tdd_alts.index
        return tdd_alts

    tdd_alts = read_tdd_alts()

    if not tours.start.isin(tdd_alts.start).all():
        print(tours[~tours.start.isin(tdd_alts.start)])
    assert tours.start.isin(tdd_alts.start).all(), "not all tour starts in tdd_alts"

    assert tours.end.isin(tdd_alts.end).all(), "not all tour starts in tdd_alts"

    tdds = pd.merge(
        tours[["start", "end"]],
        tdd_alts,
        left_on=["start", "end"],
        right_on=["start", "end"],
        how="left",
    )

    if tdds.tdd.isna().any():
        bad_tdds = tours[tdds.tdd.isna()]
        print("Bad tour start/end times:")
        print(bad_tdds)
        bug

    # print("tdd_alts\n%s" %tdd_alts, "\n")
    # print("tours\n%s" %tours[['start', 'end']])
    # print("tdds\n%s" %tdds)
    return tdds.tdd


def patch_tour_ids(state: workflow.State, persons, tours, joint_tour_participants):
    def set_tour_index(state, tours, parent_tour_num_col, is_joint):
        group_cols = ["person_id", "tour_category", "tour_type"]

        if "parent_tour_num" in tours:
            group_cols += ["parent_tour_num"]

        tours["tour_type_num"] = (
            tours.sort_values(by=group_cols).groupby(group_cols).cumcount() + 1
        )

        return cid.set_tour_index(
            state, tours, parent_tour_num_col=parent_tour_num_col, is_joint=is_joint
        )

    assert "mandatory_tour_frequency" in persons

    # replace survey_tour ids with asim standard tour_ids (which are based on person_id and tour_type)

    #####################
    # mandatory tours
    #####################
    mandatory_tours = set_tour_index(
        state,
        tours[tours.tour_category == "mandatory"],
        parent_tour_num_col=None,
        is_joint=False,
    )

    assert mandatory_tours.index.name == "tour_id"

    #####################
    # joint tours
    #####################

    # joint tours tour_id was assigned based on person_id of the first person in household (PNUM == 1)
    # because the actual point person forthe tour is only identified later in joint_tour_participants)
    temp_point_persons = persons.loc[persons.PNUM == 1, ["household_id"]]
    temp_point_persons["person_id"] = temp_point_persons.index
    temp_point_persons.set_index("household_id", inplace=True)

    # patch person_id with value of temp_point_person_id and use it to set_tour_index
    joint_tours = tours[tours.tour_category == "joint"]
    joint_tours["cache_point_person_id"] = joint_tours["person_id"]
    joint_tours["person_id"] = reindex(
        temp_point_persons.person_id, joint_tours.household_id
    )

    joint_tours = set_tour_index(
        state, joint_tours, parent_tour_num_col=None, is_joint=True
    )
    joint_tours["person_id"] = joint_tours["cache_point_person_id"]
    del joint_tours["cache_point_person_id"]

    # patch tour_id column in patched_joint_tour_participants
    patched_joint_tour_participants = joint_tour_participants.copy()
    asim_tour_id = pd.Series(joint_tours.index, index=joint_tours[SURVEY_TOUR_ID])
    patched_joint_tour_participants[ASIM_TOUR_ID] = reindex(
        asim_tour_id, patched_joint_tour_participants[SURVEY_TOUR_ID]
    )

    # participant_id is formed by combining tour_id and participant pern.PNUM
    # pathological knowledge, but awkward to conflate with joint_tour_participation.py logic
    participant_pnum = reindex(
        persons.PNUM, patched_joint_tour_participants["person_id"]
    )
    patched_joint_tour_participants[ASIM_PARTICIPANT_ID] = (
        patched_joint_tour_participants[ASIM_TOUR_ID] * cid.MAX_PARTICIPANT_PNUM
    ) + participant_pnum

    #####################
    # non_mandatory tours
    #####################

    non_mandatory_tours = set_tour_index(
        state,
        tours[tours.tour_category == "non_mandatory"],
        parent_tour_num_col=None,
        is_joint=False,
    )

    #####################
    # atwork tours
    #####################

    atwork_tours = tours[tours.tour_category == "atwork"]

    # patch atwork tours parent_tour_id before assigning their tour_id

    # tours for workers with both work and school trips should have lower tour_num for work,
    # tours for students with both work and school trips should have lower tour_num for school
    # tours are already sorted, but schools comes before work (which is alphabetical, not the alternative id order),
    # so work_and_school tour_nums are correct for students (school=1, work=2) but workers need to be flipped
    mandatory_tour_frequency = reindex(
        persons.mandatory_tour_frequency, mandatory_tours.person_id
    )
    is_worker = reindex(persons.pemploy, mandatory_tours.person_id).isin(
        [CONSTANTS["PEMPLOY_FULL"], CONSTANTS["PEMPLOY_PART"]]
    )
    work_and_school_and_worker = (
        mandatory_tour_frequency == "work_and_school"
    ) & is_worker

    # calculate tour_num for work tours (required to set_tour_index for atwork subtours)

    parent_tours = mandatory_tours[[SURVEY_TOUR_ID]]
    parent_tours["tour_num"] = (
        mandatory_tours.sort_values(by=["person_id", "tour_category", "tour_type"])
        .groupby(["person_id", "tour_category"])
        .cumcount()
        + 1
    )

    parent_tours.tour_num = parent_tours.tour_num.where(
        ~work_and_school_and_worker, 3 - parent_tours.tour_num
    )
    parent_tours = parent_tours.set_index(SURVEY_TOUR_ID, drop=True)

    # temporarily add parent_tour_num column to atwork tours, call set_tour_index, and then delete it
    atwork_tours["parent_tour_num"] = reindex(
        parent_tours.tour_num, atwork_tours[SURVEY_PARENT_TOUR_ID]
    )

    atwork_tours = set_tour_index(
        state, atwork_tours, parent_tour_num_col="parent_tour_num", is_joint=False
    )

    del atwork_tours["parent_tour_num"]

    # tours['household_id'] = reindex(persons.household_id, tours.person_id)
    asim_tour_id = pd.Series(
        mandatory_tours.index, index=mandatory_tours[SURVEY_TOUR_ID]
    )
    atwork_tours[ASIM_PARENT_TOUR_ID] = reindex(
        asim_tour_id, atwork_tours[SURVEY_PARENT_TOUR_ID]
    )

    #####################
    # concat tours
    #####################

    # only true for fake data
    assert (
        mandatory_tours.index == unmangle_ids(mandatory_tours[SURVEY_TOUR_ID])
    ).all()
    assert (joint_tours.index == unmangle_ids(joint_tours[SURVEY_TOUR_ID])).all()
    assert (
        non_mandatory_tours.index == unmangle_ids(non_mandatory_tours[SURVEY_TOUR_ID])
    ).all()

    patched_tours = pd.concat(
        [mandatory_tours, joint_tours, non_mandatory_tours, atwork_tours]
    )

    assert patched_tours.index.name == ASIM_TOUR_ID
    patched_tours = patched_tours.reset_index()

    del patched_tours["tour_type_num"]

    assert ASIM_TOUR_ID in patched_tours
    assert ASIM_PARENT_TOUR_ID in patched_tours

    return patched_tours, patched_joint_tour_participants


def infer_atwork_subtour_frequency(configs_dir, tours):
    # first column is 'atwork_subtour_frequency' nickname, remaining columns are trip type counts
    alts = pd.read_csv(
        os.path.join(configs_dir, "atwork_subtour_frequency_alternatives.csv"),
        comment="#",
    )
    tour_types = list(
        alts.drop(columns=alts.columns[0]).columns
    )  # get trip_types, ignoring first column
    alts["alt_id"] = alts.index

    #             alt  eat  business  maint  alt_id
    # 0   no_subtours    0         0      0       0
    # 1           eat    1         0      0       1
    # 2     business1    0         1      0       2
    # 3         maint    0         0      1       3
    # 4     business2    0         2      0       4
    # 5  eat_business    1         1      0       5

    work_tours = tours[tours.tour_type == "work"]
    work_tours = work_tours[[ASIM_TOUR_ID]]

    subtours = tours[tours.tour_category == "atwork"]
    subtours = subtours[["tour_id", "tour_type", "parent_tour_id"]]

    # actual tour counts (may exceed counts envisioned by alts)
    tour_counts = pd.DataFrame(index=work_tours[ASIM_TOUR_ID])
    for tour_type in tour_types:
        # count subtours of this type by parent_tour_id
        tour_type_count = (
            subtours[subtours.tour_type == tour_type].groupby("parent_tour_id").size()
        )
        # backfill with 0 count
        tour_counts[tour_type] = (
            tour_type_count.reindex(tour_counts.index).fillna(0).astype(np.int8)
        )

    # determine alt id corresponding to constrained_tour_counts
    # need to do index waltz because pd.merge doesn't preserve index in this case
    tour_counts = pd.merge(
        tour_counts.reset_index(),
        alts,
        left_on=tour_types,
        right_on=tour_types,
        how="left",
    ).set_index(tour_counts.index.name)

    atwork_subtour_frequency = tour_counts.alt

    # did we end up with any tour frequencies not in alts?
    if atwork_subtour_frequency.isna().any():
        bad_tour_frequencies = atwork_subtour_frequency.isna()
        logger.warning(
            "WARNING Bad atwork subtour frequencies for %s work tours"
            % bad_tour_frequencies.sum()
        )
        logger.warning(
            "WARNING Bad atwork subtour frequencies: num_tours\n%s"
            % tour_counts[bad_tour_frequencies]
        )
        logger.warning(
            "WARNING Bad atwork subtour frequencies: num_tours\n%s"
            % subtours[
                subtours.parent_tour_id.isin(tour_counts[bad_tour_frequencies].index)
            ].sort_values("parent_tour_id")
        )
        bug

    atwork_subtour_frequency = reindex(
        atwork_subtour_frequency, tours[ASIM_TOUR_ID]
    ).fillna("")

    return atwork_subtour_frequency


def patch_trip_ids(state: workflow.State, tours, trips):
    """
    replace survey trip_ids with asim standard trip_id
    replace survey tour_id foreign key with asim standard tour_id
    """

    # tour_id is a column, not index
    assert ASIM_TOUR_ID in tours

    # patch tour_id foreign key
    # tours['household_id'] = reindex(persons.household_id, tours.person_id)
    asim_tour_id = pd.Series(
        tours[ASIM_TOUR_ID].values, index=tours[SURVEY_TOUR_ID].values
    )
    trips[ASIM_TOUR_ID] = reindex(asim_tour_id, trips[SURVEY_TOUR_ID])

    # person_is_university = persons.pstudent == constants.PSTUDENT_UNIVERSITY
    # tour_is_university = reindex(person_is_university, tours.person_id)
    # tour_primary_purpose = tours.tour_type.where((tours.tour_type != 'school') | ~tour_is_university, 'univ')
    # tour_primary_purpose = tour_primary_purpose.where(tours.tour_category!='atwork', 'atwork')
    #
    # trips['primary_purpose'] = reindex(tour_primary_purpose, trips.tour_id)

    # if order is ambiguous if trips depart in same time slot - order by SURVEY_TRIP_ID hoping that increases with time
    if "trip_num" not in trips:
        trips["trip_num"] = (
            trips.sort_values(by=["tour_id", "outbound", "depart", SURVEY_TRIP_ID])
            .groupby(["tour_id", "outbound"])
            .cumcount()
            + 1
        )

    cid.set_trip_index(state, trips)

    assert trips.index.name == ASIM_TRIP_ID
    trips = trips.reset_index().rename(columns={"trip_id": ASIM_TRIP_ID})

    return trips


def infer_stop_frequency(configs_dir, tours, trips):
    # alt,out,in
    # 0out_0in,0,0
    # 0out_1in,0,1
    # ...
    alts = pd.read_csv(
        os.path.join(configs_dir, "stop_frequency_alternatives.csv"), comment="#"
    )
    assert "alt" in alts
    assert "in" in alts
    assert "out" in alts

    freq = pd.DataFrame(index=tours[SURVEY_TOUR_ID])

    # number of trips is one less than number of stops
    freq["out"] = trips[trips.outbound].groupby(SURVEY_TOUR_ID).trip_num.max() - 1
    freq["in"] = trips[~trips.outbound].groupby(SURVEY_TOUR_ID).trip_num.max() - 1

    freq = pd.merge(freq.reset_index(), alts, on=["out", "in"], how="left")

    assert (freq[SURVEY_TOUR_ID] == tours[SURVEY_TOUR_ID]).all()

    return freq.alt


def read_tables(input_dir, tables):
    for table, info in tables.items():
        table = pd.read_csv(
            os.path.join(input_dir, info["file_name"]), index_col=info.get("index")
        )
        # coerce missing data in string columns to empty strings, not NaNs
        for c in table.columns:
            # read_csv converts empty string to NaN, even if all non-empty values are strings
            if table[c].dtype == "object":
                print("##### converting", c, table[c].dtype)
                table[c] = table[c].fillna("").astype(str)
        info["table"] = table

    households = tables["households"].get("table")
    persons = tables["persons"].get("table")
    tours = tables["tours"].get("table")
    joint_tour_participants = tables["joint_tour_participants"].get("table")
    trips = tables["trips"].get("table")

    return households, persons, tours, joint_tour_participants, trips


def check_controls(table_name, column_name):
    table = survey_tables[table_name].get("table")
    c_table = control_tables[table_name].get("table")

    if column_name == "index":
        dont_match = table.index != c_table.index
    else:
        dont_match = table[column_name] != c_table[column_name]

    if dont_match.any():
        print(
            "check_controls %s.%s: %s out of %s do not match"
            % (table_name, column_name, dont_match.sum(), len(table))
        )
        print("control\n%s" % c_table[dont_match][[column_name]])
        print("survey\n%s" % table[dont_match][[column_name]])

        print("control\n%s" % c_table[dont_match][table.columns])
        print("survey\n%s" % table[dont_match][table.columns])
        return False

    return True


def infer(state: workflow.State, configs_dir, input_dir, output_dir):
    households, persons, tours, joint_tour_participants, trips = read_tables(
        input_dir, survey_tables
    )

    # be explicit about all tour_ids to avoid confusion between asim and survey ids
    tours = tours.rename(
        columns={"tour_id": SURVEY_TOUR_ID, "parent_tour_id": SURVEY_PARENT_TOUR_ID}
    )
    joint_tour_participants = joint_tour_participants.rename(
        columns={"tour_id": SURVEY_TOUR_ID, "participant_id": SURVEY_PARTICIPANT_ID}
    )
    trips = trips.rename(columns={"trip_id": SURVEY_TRIP_ID, "tour_id": SURVEY_TOUR_ID})

    # mangle survey tour ids to keep us honest
    tours[SURVEY_TOUR_ID] = mangle_ids(tours[SURVEY_TOUR_ID])
    tours[SURVEY_PARENT_TOUR_ID] = mangle_ids(tours[SURVEY_PARENT_TOUR_ID])
    joint_tour_participants[SURVEY_TOUR_ID] = mangle_ids(
        joint_tour_participants[SURVEY_TOUR_ID]
    )
    joint_tour_participants[SURVEY_PARTICIPANT_ID] = mangle_ids(
        joint_tour_participants[SURVEY_PARTICIPANT_ID]
    )
    trips[SURVEY_TRIP_ID] = mangle_ids(trips[SURVEY_TRIP_ID])
    trips[SURVEY_TOUR_ID] = mangle_ids(trips[SURVEY_TOUR_ID])

    # persons.cdap_activity
    persons["cdap_activity"] = infer_cdap_activity(
        persons, tours, joint_tour_participants
    )
    # check but don't assert as this is not deterministic
    skip_controls or check_controls("persons", "cdap_activity")

    # persons.mandatory_tour_frequency
    persons["mandatory_tour_frequency"] = infer_mandatory_tour_frequency(persons, tours)
    assert skip_controls or check_controls("persons", "mandatory_tour_frequency")

    # persons.non_mandatory_tour_frequency
    tour_frequency = infer_non_mandatory_tour_frequency(configs_dir, persons, tours)
    for c in tour_frequency.columns:
        print("assigning persons", c)
        persons[c] = tour_frequency[c]
    assert skip_controls or check_controls("persons", "non_mandatory_tour_frequency")

    # patch_tour_ids
    tours, joint_tour_participants = patch_tour_ids(
        state, persons, tours, joint_tour_participants
    )
    survey_tables["tours"]["table"] = tours
    survey_tables["joint_tour_participants"]["table"] = joint_tour_participants

    assert skip_controls or check_controls("tours", "index")
    assert skip_controls or check_controls("joint_tour_participants", "index")

    # patch_tour_ids
    trips = patch_trip_ids(state, tours, trips)
    survey_tables["trips"]["table"] = trips  # so we can check_controls
    assert skip_controls or check_controls("trips", "index")

    # households.joint_tour_frequency
    households["joint_tour_frequency"] = infer_joint_tour_frequency(
        configs_dir, households, tours
    )
    assert skip_controls or check_controls("households", "joint_tour_frequency")

    # tours.composition
    tours["composition"] = infer_joint_tour_composition(
        persons, tours, joint_tour_participants
    )
    assert skip_controls or check_controls("tours", "composition")

    # tours.tdd
    tours["tdd"] = infer_tour_scheduling(configs_dir, tours)
    assert skip_controls or check_controls("tours", "tdd")

    tours["atwork_subtour_frequency"] = infer_atwork_subtour_frequency(
        configs_dir, tours
    )
    assert skip_controls or check_controls("tours", "atwork_subtour_frequency")

    tours["stop_frequency"] = infer_stop_frequency(configs_dir, tours, trips)
    assert skip_controls or check_controls("tours", "stop_frequency")

    # write output files
    households.to_csv(os.path.join(output_dir, outputs["households"]), index=True)
    persons.to_csv(os.path.join(output_dir, outputs["persons"]), index=True)
    tours.to_csv(os.path.join(output_dir, outputs["tours"]), index=False)
    joint_tour_participants.to_csv(
        os.path.join(output_dir, outputs["joint_tour_participants"]), index=False
    )
    trips.to_csv(os.path.join(output_dir, outputs["trips"]), index=False)


# python infer.py data
args = sys.argv[1:]
assert len(args) == 3, "usage: python infer.py <data_dir> <configs_dir> <output_dir>"

data_dir = args[0]
configs_dir = args[1]
output_dir = args[2]

with open(os.path.join(configs_dir, "constants.yaml")) as stream:
    CONSTANTS = yaml.load(stream, Loader=yaml.SafeLoader)

input_dir = os.path.join(data_dir, "survey_data/")

if apply_controls:
    read_tables(input_dir, control_tables)

state = (
    workflow.State()
    .initialize_filesystem(
        configs_dir=(configs_dir,),
        output_dir=output_dir,
        data_dir=(data_dir,),
    )
    .load_settings()
)
infer(state, configs_dir, input_dir, output_dir)
