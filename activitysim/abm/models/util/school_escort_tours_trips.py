from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.school_escorting import NUM_ESCORTEES
from activitysim.abm.models.util import canonical_ids
from activitysim.core import workflow
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


def determine_chauf_outbound_flag(row, i):
    if row["school_escort_direction"] == "outbound":
        outbound = True
    elif (
        (row["school_escort_direction"] == "inbound")
        & (i == 0)
        & (row["escort_type"] == "pure_escort")
    ):
        # chauf is going to pick up the first child
        outbound = True
    else:
        # chauf is inbound and has already picked up a child or taken their mandatory tour
        outbound = False
    return outbound


def create_chauf_trip_table(row):
    dropoff = True if row["school_escort_direction"] == "outbound" else False

    row["person_id"] = row["chauf_id"]
    row["destination"] = row["school_destinations"].split("_")

    participants = []
    school_escort_trip_num = []
    outbound = []
    purposes = []

    for i, child_id in enumerate(row["escortees"].split("_")):
        if dropoff:
            # have the remaining children in car
            participants.append("_".join(row["escortees"].split("_")[i:]))
        else:
            # remaining children not yet in car
            participants.append("_".join(row["escortees"].split("_")[: i + 1]))
        school_escort_trip_num.append(i + 1)
        outbound.append(determine_chauf_outbound_flag(row, i))
        purposes.append("escort")

    if not dropoff:
        # adding trip home
        outbound.append(False)
        school_escort_trip_num.append(i + 2)
        purposes.append("home")
        row["destination"].append(row["home_zone_id"])
        # kids aren't in car until after they are picked up, inserting empty car for first trip
        participants = [""] + participants

    if dropoff & (row["escort_type"] == "ride_share"):
        # adding trip to work
        outbound.append(True)
        school_escort_trip_num.append(i + 2)
        purposes.append(row["first_mand_tour_purpose"])
        row["destination"].append(row["first_mand_tour_dest"])
        # kids have already been dropped off
        participants = participants + [""]

    row["escort_participants"] = participants
    row["school_escort_trip_num"] = school_escort_trip_num
    row["outbound"] = outbound
    row["purpose"] = purposes
    return row


def create_chauf_escort_trips(bundles):
    chauf_trip_bundles = bundles.apply(lambda row: create_chauf_trip_table(row), axis=1)
    chauf_trip_bundles["tour_id"] = bundles["chauf_tour_id"].astype(int)

    # departure time is the first school start in the outbound school_escort_direction and the last school end in the inbound school_escort_direction
    starts = (
        chauf_trip_bundles["school_starts"].str.split("_", expand=True).astype(float)
    )
    ends = chauf_trip_bundles["school_ends"].str.split("_", expand=True).astype(float)
    chauf_trip_bundles["depart"] = np.where(
        chauf_trip_bundles["school_escort_direction"] == "outbound",
        starts.min(axis=1),
        ends.max(axis=1),
    )

    # create a new trip for each escortee destination
    chauf_trips = chauf_trip_bundles.explode(
        [
            "destination",
            "escort_participants",
            "school_escort_trip_num",
            "outbound",
            "purpose",
        ]
    ).reset_index(drop=True)

    # numbering trips such that outbound escorting trips must come first and inbound trips must come last
    outbound_trip_num = -1 * (
        chauf_trips.groupby(["tour_id", "outbound"]).cumcount(ascending=False) + 1
    )
    inbound_trip_num = 100 + chauf_trips.groupby(["tour_id", "outbound"]).cumcount(
        ascending=True
    )
    chauf_trips["trip_num"] = np.where(
        chauf_trips.outbound == True, outbound_trip_num, inbound_trip_num
    )

    # --- determining trip origin
    # origin is previous destination
    chauf_trips["origin"] = chauf_trips.groupby("tour_id")["destination"].shift()
    # outbound trips start at home
    first_outbound_trips = (chauf_trips["outbound"] == True) & (
        chauf_trips["school_escort_trip_num"] == 1
    )
    chauf_trips.loc[first_outbound_trips, "origin"] = chauf_trips.loc[
        first_outbound_trips, "home_zone_id"
    ]
    # inbound school escort ride sharing trips start at work
    first_rs_inb = (
        (chauf_trips["outbound"] == False)
        & (chauf_trips["school_escort_trip_num"] == 1)
        & (chauf_trips["escort_type"] == "ride_share")
    )
    chauf_trips.loc[first_rs_inb, "origin"] = chauf_trips.loc[
        first_rs_inb, "first_mand_tour_dest"
    ]

    assert all(
        ~chauf_trips["origin"].isna()
    ), f"Missing trip origins for {chauf_trips[chauf_trips['origin'].isna()]}"

    chauf_trips["primary_purpose"] = np.where(
        chauf_trips["escort_type"] == "pure_escort",
        "escort",
        chauf_trips["first_mand_tour_purpose"],
    )
    assert all(
        ~chauf_trips["primary_purpose"].isna()
    ), f"Missing tour purpose for {chauf_trips[chauf_trips['primary_purpose'].isna()]}"

    chauf_trips.loc[
        chauf_trips["purpose"] == "home", "trip_num"
    ] = 999  # trips home are always last
    chauf_trips.sort_values(
        by=["household_id", "tour_id", "outbound", "trip_num"],
        ascending=[True, True, False, True],
        inplace=True,
    )

    return chauf_trips


def create_child_escorting_stops(row, escortee_num):
    escortees = row["escortees"].split("_")
    if escortee_num > (len(escortees) - 1):
        # this bundle does not have this many escortees
        return row
    dropoff = True if row["school_escort_direction"] == "outbound" else False

    row["person_id"] = int(escortees[escortee_num])
    row["tour_id"] = row["school_tour_ids"].split("_")[escortee_num]
    school_dests = row["school_destinations"].split("_")

    destinations = []
    purposes = []
    participants = []
    school_escort_trip_num = []

    escortee_order = (
        escortees[: escortee_num + 1] if dropoff else escortees[escortee_num:]
    )

    # for i, child_id in enumerate(escortees[:escortee_num+1]):
    for i, child_id in enumerate(escortee_order):
        is_last_stop = i == len(escortee_order) - 1

        if dropoff:
            # dropping childen off
            # children in car are the child and the children after
            participants.append("_".join(escortees[i:]))
            dest = school_dests[i]
            purpose = "school" if row["person_id"] == int(child_id) else "escort"

        else:
            # picking children up
            # children in car are the child and those already picked up
            participants.append("_".join(escortees[: escortee_num + i + 1]))
            # going home if last stop, otherwise to next school destination
            dest = (
                row["home_zone_id"]
                if is_last_stop
                else school_dests[escortee_num + i + 1]
            )
            purpose = "home" if is_last_stop else "escort"

        # filling arrays
        destinations.append(dest)
        school_escort_trip_num.append(i + 1)
        purposes.append(purpose)

    row["escort_participants"] = participants
    row["school_escort_trip_num"] = school_escort_trip_num
    row["purpose"] = purposes
    row["destination"] = destinations
    return row


def create_escortee_trips(bundles):
    escortee_trips = []
    for escortee_num in range(0, int(bundles.num_escortees.max()) + 1):
        escortee_bundles = bundles.apply(
            lambda row: create_child_escorting_stops(row, escortee_num), axis=1
        )
        escortee_trips.append(escortee_bundles)

    escortee_trips = pd.concat(escortee_trips)
    escortee_trips = escortee_trips[~escortee_trips.person_id.isna()]

    # departure time is the first school start in the outbound direction and the last school end in the inbound direction
    starts = escortee_trips["school_starts"].str.split("_", expand=True).astype(float)
    ends = escortee_trips["school_ends"].str.split("_", expand=True).astype(float)
    escortee_trips["outbound"] = np.where(
        escortee_trips["school_escort_direction"] == "outbound", True, False
    )
    escortee_trips["depart"] = np.where(
        escortee_trips["school_escort_direction"] == "outbound",
        starts.min(axis=1),
        ends.max(axis=1),
    ).astype(int)
    escortee_trips["primary_purpose"] = "school"

    # create a new trip for each escortee destination
    escortee_trips = escortee_trips.explode(
        ["destination", "escort_participants", "school_escort_trip_num", "purpose"]
    ).reset_index(drop=True)

    # numbering trips such that outbound escorting trips must come first and inbound trips must come last
    # this comes in handy when merging trips to others in the tour decided downstream
    outbound_trip_num = -1 * (
        escortee_trips.groupby(["tour_id", "outbound"]).cumcount(ascending=False) + 1
    )
    inbound_trip_num = 100 + escortee_trips.groupby(["tour_id", "outbound"]).cumcount(
        ascending=True
    )
    escortee_trips["trip_num"] = np.where(
        escortee_trips.outbound == True, outbound_trip_num, inbound_trip_num
    )
    escortee_trips["trip_count"] = escortee_trips["trip_num"] + escortee_trips.groupby(
        ["tour_id", "outbound"]
    ).trip_num.transform("count")

    id_cols = ["household_id", "person_id", "tour_id"]
    escortee_trips[id_cols] = escortee_trips[id_cols].astype("int64")

    escortee_trips.loc[
        escortee_trips["purpose"] == "home", "trip_num"
    ] = 999  # trips home are always last
    escortee_trips.sort_values(
        by=["household_id", "tour_id", "outbound", "trip_num"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    escortee_trips["origin"] = escortee_trips.groupby("tour_id")["destination"].shift()
    # first trips on tour start from home (except for atwork subtours, but school escorting doesn't happen on those tours)
    escortee_trips["origin"] = np.where(
        escortee_trips["origin"].isna(),
        escortee_trips["home_zone_id"],
        escortee_trips["origin"],
    )

    return escortee_trips


def create_school_escort_trips(escort_bundles):
    chauf_trips = create_chauf_escort_trips(escort_bundles)
    escortee_trips = create_escortee_trips(escort_bundles)
    school_escort_trips = pd.concat([chauf_trips, escortee_trips], axis=0)

    # Can't assign a true trip id yet because they are numbered based on the number of stops in each direction.
    # This isn't decided until after the stop frequency model runs.
    # Creating this temporary school_escort_trip_id column to match them downstream
    school_escort_trips["school_escort_trip_id"] = (
        school_escort_trips["tour_id"].astype("int64") * 10
        + school_escort_trips.groupby("tour_id")["trip_num"].cumcount()
    )

    return school_escort_trips


def add_pure_escort_tours(tours, school_escort_tours):
    missing_cols = [
        col for col in tours.columns if col not in school_escort_tours.columns
    ]
    assert (
        len(missing_cols) == 0
    ), f"missing columns {missing_cols} in school_escort_tours"
    if len(missing_cols) > 0:
        logger.warning(f"Columns {missing_cols} are missing from school escort tours")
        school_escort_tours[missing_cols] = pd.NA

    tours_to_add = school_escort_tours[~school_escort_tours.index.isin(tours.index)]
    tours = pd.concat([tours, tours_to_add[tours.columns]])

    return tours


def add_school_escorting_type_to_tours_table(escort_bundles, tours):
    school_tour = (tours.tour_type == "school") & (tours.tour_num == 1)

    escort_type_cat = pd.api.types.CategoricalDtype(
        ["pure_escort", "ride_share"], ordered=False
    )
    tours["school_esc_outbound"] = pd.NA
    tours["school_esc_inbound"] = pd.NA
    tours["school_esc_outbound"] = tours["school_esc_outbound"].astype(escort_type_cat)
    tours["school_esc_inbound"] = tours["school_esc_inbound"].astype(escort_type_cat)

    for school_escort_direction in ["outbound", "inbound"]:
        for escort_type in ["ride_share", "pure_escort"]:
            bundles = escort_bundles[
                (escort_bundles.school_escort_direction == school_escort_direction)
                & (escort_bundles.escort_type == escort_type)
            ]
            # Setting for child school tours
            for child_num in range(1, NUM_ESCORTEES + 1):
                i = str(child_num)
                filter = school_tour & tours["person_id"].isin(
                    bundles["bundle_child" + i]
                )
                tours.loc[filter, "school_esc_" + school_escort_direction] = escort_type

            tours.loc[
                bundles.chauf_tour_id, "school_esc_" + school_escort_direction
            ] = escort_type

    return tours


def process_tours_after_escorting_model(state: workflow.State, escort_bundles, tours):
    # adding indicators to tours that include school escorting
    tours = add_school_escorting_type_to_tours_table(escort_bundles, tours)

    # setting number of escortees on tour
    num_escortees = escort_bundles.drop_duplicates("chauf_tour_id").set_index(
        "chauf_tour_id"
    )["num_escortees"]
    tours.loc[num_escortees.index, "num_escortees"] = num_escortees

    # set same start / end time for tours if they are bundled together
    tour_segment_id_cols = [
        "school_tour_id_child" + str(i) for i in range(1, NUM_ESCORTEES + 1)
    ] + ["chauf_tour_id"]

    for id_col in tour_segment_id_cols:
        out_segment_bundles = escort_bundles[
            (escort_bundles[id_col] > 1)
            & (escort_bundles.school_escort_direction == "outbound")
        ].set_index(id_col)
        starts = (
            out_segment_bundles["school_starts"].str.split("_").str[0].astype(int)
        )  # first start
        tours.loc[starts.index, "start"] = starts

        inb_segment_bundles = escort_bundles[
            (escort_bundles[id_col] > 1)
            & (escort_bundles.school_escort_direction == "inbound")
        ].set_index(id_col)
        ends = (
            inb_segment_bundles["school_ends"].str.split("_").str[-1].astype(int)
        )  # last end
        tours.loc[ends.index, "end"] = ends

    bad_end_times = tours["start"] > tours["end"]
    tours.loc[bad_end_times, "end"] = tours.loc[bad_end_times, "start"]

    # updating tdd to match start and end times
    tdd_alts = state.get_injectable("tdd_alts")
    tdd_alts["tdd"] = tdd_alts.index
    tours.drop(columns="tdd", inplace=True)

    tours["tdd"] = pd.merge(
        tours.reset_index(), tdd_alts, how="left", on=["start", "end"]
    ).set_index("tour_id")["tdd"]
    # since this is an injectable, we want to leave it how we found it
    # not removing tdd created here will caues problems downstream
    tdd_alts.drop(columns="tdd", inplace=True)

    assert all(
        ~tours.tdd.isna()
    ), f"Tours have missing tdd values: {tours[tours.tdd.isna()][['tour_type', 'start', 'end', 'tdd']]}"

    return tours


def merge_school_escort_trips_into_pipeline(state: workflow.State):
    school_escort_trips = state.get_dataframe("school_escort_trips")
    tours = state.get_dataframe("tours")
    trips = state.get_dataframe("trips")

    # checking to see if there are school escort trips to merge in
    if len(school_escort_trips) == 0:
        # if no trips, fill escorting columns with NA
        trips[
            [
                "escort_participants",
                "school_escort_direction",
                "school_escort_trip_id",
            ]
        ] = pd.NA
        state.add_table("trips", trips)
        return trips

    # want to remove stops if school escorting takes place on that half tour so we can replace them with the actual stops
    out_se_tours = tours[
        tours["school_esc_outbound"].isin(["pure_escort", "ride_share"])
    ]
    inb_se_tours = tours[
        tours["school_esc_inbound"].isin(["pure_escort", "ride_share"])
    ]
    # removing outbound stops
    trips = trips[
        ~(trips.tour_id.isin(out_se_tours.index) & (trips["outbound"] == True))
    ]
    # removing inbound stops
    trips = trips[
        ~(trips.tour_id.isin(inb_se_tours.index) & (trips["outbound"] == False))
    ]

    # don't want to double count the non-escort half-tour of chauffeurs doing pure escort
    inb_chauf_pe_tours = tours[
        (tours["school_esc_inbound"] == "pure_escort")
        & (tours.primary_purpose == "escort")
    ]
    out_chauf_pe_tours = tours[
        (tours["school_esc_outbound"] == "pure_escort")
        & (tours.primary_purpose == "escort")
    ]
    school_escort_trips = school_escort_trips[
        ~(
            school_escort_trips.tour_id.isin(inb_chauf_pe_tours.index)
            & (school_escort_trips["outbound"] == True)
        )
    ]
    school_escort_trips = school_escort_trips[
        ~(
            school_escort_trips.tour_id.isin(out_chauf_pe_tours.index)
            & (school_escort_trips["outbound"] == False)
        )
    ]

    # for better merge with trips created in stop frequency
    school_escort_trips["failed"] = False

    # make sure the pandas categorical columns share the same categories before cancat
    # union categoricals
    for c in trips.columns.intersection(school_escort_trips.columns):
        if isinstance(trips[c].dtype, pd.api.types.CategoricalDtype):
            if isinstance(school_escort_trips[c].dtype, pd.api.types.CategoricalDtype):
                from pandas.api.types import union_categoricals

                uc = union_categoricals([trips[c], school_escort_trips[c]])
                trips[c] = pd.Categorical(trips[c], categories=uc.categories)
                school_escort_trips[c] = pd.Categorical(
                    school_escort_trips[c], categories=uc.categories
                )

    trips = pd.concat(
        [
            trips,
            school_escort_trips[
                list(trips.columns)
                + [
                    "escort_participants",
                    "school_escort_direction",
                    "school_escort_trip_id",
                ]
            ],
        ]
    )
    # sorting by escorting order as determined when creating the school escort trips
    trips.sort_values(
        by=["household_id", "tour_id", "outbound", "trip_num"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    grouped = trips.groupby(["tour_id", "outbound"])
    trips["trip_num"] = trips.groupby(["tour_id", "outbound"]).cumcount() + 1
    trips["trip_count"] = trips["trip_num"] + grouped.cumcount(ascending=False)

    # ensuring data types
    trips["outbound"] = trips["outbound"].astype(bool)
    trips["origin"] = trips["origin"].astype(int)
    trips["destination"] = trips["destination"].astype(int)

    # converting to categoricals
    trips["school_escort_direction"] = trips["school_escort_direction"].astype(
        pd.api.types.CategoricalDtype(["outbound", "inbound"], ordered=False)
    )
    # trips["escort_participants"] is left with dtype of object (i.e. Python strings)
    #  as it doesn't have a fixed number of categories

    # updating trip_id now that we have all trips
    trips = canonical_ids.set_trip_index(state, trips)
    school_escort_trip_id_map = {
        v: k
        for k, v in trips.loc[
            ~trips["school_escort_trip_id"].isna(), "school_escort_trip_id"
        ]
        .to_dict()
        .items()
    }

    school_escort_trips["trip_id"] = np.where(
        school_escort_trips["school_escort_trip_id"].isin(
            school_escort_trip_id_map.keys()
        ),
        school_escort_trips["school_escort_trip_id"].map(school_escort_trip_id_map),
        school_escort_trips["school_escort_trip_id"],
    )
    school_escort_trips.set_index("trip_id", inplace=True)

    # can drop school_escort_trip_id column now since it has been replaced
    trips.drop(columns="school_escort_trip_id", inplace=True)

    # replace trip table and pipeline and register with the random number generator
    state.add_table("trips", trips)
    state.get_rn_generator().drop_channel("trips")
    state.get_rn_generator().add_channel("trips", trips)
    state.add_table("school_escort_trips", school_escort_trips)

    # updating stop frequency in tours tabel to be consistent
    num_outbound_stops = (
        trips[trips.outbound == True].groupby("tour_id")["trip_num"].count() - 1
    )
    num_inbound_stops = (
        trips[trips.outbound == False].groupby("tour_id")["trip_num"].count() - 1
    )
    stop_freq = (
        num_outbound_stops.astype(str) + "out_" + num_inbound_stops.astype(str) + "in"
    )
    tours.loc[stop_freq.index, "stop_frequency"] = stop_freq

    # no need to reset random number generator since no tours added
    state.add_table("tours", tours)

    return trips


def recompute_tour_count_statistics(state: workflow.State):
    tours = state.get_dataframe("tours")

    grouped = tours.groupby(["person_id", "tour_type"])
    tours["tour_type_num"] = grouped.cumcount() + 1
    tours["tour_type_count"] = tours["tour_type_num"] + grouped.cumcount(
        ascending=False
    )

    grouped = tours.groupby("person_id")
    tours["tour_num"] = grouped.cumcount() + 1
    tours["tour_count"] = tours["tour_num"] + grouped.cumcount(ascending=False)

    # downcast
    tours["tour_count"] = tours["tour_count"].astype("int8")
    tours["tour_num"] = tours["tour_num"].astype("int8")
    tours["tour_type_num"] = tours["tour_type_num"].astype("int8")
    tours["tour_type_count"] = tours["tour_type_count"].astype("int8")

    state.add_table("tours", tours)


def create_pure_school_escort_tours(state: workflow.State, bundles):
    # creating home to school tour for chauffers making pure escort tours
    # ride share tours are already created since they go off the mandatory tour
    pe_tours = bundles[bundles["escort_type"] == "pure_escort"]

    pe_tours["origin"] = pe_tours["home_zone_id"]
    # desination is the last dropoff / pickup location
    pe_tours["destination"] = (
        pe_tours["school_destinations"].str.split("_").str[-1].astype(int)
    )
    # start is the first start time for outbound trips or the last school end time for inbound trips
    starts = pe_tours["school_starts"].str.split("_").str[0].astype(int)
    ends = pe_tours["school_ends"].str.split("_").str[-1].astype(int)
    pe_tours["start"] = np.where(
        pe_tours["school_escort_direction"] == "outbound", starts, ends
    )

    # just set end equal to start time -- non-escort half of tour is determined downstream
    pe_tours["end"] = pe_tours["start"]
    pe_tours["duration"] = pe_tours["end"] - pe_tours["start"]
    pe_tours["tdd"] = pd.NA  # updated with full tours table

    pe_tours["person_id"] = pe_tours["chauf_id"]

    pe_tours["tour_category"] = "non_mandatory"
    # convert tour category to categorical
    pe_tours["tour_category"] = pe_tours["tour_category"].astype(
        state.get_dataframe("tours").tour_category.dtype
    )
    pe_tours["number_of_participants"] = 1
    pe_tours["tour_type"] = "escort"
    # convert tour type to categorical
    pe_tours["tour_type"] = pe_tours["tour_type"].astype(
        state.get_dataframe("tours").tour_type.dtype
    )
    pe_tours["school_esc_outbound"] = np.where(
        pe_tours["school_escort_direction"] == "outbound", "pure_escort", pd.NA
    )
    pe_tours["school_esc_inbound"] = np.where(
        pe_tours["school_escort_direction"] == "inbound", "pure_escort", pd.NA
    )

    pe_tours = pe_tours.sort_values(by=["household_id", "person_id", "start"])

    # finding what the next start time for that person for scheduling
    pe_tours["next_pure_escort_start"] = (
        pe_tours.groupby("person_id")["start"].shift(-1).fillna(0)
    )

    grouped = pe_tours.groupby(["person_id", "tour_type"])
    pe_tours["tour_type_num"] = grouped.cumcount() + 1
    pe_tours["tour_type_count"] = pe_tours["tour_type_num"] + grouped.cumcount(
        ascending=False
    )

    grouped = pe_tours.groupby("person_id")
    pe_tours["tour_num"] = grouped.cumcount() + 1
    pe_tours["tour_count"] = pe_tours["tour_num"] + grouped.cumcount(ascending=False)

    pe_tours = canonical_ids.set_tour_index(state, pe_tours, is_school_escorting=True)

    return pe_tours


def split_out_school_escorting_trips(trips, school_escort_trips):
    # separate out school escorting trips to exclude them from the model
    full_trips_index = trips.index
    se_trips_mask = trips.index.isin(school_escort_trips.index)
    se_trips = trips[se_trips_mask]
    trips = trips[~se_trips_mask]

    return trips, se_trips, full_trips_index


def force_escortee_tour_modes_to_match_chauffeur(state: workflow.State, tours):
    # FIXME: escortee tour can have different chauffeur in outbound vs inbound direction
    # which tour mode should it be set to?  Currently it's whatever comes last.
    # Does it even matter if trip modes are getting matched later?
    escort_bundles = state.get_dataframe("escort_bundles")

    if len(escort_bundles) == 0:
        # do not need to do anything if no escorting
        return tours

    # grabbing the school tour ids for each school escort bundle
    se_tours = escort_bundles[["school_tour_ids", "chauf_tour_id"]].copy()
    # merging in chauffeur tour mode
    se_tours["tour_mode"] = reindex(tours.tour_mode, se_tours.chauf_tour_id)
    # creating entry for each escort school tour
    se_tours["school_tour_ids"] = se_tours.school_tour_ids.str.split("_")
    se_tours = se_tours.explode(["school_tour_ids"])
    # create mapping between school tour id and chauffeur tour mode
    se_tours["school_tour_ids"] = se_tours["school_tour_ids"].astype("int64")
    mode_mapping = se_tours.set_index("school_tour_ids")["tour_mode"]

    # setting escortee tours to have the same tour mode as chauffeur
    original_modes = tours["tour_mode"].copy()
    tours.loc[mode_mapping.index, "tour_mode"] = mode_mapping
    diff = tours["tour_mode"] != original_modes
    logger.info(
        f"Changed {diff.sum()} tour modes of school escortees to match their chauffeur"
    )

    assert (
        ~tours.tour_mode.isna()
    ).all(), f"Missing tour mode for {tours[tours.tour_mode.isna()]}"
    return tours


def force_escortee_trip_modes_to_match_chauffeur(state: workflow.State, trips):
    school_escort_trips = state.get_dataframe("school_escort_trips")

    # starting with only trips that are created as part of the school escorting model
    se_trips = trips[trips.index.isin(school_escort_trips.index)].copy()

    # getting chauffeur tour id
    se_trips["chauf_tour_id"] = reindex(
        school_escort_trips.chauf_tour_id, se_trips.index
    )
    # merging chauffeur trips onto escortee trips
    se_trips = (
        se_trips.reset_index()
        .merge(
            se_trips[
                [
                    "origin",
                    "destination",
                    "depart",
                    "escort_participants",
                    "chauf_tour_id",
                    "trip_mode",
                ]
            ],
            how="left",
            left_on=[
                "origin",
                "destination",
                "depart",
                "escort_participants",
                "tour_id",
            ],
            right_on=[
                "origin",
                "destination",
                "depart",
                "escort_participants",
                "chauf_tour_id",
            ],
            suffixes=("", "_chauf"),
        )
        .set_index("trip_id")
    )
    # trip_mode_chauf is na if the trip belongs to a chauffeur instead of an escortee
    # only want to change mode for escortees
    mode_mapping = se_trips[~se_trips["trip_mode_chauf"].isna()]["trip_mode_chauf"]

    # setting escortee trips to have the same trip mode as chauffeur
    original_modes = trips["trip_mode"].copy()
    trips.loc[mode_mapping.index, "trip_mode"] = mode_mapping
    diff = trips["trip_mode"] != original_modes
    logger.info(
        f"Changed {diff.sum()} trip modes of school escortees to match their chauffeur"
    )

    assert (
        ~trips.trip_mode.isna()
    ).all(), f"Missing trip mode for {trips[trips.trip_mode.isna()]}"
    return trips
