# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from activitysim.abm.models.util.trip import get_time_windows
from activitysim.core import (
    chunk,
    config,
    expressions,
    interaction_simulate,
    logit,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.skim_dataset import SkimDataset
from activitysim.core.skim_dictionary import SkimDict
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)

MAIN_LEG_DURATION = "main_leg_duration"
IB_DURATION = "inbound_duration"
OB_DURATION = "outbound_duration"

TOUR_ID = "tour_id"
TRIP_ID = "trip_id"
TOUR_LEG_ID = "tour_leg_id"
PATTERN_ID = "pattern_id"
TRIP_DURATION = "trip_duration"
STOP_TIME_DURATION = "stop_time_duration"
TRIP_NUM = "trip_num"
TRIP_COUNT = "trip_count"
OUTBOUND = "outbound"

MAX_TOUR_ID = int(1e9)


def generate_tour_leg_id(tour_leg_row):
    return tour_leg_row.tour_id + (
        int(MAX_TOUR_ID) if tour_leg_row.outbound else int(2 * MAX_TOUR_ID)
    )


def get_tour_legs(trips):
    tour_legs = trips.groupby([TOUR_ID, OUTBOUND], as_index=False)[TRIP_NUM].max()
    tour_legs[TOUR_LEG_ID] = tour_legs.apply(generate_tour_leg_id, axis=1)
    tour_legs = tour_legs.set_index(TOUR_LEG_ID)
    return tour_legs


# def trip_departure_rpc(chunk_size, choosers, trace_label):
#
#     # NOTE we chunk chunk_id
#     num_choosers = choosers['chunk_id'].max() + 1
#
#     chooser_row_size = choosers.shape[1] + 1
#
#     # scale row_size by average number of chooser rows per chunk_id
#     rows_per_chunk_id = choosers.shape[0] / num_choosers
#     row_size = (rows_per_chunk_id * chooser_row_size)
#
#     return chunk.rows_per_chunk(chunk_size, row_size, num_choosers, trace_label)


def generate_alternatives(trips, alternative_col_name):
    """
    This method creates an alternatives list of all possible
    trip durations less than the total trip leg duration. If
    the trip only has one trip on the leg, the trip alternative
    only has one alternative for that trip equal to the trip
    duration.
    :param trips: pd.DataFrame
    :param alternative_col_name: column name for the alternative column
    :return: pd.DataFrame
    """
    legs = trips[trips[TRIP_COUNT] > 1]

    leg_alts = None
    durations = np.where(legs[OUTBOUND], legs[OB_DURATION], legs[IB_DURATION])
    if len(durations) > 0:
        leg_alts = pd.Series(
            np.concatenate([np.arange(0, duration + 1) for duration in durations]),
            np.repeat(legs.index, durations + 1),
            name=alternative_col_name,
        ).to_frame()

    single_trips = trips[trips[TRIP_COUNT] == 1]
    single_alts = None
    durations = np.where(
        single_trips[OUTBOUND], single_trips[OB_DURATION], single_trips[IB_DURATION]
    )
    if len(durations) > 0:
        single_alts = pd.Series(
            durations, single_trips.index, name=alternative_col_name
        ).to_frame()

    if not legs.empty and not single_trips.empty:
        return pd.concat([leg_alts, single_alts])

    return leg_alts if not legs.empty else single_alts


def build_patterns(trips, time_windows):
    tours = trips.groupby([TOUR_ID])[[TRIP_DURATION, TRIP_COUNT]].first()
    duration_and_counts = tours[[TRIP_DURATION, TRIP_COUNT]].values

    # We subtract 1 here, because we already know
    # the one trip of the tour leg based on main tour
    # leg duration
    max_trip_count = trips[TRIP_COUNT].max() - 1

    patterns = []
    pattern_sizes = []

    for duration, trip_count in duration_and_counts:
        possible_windows = time_windows[
            : trip_count - 1,
            np.where(time_windows[: trip_count - 1].sum(axis=0) == duration)[0],
        ]
        possible_windows = np.unique(possible_windows, axis=1).transpose()
        filler = np.full((possible_windows.shape[0], max_trip_count), np.nan)
        filler[
            : possible_windows.shape[0], : possible_windows.shape[1]
        ] = possible_windows
        patterns.append(filler)
        pattern_sizes.append(filler.shape[0])

    patterns = np.concatenate(patterns)
    pattern_names = ["_".join("%0.0f" % x for x in y[~np.isnan(y)]) for y in patterns]
    indexes = np.repeat(tours.index, pattern_sizes)

    # If we've done everything right, the indexes
    # calculated above should be the same length as
    # the pattern options
    assert patterns.shape[0] == len(indexes)

    patterns = pd.DataFrame(index=indexes, data=patterns)
    patterns.index.name = tours.index.name
    patterns[PATTERN_ID] = pattern_names

    patterns = patterns.melt(
        id_vars=PATTERN_ID,
        value_name=STOP_TIME_DURATION,
        var_name=TRIP_NUM,
        ignore_index=False,
    ).reset_index()
    patterns = patterns[~patterns[STOP_TIME_DURATION].isnull()].copy()

    patterns[TRIP_NUM] = patterns[TRIP_NUM] + 1
    patterns[STOP_TIME_DURATION] = patterns[STOP_TIME_DURATION].astype(int)

    patterns = pd.merge(
        patterns,
        trips.reset_index()[[TOUR_ID, TRIP_ID, TRIP_NUM, OUTBOUND]],
        on=[TOUR_ID, TRIP_NUM],
    )

    patterns.index = patterns.apply(generate_tour_leg_id, axis=1)
    patterns.index.name = TOUR_LEG_ID

    return patterns


def get_spec_for_segment(omnibus_spec, segment):
    spec = omnibus_spec[[segment]]

    # might as well ignore any spec rows with 0 utility
    spec = spec[spec.iloc[:, 0] != 0]
    assert spec.shape[0] > 0

    return spec


def choose_tour_leg_pattern(
    state,
    trip_segment,
    patterns,
    spec,
    trace_label="trace_label",
    *,
    chunk_sizer: chunk.ChunkSizer,
):
    alternatives = generate_alternatives(trip_segment, STOP_TIME_DURATION).sort_index()
    have_trace_targets = state.tracing.has_trace_targets(trip_segment)

    if have_trace_targets:
        state.tracing.trace_df(
            trip_segment, tracing.extend_trace_label(trace_label, "choosers")
        )
        state.tracing.trace_df(
            alternatives,
            tracing.extend_trace_label(trace_label, "alternatives"),
            transpose=False,
        )

    if len(spec.columns) > 1:
        raise RuntimeError("spec must have only one column")

    # - join choosers and alts
    # in vanilla interaction_simulate interaction_df is cross join of choosers and alternatives
    # interaction_df = logit.interaction_dataset(choosers, alternatives, sample_size)
    # here, alternatives is sparsely repeated once for each (non-dup) sample
    # we expect alternatives to have same index of choosers (but with duplicate index values)
    # so we just need to left join alternatives with choosers
    assert alternatives.index.name == trip_segment.index.name

    interaction_df = alternatives.join(trip_segment, how="left", rsuffix="_chooser")

    chunk_sizer.log_df(trace_label, "interaction_df", interaction_df)

    if have_trace_targets:
        trace_rows, trace_ids = state.tracing.interaction_trace_rows(
            interaction_df, trip_segment
        )

        state.tracing.trace_df(
            interaction_df,
            tracing.extend_trace_label(trace_label, "interaction_df"),
            transpose=False,
        )
    else:
        trace_rows = trace_ids = None

    (
        interaction_utilities,
        trace_eval_results,
    ) = interaction_simulate.eval_interaction_utilities(
        state, spec, interaction_df, None, trace_label, trace_rows, estimator=None
    )

    interaction_utilities = pd.concat(
        [interaction_df[STOP_TIME_DURATION], interaction_utilities], axis=1
    )
    chunk_sizer.log_df(trace_label, "interaction_utilities", interaction_utilities)

    interaction_utilities = pd.merge(
        interaction_utilities.reset_index(),
        patterns[patterns[TRIP_ID].isin(interaction_utilities.index)],
        on=[TRIP_ID, STOP_TIME_DURATION],
        how="left",
    )

    if have_trace_targets:
        state.tracing.trace_interaction_eval_results(
            trace_eval_results,
            trace_ids,
            tracing.extend_trace_label(trace_label, "eval"),
        )

        state.tracing.trace_df(
            interaction_utilities,
            tracing.extend_trace_label(trace_label, "interaction_utilities"),
            transpose=False,
        )

    del interaction_df
    chunk_sizer.log_df(trace_label, "interaction_df", None)

    interaction_utilities = interaction_utilities.groupby(
        [TOUR_ID, OUTBOUND, PATTERN_ID], as_index=False
    )[["utility"]].sum()

    interaction_utilities[TOUR_LEG_ID] = interaction_utilities.apply(
        generate_tour_leg_id, axis=1
    )

    tour_choosers = interaction_utilities.set_index(TOUR_LEG_ID)
    interaction_utilities = tour_choosers[["utility"]].copy()

    # reshape utilities (one utility column and one row per row in model_design)
    # to a dataframe with one row per chooser and one column per alternative
    # interaction_utilities is sparse because duplicate sampled alternatives were dropped
    # so we need to pad with dummy utilities so low that they are never chosen

    # number of samples per chooser
    sample_counts = (
        interaction_utilities.groupby(interaction_utilities.index).size().values
    )
    chunk_sizer.log_df(trace_label, "sample_counts", sample_counts)

    # max number of alternatvies for any chooser
    max_sample_count = sample_counts.max()

    # offsets of the first and last rows of each chooser in sparse interaction_utilities
    last_row_offsets = sample_counts.cumsum()
    first_row_offsets = np.insert(last_row_offsets[:-1], 0, 0)

    # repeat the row offsets once for each dummy utility to insert
    # (we want to insert dummy utilities at the END of the list of alternative utilities)
    # inserts is a list of the indices at which we want to do the insertions
    inserts = np.repeat(last_row_offsets, max_sample_count - sample_counts)

    del sample_counts
    chunk_sizer.log_df(trace_label, "sample_counts", None)

    # insert the zero-prob utilities to pad each alternative set to same size
    padded_utilities = np.insert(interaction_utilities.utility.values, inserts, -999)
    del inserts

    del interaction_utilities
    chunk_sizer.log_df(trace_label, "interaction_utilities", None)

    # reshape to array with one row per chooser, one column per alternative
    padded_utilities = padded_utilities.reshape(-1, max_sample_count)
    chunk_sizer.log_df(trace_label, "padded_utilities", padded_utilities)

    # convert to a dataframe with one row per chooser and one column per alternative
    utilities_df = pd.DataFrame(padded_utilities, index=tour_choosers.index.unique())
    chunk_sizer.log_df(trace_label, "utilities_df", utilities_df)

    del padded_utilities
    chunk_sizer.log_df(trace_label, "padded_utilities", None)

    if have_trace_targets:
        state.tracing.trace_df(
            utilities_df,
            tracing.extend_trace_label(trace_label, "utilities"),
            column_labels=["alternative", "utility"],
        )

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(
        state, utilities_df, trace_label=trace_label, trace_choosers=trip_segment
    )

    chunk_sizer.log_df(trace_label, "probs", probs)

    del utilities_df
    chunk_sizer.log_df(trace_label, "utilities_df", None)

    if have_trace_targets:
        state.tracing.trace_df(
            probs,
            tracing.extend_trace_label(trace_label, "probs"),
            column_labels=["alternative", "probability"],
        )

    # make choices
    # positions is series with the chosen alternative represented as a column index in probs
    # which is an integer between zero and num alternatives in the alternative sample
    positions, rands = logit.make_choices(
        state, probs, trace_label=trace_label, trace_choosers=trip_segment
    )

    chunk_sizer.log_df(trace_label, "positions", positions)
    chunk_sizer.log_df(trace_label, "rands", rands)

    del probs
    chunk_sizer.log_df(trace_label, "probs", None)

    # shouldn't have chosen any of the dummy pad utilities
    assert positions.max() < max_sample_count

    # need to get from an integer offset into the alternative sample to the alternative index
    # that is, we want the index value of the row that is offset by <position> rows into the
    # tranche of this choosers alternatives created by cross join of alternatives and choosers

    # resulting pandas Int64Index has one element per chooser row and is in same order as choosers
    choices = tour_choosers[PATTERN_ID].take(positions + first_row_offsets)

    chunk_sizer.log_df(trace_label, "choices", choices)

    if have_trace_targets:
        state.tracing.trace_df(
            choices,
            tracing.extend_trace_label(trace_label, "choices"),
            columns=[None, PATTERN_ID],
        )
        state.tracing.trace_df(
            rands,
            tracing.extend_trace_label(trace_label, "rands"),
            columns=[None, "rand"],
        )

    return choices


def apply_stage_two_model(state, omnibus_spec, trips, chunk_size, trace_label):
    if not trips.index.is_monotonic_increasing:
        trips = trips.sort_index()

    # Assign the duration of the appropriate leg to the trip
    trips[TRIP_DURATION] = np.where(
        trips[OUTBOUND], trips[OB_DURATION], trips[IB_DURATION]
    )

    trips["depart"] = -1

    # If this is the first outbound trip, the choice is easy, assign the depart time
    # to equal the tour start time.
    trips.loc[(trips["trip_num"] == 1) & (trips[OUTBOUND]), "depart"] = trips["start"]

    # If its the first return leg, it is easy too. Just assign the trip start time to the
    # end time minus the IB duration
    trips.loc[(trips["trip_num"] == 1) & (~trips[OUTBOUND]), "depart"] = (
        trips["end"] - trips[IB_DURATION]
    )

    # The last leg of the outbound tour needs to begin at the start plus OB duration
    trips.loc[
        (trips["trip_count"] == trips["trip_num"]) & (trips[OUTBOUND]), "depart"
    ] = (trips["start"] + trips[OB_DURATION])

    # The last leg of the inbound tour needs to begin at the end time of the tour
    trips.loc[
        (trips["trip_count"] == trips["trip_num"]) & (~trips[OUTBOUND]), "depart"
    ] = trips["end"]

    # Slice off the remaining trips with an intermediate stops to deal with.
    # Hopefully, with the tricks above we've sliced off a lot of choices.
    # This slice should only include trip numbers greater than 2 since the
    side_trips = trips[
        (trips["trip_num"] != 1) & (trips["trip_count"] != trips["trip_num"])
    ]

    # No processing needs to be done because we have simple trips / tours
    if side_trips.empty:
        assert trips["depart"].notnull().all
        return trips["depart"].astype(int)

    # Get the potential time windows
    time_windows = get_time_windows(
        side_trips[TRIP_DURATION].max(), side_trips[TRIP_COUNT].max() - 1
    )

    trip_list = []

    for (
        i,
        chooser_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers_by_chunk_id(state, side_trips, trace_label):
        for is_outbound, trip_segment in chooser_chunk.groupby(OUTBOUND):
            direction = OUTBOUND if is_outbound else "inbound"
            spec = get_spec_for_segment(omnibus_spec, direction)
            segment_trace_label = "{}_{}".format(direction, chunk_trace_label)

            patterns = build_patterns(trip_segment, time_windows)

            choices = choose_tour_leg_pattern(
                state,
                trip_segment,
                patterns,
                spec,
                trace_label=segment_trace_label,
                chunk_sizer=chunk_sizer,
            )

            choices = pd.merge(
                choices.reset_index(),
                patterns.reset_index(),
                on=[TOUR_LEG_ID, PATTERN_ID],
                how="left",
            )

            choices = choices[["trip_id", "stop_time_duration"]].copy()

            trip_list.append(choices)

    trip_list = pd.concat(trip_list, sort=True).set_index("trip_id")
    trips["stop_time_duration"] = 0
    trips.update(trip_list)
    trips.loc[trips["trip_num"] == 1, "stop_time_duration"] = trips["depart"]
    trips.sort_values(["tour_id", "outbound", "trip_num"])
    trips["stop_time_duration"] = trips.groupby(["tour_id", "outbound"])[
        "stop_time_duration"
    ].cumsum()
    trips.loc[trips["trip_num"] != trips["trip_count"], "depart"] = trips[
        "stop_time_duration"
    ]
    return trips["depart"].astype(int)


class TripDepartureChoiceSettings(PydanticReadable, extra="forbid"):
    """
    Settings for the `trip_departure_choice` component.
    """

    PREPROCESSOR: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    SPECIFICATION: str = "trip_departure_choice.csv"
    """Filename for the trip departure choice (.csv) file."""

    CONSTANTS: dict[str, Any] = {}


@workflow.step
def trip_departure_choice(
    state: workflow.State,
    trips: pd.DataFrame,
    trips_merged: pd.DataFrame,
    skim_dict: SkimDict | SkimDataset,
    model_settings: TripDepartureChoiceSettings | None = None,
    model_settings_file_name: str = "trip_departure_choice.yaml",
    trace_label: str = "trip_departure_choice",
) -> None:

    if model_settings is None:
        model_settings = TripDepartureChoiceSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    spec = state.filesystem.read_model_spec(file_name=model_settings.SPECIFICATION)

    trips_merged_df = trips_merged
    # add tour-based chunk_id so we can chunk all trips in tour together
    tour_ids = trips_merged[TOUR_ID].unique()
    trips_merged_df["chunk_id"] = reindex(
        pd.Series(list(range(len(tour_ids))), tour_ids), trips_merged_df.tour_id
    )

    max_tour_id = trips_merged[TOUR_ID].max()

    trip_departure_choice.MAX_TOUR_ID = int(
        np.power(10, np.ceil(np.log10(max_tour_id)))
    )
    locals_d = config.get_model_constants(model_settings).copy()

    preprocessor_settings = model_settings.PREPROCESSOR
    tour_legs = get_tour_legs(trips_merged_df)
    state.get_rn_generator().add_channel("tour_legs", tour_legs)

    if preprocessor_settings:
        od_skim = skim_dict.wrap("origin", "destination")
        do_skim = skim_dict.wrap("destination", "origin")

        skims = [od_skim, do_skim]

        simulate.set_skim_wrapper_targets(trips_merged_df, skims)

        locals_d.update(
            {
                "od_skims": od_skim,
                "do_skims": do_skim,
            }
        )

        expressions.assign_columns(
            state,
            df=trips_merged_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    choices = apply_stage_two_model(
        state, spec, trips_merged_df, state.settings.chunk_size, trace_label
    )

    trips_df = trips
    trip_length = len(trips_df)
    trips_df = pd.concat([trips_df, choices], axis=1)
    assert len(trips_df) == trip_length
    assert trips_df[trips_df["depart"].isnull()].empty

    state.add_table("trips", trips_df)
