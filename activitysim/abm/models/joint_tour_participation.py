# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util.canonical_ids import MAX_PARTICIPANT_PNUM
from activitysim.abm.models.util.overlap import person_time_window_overlap
from activitysim.core import (
    config,
    estimation,
    expressions,
    logit,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings
from activitysim.core.configuration.logit import LogitComponentSettings
from activitysim.core.util import assign_in_place, reindex

logger = logging.getLogger(__name__)


def joint_tour_participation_candidates(joint_tours, persons_merged):
    # - only interested in persons from households with joint_tours
    persons_merged = persons_merged[persons_merged.num_hh_joint_tours > 0]

    # person_cols = ['household_id', 'PNUM', 'ptype', 'adult', 'travel_active']
    # household_cols = ['num_hh_joint_tours', 'home_is_urban', 'home_is_rural',
    #                   'car_sufficiency', 'income_in_thousands',
    #                   'num_adults', 'num_children', 'num_travel_active',
    #                   'num_travel_active_adults', 'num_travel_active_children']
    # persons_merged = persons_merged[person_cols + household_cols]

    # - create candidates table
    candidates = pd.merge(
        joint_tours.reset_index().rename(columns={"person_id": "point_person_id"}),
        persons_merged.reset_index().rename(
            columns={persons_merged.index.name: "person_id"}
        ),
        left_on=["household_id"],
        right_on=["household_id"],
    )

    # should have all joint_tours
    assert len(candidates["tour_id"].unique()) == joint_tours.shape[0]

    # - filter out ineligible candidates (adults for children-only tours, and vice-versa)
    eligible = ~(
        ((candidates.composition == "adults") & ~candidates.adult)
        | ((candidates.composition == "children") & candidates.adult)
    )
    candidates = candidates[eligible]

    # - stable (predictable) index
    # if this happens, participant_id may not be unique
    # channel random seeds will overlap at MAX_PARTICIPANT_PNUM (probably not a big deal)
    # and estimation infer will fail
    assert (
        candidates.PNUM.max() < MAX_PARTICIPANT_PNUM
    ), f"max persons.PNUM ({candidates.PNUM.max()}) > MAX_PARTICIPANT_PNUM ({MAX_PARTICIPANT_PNUM})"
    candidates["participant_id"] = (
        candidates[joint_tours.index.name] * MAX_PARTICIPANT_PNUM
    ) + candidates.PNUM
    candidates["participant_id"] = candidates["participant_id"].astype(np.int64)
    candidates.set_index(
        "participant_id", drop=True, inplace=True, verify_integrity=True
    )

    return candidates


def get_tour_satisfaction(candidates, participate):
    tour_ids = candidates.tour_id.unique()

    if participate.any():
        candidates = candidates[participate]

        # if this happens, we would need to filter them out!
        assert not ((candidates.composition == "adults") & ~candidates.adult).any()
        assert not ((candidates.composition == "children") & candidates.adult).any()

        # FIXME tour satisfaction - hack
        # annotate_households_cdap.csv says there has to be at least one non-preschooler in household
        # so presumably there also has to be at least one non-preschooler in joint tour
        # participates_in_jtf_model,(num_travel_active > 1) & (num_travel_active_non_preschoolers > 0)
        cols = ["tour_id", "composition", "adult", "person_is_preschool"]

        x = (
            candidates[cols]
            .groupby(["tour_id", "composition"], observed=True)
            .agg(
                participants=("adult", "size"),
                adults=("adult", "sum"),
                preschoolers=("person_is_preschool", "sum"),
            )
            .reset_index("composition")
        )

        # satisfaction = \
        #     (x.composition == 'adults') & (x.participants > 1) | \
        #     (x.composition == 'children') & (x.participants > 1) & (x.preschoolers < x.participants) | \
        #     (x.composition == 'mixed') & (x.adults > 0) & (x.participants > x.adults)

        satisfaction = (x.composition != "mixed") & (x.participants > 1) | (
            x.composition == "mixed"
        ) & (x.adults > 0) & (x.participants > x.adults)

        satisfaction = satisfaction.reindex(tour_ids).fillna(False).astype(bool)

    else:
        satisfaction = pd.Series(dtype=bool)

    # ensure we return a result for every joint tour, even if no participants
    satisfaction = satisfaction.reindex(tour_ids).fillna(False).astype(bool)

    return satisfaction


def participants_chooser(
    state: workflow.State,
    probs: pd.DataFrame,
    choosers: pd.DataFrame,
    spec: pd.DataFrame,
    trace_label: str,
) -> tuple[pd.Series, pd.Series]:
    """
    custom alternative to logit.make_choices for simulate.simple_simulate

    Choosing participants for mixed tours is trickier than adult or child tours becuase we
    need at least one adult and one child participant in a mixed tour. We call logit.make_choices
    and then check to see if the tour statisfies this requirement, and rechoose for any that
    fail until all are satisfied.

    In principal, this shold always occur eventually, but we fail after MAX_ITERATIONS,
    just in case there is some failure in program logic (haven't seen this occur.)

    The return values are the same as logit.make_choices

    Parameters
    ----------
    probs : pandas.DataFrame
        Rows for choosers and columns for the alternatives from which they
        are choosing. Values are expected to be valid probabilities across
        each row, e.g. they should sum to 1.
    choosers : pandas.dataframe
        simple_simulate choosers df
    spec : pandas.DataFrame
        simple_simulate spec df
        We only need spec so we can know the column index of the 'participate' alternative
        indicating that the participant has been chosen to participate in the tour
    trace_label : str

    Returns
    -------
    choices, rands
        choices, rands as returned by logit.make_choices (in same order as probs)

    """

    assert probs.index.equals(choosers.index)

    # choice is boolean (participate or not)
    model_settings = JointTourParticipationSettings.read_settings_file(
        state.filesystem, "joint_tour_participation.yaml", mandatory=False
    )

    choice_col = model_settings.participation_choice
    assert (
        choice_col in spec.columns
    ), "couldn't find participation choice column '%s' in spec"
    PARTICIPATE_CHOICE = spec.columns.get_loc(choice_col)
    MAX_ITERATIONS = model_settings.max_participation_choice_iterations

    trace_label = tracing.extend_trace_label(trace_label, "participants_chooser")

    candidates = choosers.copy()
    choices_list = []
    rands_list = []

    num_tours_remaining = len(candidates.tour_id.unique())
    logger.info(
        "%s %s joint tours to satisfy.",
        trace_label,
        num_tours_remaining,
    )

    iter = 0
    while candidates.shape[0] > 0:
        iter += 1

        if iter > MAX_ITERATIONS:
            logger.warning(
                "%s max iterations exceeded (%s).", trace_label, MAX_ITERATIONS
            )
            diagnostic_cols = ["tour_id", "household_id", "composition", "adult"]
            unsatisfied_candidates = candidates[diagnostic_cols].join(probs)
            state.tracing.write_csv(
                unsatisfied_candidates,
                file_name="%s.UNSATISFIED" % trace_label,
                transpose=False,
            )
            print(unsatisfied_candidates.head(20))

            if model_settings.FORCE_PARTICIPATION:
                logger.warning(
                    f"Forcing joint tour participation for {num_tours_remaining} tours."
                )
                # anybody with probability > 0 is forced to join the joint tour
                probs[choice_col] = np.where(probs[choice_col] > 0, 1, 0)
                non_choice_col = [col for col in probs.columns if col != choice_col][0]
                probs[non_choice_col] = 1 - probs[choice_col]
                if iter > MAX_ITERATIONS + 1:
                    raise RuntimeError(
                        f"{num_tours_remaining} tours could not be satisfied even with forcing participation"
                    )
            else:
                raise RuntimeError(
                    f"{num_tours_remaining} tours could not be satisfied after {iter} iterations"
                )

        choices, rands = logit.make_choices(
            state, probs, trace_label=trace_label, trace_choosers=choosers
        )
        participate = choices == PARTICIPATE_CHOICE

        # satisfaction indexed by tour_id
        tour_satisfaction = get_tour_satisfaction(candidates, participate)
        num_tours_satisfied_this_iter = tour_satisfaction.sum()

        if (iter > MAX_ITERATIONS) & (
            num_tours_remaining != num_tours_satisfied_this_iter
        ):
            logger.error(
                f"Still do not satisfy participation for {num_tours_remaining - num_tours_satisfied_this_iter} tours."
            )

        if num_tours_satisfied_this_iter > 0:
            num_tours_remaining -= num_tours_satisfied_this_iter

            satisfied = reindex(tour_satisfaction, candidates.tour_id)

            choices_list.append(choices[satisfied])
            rands_list.append(rands[satisfied])

            # remove candidates of satisfied tours
            probs = probs[~satisfied]
            candidates = candidates[~satisfied]

        logger.debug(
            f"{trace_label} iteration {iter} : "
            f"{num_tours_satisfied_this_iter} joint tours satisfied {num_tours_remaining} remaining"
        )

    choices = pd.concat(choices_list)
    rands = pd.concat(rands_list).reindex(choosers.index)

    # reindex choices and rands to match probs and v index
    choices = choices.reindex(choosers.index)
    rands = rands.reindex(choosers.index)
    assert choices.index.equals(choosers.index)
    assert rands.index.equals(choosers.index)

    logger.info(
        "%s %s iterations to satisfy all joint tours.",
        trace_label,
        iter,
    )

    return choices, rands


def annotate_jtp(
    state: workflow.State,
    model_settings: JointTourParticipationSettings,
    trace_label: str,
):
    # - annotate persons
    persons = state.get_dataframe("persons")
    expressions.assign_columns(
        state,
        df=persons,
        model_settings=model_settings.annotate_persons,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
    )
    state.add_table("persons", persons)


def add_null_results(
    state: workflow.State,
    model_settings: JointTourParticipationSettings,
    trace_label: str,
):
    logger.info("Skipping %s: joint tours", trace_label)
    # participants table is used downstream in non-joint tour expressions

    PARTICIPANT_COLS = ["tour_id", "household_id", "person_id", "participant_num"]

    participants = pd.DataFrame(columns=PARTICIPANT_COLS)
    participants.index.name = "participant_id"
    state.add_table("joint_tour_participants", participants)

    # - run annotations
    annotate_jtp(state, model_settings, trace_label)


class JointTourParticipationSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `joint_tour_participation` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    annotate_persons: PreprocessorSettings | None = None
    """Instructions for annotating the persons table."""

    participation_choice: str = "participate"

    max_participation_choice_iterations: int = 5000

    FORCE_PARTICIPATION: bool = False


@workflow.step
def joint_tour_participation(
    state: workflow.State,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    model_settings: JointTourParticipationSettings | None = None,
    model_settings_file_name: str = "joint_tour_participation.yaml",
    trace_label: str = "joint_tour_participation",
) -> None:
    """
    Predicts for each eligible person to participate or not participate in each joint tour.
    """

    if model_settings is None:
        model_settings = JointTourParticipationSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )
    trace_hh_id = state.settings.trace_hh_id

    joint_tours = tours[tours.tour_category == "joint"]

    # - if no joint tours
    if joint_tours.shape[0] == 0:
        add_null_results(state, model_settings, trace_label)
        return

    # - create joint_tour_participation_candidates table
    candidates = joint_tour_participation_candidates(joint_tours, persons_merged)
    state.tracing.register_traceable_table("joint_tour_participants", candidates)
    state.get_rn_generator().add_channel("joint_tour_participants", candidates)

    logger.info(
        "Running joint_tours_participation with %d potential participants (candidates)"
        % candidates.shape[0]
    )

    # - preprocessor
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        locals_dict = {
            "person_time_window_overlap": lambda x: person_time_window_overlap(
                state, x
            ),
            "persons": persons_merged,
        }

        expressions.assign_columns(
            state,
            df=candidates,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    # - simple_simulate

    estimator = estimation.manager.begin_estimation(state, "joint_tour_participation")

    model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(candidates)

    # add tour-based chunk_id so we can chunk all trips in tour together
    assert "chunk_id" not in candidates.columns
    unique_household_ids = candidates.household_id.unique()
    household_chunk_ids = pd.Series(
        range(len(unique_household_ids)), index=unique_household_ids
    )
    candidates["chunk_id"] = reindex(household_chunk_ids, candidates.household_id)

    choices = simulate.simple_simulate_by_chunk_id(
        state,
        choosers=candidates,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="participation",
        custom_chooser=participants_chooser,
        estimator=estimator,
        overflow_protection=model_settings.overflow_protection,
    )

    # choice is boolean (participate or not)
    choice_col = model_settings.participation_choice
    assert (
        choice_col in model_spec.columns
    ), "couldn't find participation choice column '%s' in spec"
    PARTICIPATE_CHOICE = model_spec.columns.get_loc(choice_col)

    participate = choices == PARTICIPATE_CHOICE

    if estimator:
        estimator.write_choices(choices)

        # we override the 'participate' boolean series, instead of raw alternative index in 'choices' series
        # its value depends on whether the candidate's 'participant_id' is in the joint_tour_participant index
        survey_participants_df = estimator.get_survey_table("joint_tour_participants")
        participate = pd.Series(
            choices.index.isin(survey_participants_df.index.values), index=choices.index
        )

        # but estimation software wants to know the choices value (alternative index)
        choices = participate.replace(
            {True: PARTICIPATE_CHOICE, False: 1 - PARTICIPATE_CHOICE}
        )
        # estimator.write_override_choices(participate)  # write choices as boolean participate
        estimator.write_override_choices(choices)  # write choices as int alt indexes

        estimator.end_estimation()

    # satisfaction indexed by tour_id
    tour_satisfaction = get_tour_satisfaction(candidates, participate)

    assert tour_satisfaction.all()

    candidates["satisfied"] = reindex(tour_satisfaction, candidates.tour_id)

    PARTICIPANT_COLS = ["tour_id", "household_id", "person_id"]
    participants = candidates[participate][PARTICIPANT_COLS].copy()

    # assign participant_num
    # FIXME do we want something smarter than the participant with the lowest person_id?
    participants["participant_num"] = (
        participants.sort_values(by=["tour_id", "person_id"])
        .groupby("tour_id")
        .cumcount()
        + 1
    )

    state.add_table("joint_tour_participants", participants)

    # drop channel as we aren't using any more (and it has candidates that weren't chosen)
    state.get_rn_generator().drop_channel("joint_tour_participants")

    # - assign joint tour 'point person' (participant_num == 1)
    point_persons = participants[participants.participant_num == 1]
    joint_tours["person_id"] = point_persons.set_index("tour_id").person_id

    # update number_of_participants which was initialized to 1
    joint_tours["number_of_participants"] = participants.groupby("tour_id").size()

    assign_in_place(
        tours,
        joint_tours[["person_id", "number_of_participants"]],
        state.settings.downcast_int,
        state.settings.downcast_float,
    )

    state.add_table("tours", tours)

    # - run annotations
    annotate_jtp(state, model_settings, trace_label)

    if trace_hh_id:
        state.tracing.trace_df(
            participants, label="joint_tour_participation.participants"
        )

        state.tracing.trace_df(
            joint_tours, label="joint_tour_participation.joint_tours"
        )
