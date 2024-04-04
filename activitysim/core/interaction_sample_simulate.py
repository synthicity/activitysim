# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import chunk, interaction_simulate, logit, tracing, workflow
from activitysim.core.simulate import set_skim_wrapper_targets

logger = logging.getLogger(__name__)


def _interaction_sample_simulate(
    state: workflow.State,
    choosers,
    alternatives,
    spec,
    choice_column,
    allow_zero_probs,
    zero_prob_choice_val,
    log_alt_losers,
    want_logsums,
    skims,
    locals_d,
    trace_label,
    trace_choice_name,
    estimator,
    skip_choice=False,
    *,
    chunk_sizer: chunk.ChunkSizer,
    overflow_protection: bool | None = None,
):
    """
    Run a MNL simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    Parameters are same as for public function interaction_sample_simulate

    spec : dataframe
        one row per spec expression and one col with utility coefficient

    interaction_df : dataframe
        cross join (cartesian product) of choosers with alternatives
        combines columns of choosers and alternatives
        len(df) == len(choosers) * len(alternatives)
        index values (non-unique) are index values from alternatives df

    interaction_utilities : dataframe
        the utility of each alternative is sum of the partial utilities determined by the
        various spec expressions and their corresponding coefficients
        yielding a dataframe  with len(interaction_df) rows and one utility column
        having the same index as interaction_df (non-unique values from alternatives df)

    utilities : dataframe
        dot product of model_design.dot(spec)
        yields utility value for element in the cross product of choosers and alternatives
        this is then reshaped as a dataframe with one row per chooser and one column per alternative

    probs : dataframe
        utilities exponentiated and converted to probabilities
        same shape as utilities, one row per chooser and one column for alternative

    positions : series
        choices among alternatives with the chosen alternative represented
        as the integer index of the selected alternative column in probs

    choices : series
        series with the alternative chosen for each chooser
        the index is same as choosers
        and the series value is the alternative df index of chosen alternative

    overflow_protection : bool, optional
        If True, use overflow-protected exponentiation and normalization to
        compute probabilities, which will help prevent problems if the utility
        values are very large in magnitude (positive or negative).  If False,
        overflow-protected exponentiation and normalization will not be used,
        and the model might fail if the utility values are very large in magnitude.
        If None (the default), overflow-protection is turned on if
        `allow_zero_probs` is False, and left off otherwise.

    Returns
    -------
    if want_logsums is False:

        choices : pandas.Series
            A series where index should match the index of the choosers DataFrame
            and values will match the index of the alternatives DataFrame -
            choices are simulated in the standard Monte Carlo fashion

    if want_logsums is True:

        choices : pandas.DataFrame
            choices['choice'] : same as choices series when logsums is False
            choices['logsum'] : float logsum of choice utilities across alternatives
    """

    # merge of alternatives, choosers on index requires increasing index
    assert choosers.index.is_monotonic_increasing
    assert alternatives.index.is_monotonic_increasing

    # assert choosers.index.equals(alternatives.index[~alternatives.index.duplicated(keep='first')])

    # this is the more general check (not requiring is_monotonic_increasing)
    last_repeat = alternatives.index != np.roll(alternatives.index, -1)
    assert (choosers.shape[0] == 1) or choosers.index.equals(
        alternatives.index[last_repeat]
    )

    have_trace_targets = state.tracing.has_trace_targets(choosers)

    if have_trace_targets:
        state.tracing.trace_df(
            choosers, tracing.extend_trace_label(trace_label, "choosers")
        )
        state.tracing.trace_df(
            alternatives,
            tracing.extend_trace_label(trace_label, "alternatives"),
            transpose=False,
        )

    if len(spec.columns) > 1:
        raise RuntimeError("spec must have only one column")

    # if using skims, copy index into the dataframe, so it will be
    # available as the "destination" for the skims dereference below
    # if skims is not None and alternatives.index.name not in alternatives:
    #    #FIXME - not needed ?
    #    alternatives = alternatives.copy()
    #    alternatives[alternatives.index.name] = alternatives.index

    # - join choosers and alts
    # in vanilla interaction_simulate interaction_df is cross join of choosers and alternatives
    # interaction_df = logit.interaction_dataset(choosers, alternatives, sample_size)
    # here, alternatives is sparsely repeated once for each (non-dup) sample
    # we expect alternatives to have same index of choosers (but with duplicate index values)
    # so we just need to left join alternatives with choosers

    # assert alternatives.index.name == choosers.index.name
    # asserting the index names are the same tells us nothing about the underlying data so why?
    logger.info(
        f"{trace_label} start merging choosers and alternatives to create interaction_df"
    )
    interaction_df = alternatives.join(choosers, how="left", rsuffix="_chooser")
    logger.info(
        f"{trace_label} end merging choosers and alternatives to create interaction_df"
    )

    if log_alt_losers:
        # logit.interaction_dataset adds ALT_CHOOSER_ID column if log_alt_losers is True
        # to enable detection of zero_prob-driving utils (e.g. -999 for all alts in a chooser)
        interaction_df[
            interaction_simulate.ALT_CHOOSER_ID
        ] = interaction_df.index.values

    chunk_sizer.log_df(trace_label, "interaction_df", interaction_df)

    if have_trace_targets:
        trace_rows, trace_ids = state.tracing.interaction_trace_rows(
            interaction_df, choosers
        )

        state.tracing.trace_df(
            interaction_df,
            tracing.extend_trace_label(trace_label, "interaction_df"),
            transpose=False,
        )
    else:
        trace_rows = trace_ids = None

    if skims is not None:
        set_skim_wrapper_targets(interaction_df, skims)

    # evaluate expressions from the spec multiply by coefficients and sum
    # spec is df with one row per spec expression and one col with utility coefficient
    # column names of choosers match spec index values
    # utilities has utility value for element in the cross product of choosers and alternatives
    # interaction_utilities is a df with one utility column and one row per row in alternative
    (
        interaction_utilities,
        trace_eval_results,
    ) = interaction_simulate.eval_interaction_utilities(
        state,
        spec,
        interaction_df,
        locals_d,
        trace_label,
        trace_rows,
        estimator=estimator,
        log_alt_losers=log_alt_losers,
    )
    chunk_sizer.log_df(trace_label, "interaction_utilities", interaction_utilities)

    del interaction_df
    chunk_sizer.log_df(trace_label, "interaction_df", None)

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
    chunk_sizer.log_df(trace_label, "padded_utilities", padded_utilities)
    del inserts

    del interaction_utilities
    chunk_sizer.log_df(trace_label, "interaction_utilities", None)

    # reshape to array with one row per chooser, one column per alternative
    padded_utilities = padded_utilities.reshape(-1, max_sample_count)

    # convert to a dataframe with one row per chooser and one column per alternative
    utilities_df = pd.DataFrame(padded_utilities, index=choosers.index)
    chunk_sizer.log_df(trace_label, "utilities_df", utilities_df)

    del padded_utilities
    chunk_sizer.log_df(trace_label, "padded_utilities", None)

    if have_trace_targets:
        state.tracing.trace_df(
            utilities_df,
            tracing.extend_trace_label(trace_label, "utilities"),
            column_labels=["alternative", "utility"],
        )

    if overflow_protection is None:
        overflow_protection = not allow_zero_probs

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    if want_logsums:
        probs, logsums = logit.utils_to_probs(
            state,
            utilities_df,
            allow_zero_probs=allow_zero_probs,
            trace_label=trace_label,
            trace_choosers=choosers,
            overflow_protection=overflow_protection,
            return_logsums=True,
        )
        chunk_sizer.log_df(trace_label, "logsums", logsums)
    else:
        probs = logit.utils_to_probs(
            state,
            utilities_df,
            allow_zero_probs=allow_zero_probs,
            trace_label=trace_label,
            trace_choosers=choosers,
            overflow_protection=overflow_protection,
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

    if allow_zero_probs:
        zero_probs = probs.sum(axis=1) == 0
        if zero_probs.any():
            # FIXME this is kind of gnarly, but we force choice of first alt
            probs.loc[zero_probs, 0] = 1.0

    if skip_choice:
        return choosers.join(logsums.to_frame("logsums"))

    else:
        # make choices
        # positions is series with the chosen alternative represented as a column index in probs
        # which is an integer between zero and num alternatives in the alternative sample
        positions, rands = logit.make_choices(
            state, probs, trace_label=trace_label, trace_choosers=choosers
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
        choices = alternatives[choice_column].take(positions + first_row_offsets)

        # create a series with index from choosers and the index of the chosen alternative
        choices = pd.Series(choices, index=choosers.index)

        chunk_sizer.log_df(trace_label, "choices", choices)

        if allow_zero_probs and zero_probs.any() and zero_prob_choice_val is not None:
            # FIXME this is kind of gnarly, patch choice for zero_probs
            choices.loc[zero_probs] = zero_prob_choice_val

        if have_trace_targets:
            state.tracing.trace_df(
                choices,
                tracing.extend_trace_label(trace_label, "choices"),
                columns=[None, trace_choice_name],
            )
            state.tracing.trace_df(
                rands,
                tracing.extend_trace_label(trace_label, "rands"),
                columns=[None, "rand"],
            )
            if want_logsums:
                state.tracing.trace_df(
                    logsums,
                    tracing.extend_trace_label(trace_label, "logsum"),
                    columns=[None, "logsum"],
                )

        if want_logsums:
            choices = choices.to_frame("choice")
            choices["logsum"] = logsums

        chunk_sizer.log_df(trace_label, "choices", choices)

        # handing this off to our caller
        chunk_sizer.log_df(trace_label, "choices", None)

        return choices


def interaction_sample_simulate(
    state: workflow.State,
    choosers,
    alternatives,
    spec,
    choice_column,
    allow_zero_probs=False,
    zero_prob_choice_val=None,
    log_alt_losers=False,
    want_logsums=False,
    skims=None,
    locals_d=None,
    chunk_size=0,
    chunk_tag=None,
    trace_label=None,
    trace_choice_name=None,
    estimator=None,
    skip_choice=False,
):
    """
    Run a simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    optionally (if chunk_size > 0) iterates over choosers in chunk_size chunks

    Parameters
    ----------
    choosers : pandas.DataFrame
        DataFrame of choosers
    alternatives : pandas.DataFrame
        DataFrame of alternatives - will be merged with choosers
        index domain same as choosers, but repeated for each alternative
    spec : pandas.DataFrame
        A Pandas DataFrame that gives the specification of the variables to
        compute and the coefficients for each variable.
        Variable specifications must be in the table index and the
        table should have only one column of coefficients.
    skims : Skims object
        The skims object is used to contain multiple matrices of
        origin-destination impedances.  Make sure to also add it to the
        locals_d below in order to access it in expressions.  The *only* job
        of this method in regards to skims is to call set_df with the
        dataframe that comes back from interacting choosers with
        alternatives.  See the skims module for more documentation on how
        the skims object is intended to be used.
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
    chunk_size : int
        if chunk_size > 0 iterates over choosers in chunk_size chunks
    trace_label: str
        This is the label to be used  for trace log file entries and dump file names
        when household tracing enabled. No tracing occurs if label is empty or None.
    trace_choice_name: str
        This is the column label to be used in trace file csv dump of choices
    skip_choice: bool
        This skips the logit choice step and simply returns the alternatives table with logsums
        (used in disaggregate accessibility)

    Returns
    -------
    if want_logsums is False:

        choices : pandas.Series
            A series where index should match the index of the choosers DataFrame
            and values will match the index of the alternatives DataFrame -
            choices are simulated in the standard Monte Carlo fashion

    if want_logsums is True:

        choices : pandas.DataFrame
            choices['choice'] : same as choices series when logsums is False
            choices['logsum'] : float logsum of choice utilities across alternatives

    """

    trace_label = tracing.extend_trace_label(trace_label, "interaction_sample_simulate")
    chunk_tag = chunk_tag or trace_label

    result_list = []
    for (
        i,
        chooser_chunk,
        alternative_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers_and_alts(
        state, choosers, alternatives, trace_label, chunk_tag, chunk_size=chunk_size
    ):
        choices = _interaction_sample_simulate(
            state,
            chooser_chunk,
            alternative_chunk,
            spec,
            choice_column,
            allow_zero_probs,
            zero_prob_choice_val,
            log_alt_losers,
            want_logsums,
            skims,
            locals_d,
            chunk_trace_label,
            trace_choice_name,
            estimator,
            skip_choice,
            chunk_sizer=chunk_sizer,
        )

        result_list.append(choices)

        chunk_sizer.log_df(trace_label, f"result_list", result_list)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert len(choices.index == len(choosers.index))

    return choices
