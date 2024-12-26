# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    chunk,
    estimation,
    interaction_simulate,
    logit,
    simulate,
    tracing,
    util,
    workflow,
)
from activitysim.core.configuration.base import ComputeSettings
from activitysim.core.skim_dataset import DatasetWrapper
from activitysim.core.skim_dictionary import SkimWrapper

logger = logging.getLogger(__name__)

DUMP = False


def make_sample_choices(
    state: workflow.State,
    choosers,
    probs,
    alternatives,
    sample_size,
    alternative_count,
    alt_col_name,
    allow_zero_probs,
    trace_label,
    chunk_sizer,
):
    """

    Parameters
    ----------
    choosers
    probs : pandas DataFrame
        one row per chooser and one column per alternative
    alternatives
        dataframe with index containing alt ids
    sample_size : int
        number of samples/choices to make
    alternative_count
    alt_col_name : str
    trace_label

    Returns
    -------

    """

    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (len(choosers), alternative_count)

    assert isinstance(alternatives, pd.DataFrame)
    assert len(alternatives) == alternative_count

    if allow_zero_probs:
        zero_probs = probs.sum(axis=1) == 0
        if zero_probs.all():
            return pd.DataFrame(
                columns=[alt_col_name, "rand", "prob", choosers.index.name]
            )
        if zero_probs.any():
            # remove from sample
            probs = probs[~zero_probs]
            choosers = choosers[~zero_probs]

    # get sample_size rands for each chooser
    rands = state.get_rn_generator().random_for_df(probs, n=sample_size)

    # transform as we iterate over alternatives
    # reshape so rands[i] is in broadcastable (2-D) shape for cum_probs_arr
    # i.e rands[i] is a 2-D array of one alt choice rand for each chooser
    # rands = rands.T #.reshape(sample_size, -1, 1)
    chunk_sizer.log_df(trace_label, "rands", rands)

    # TODO: is `sample_choices_maker` more efficient?  The order of samples changes, might change repro-randoms
    from .choosing import sample_choices_maker_preserve_ordering

    choices_array, choice_probs_array = sample_choices_maker_preserve_ordering(
        probs.values,
        rands,
        alternatives.index.values,
    )

    chunk_sizer.log_df(trace_label, "choices_array", choices_array)
    chunk_sizer.log_df(trace_label, "choice_probs_array", choice_probs_array)

    # explode to one row per chooser.index, alt_zone_id
    choices_df = pd.DataFrame(
        {
            alt_col_name: choices_array.flatten(order="F"),
            "rand": rands.T.flatten(order="F"),
            "prob": choice_probs_array.flatten(order="F"),
            choosers.index.name: np.repeat(np.asanyarray(choosers.index), sample_size),
        }
    )

    chunk_sizer.log_df(trace_label, "choices_df", choices_df)

    del choices_array
    chunk_sizer.log_df(trace_label, "choices_array", None)
    del rands
    chunk_sizer.log_df(trace_label, "rands", None)
    del choice_probs_array
    chunk_sizer.log_df(trace_label, "choice_probs_array", None)

    # handing this off to caller
    chunk_sizer.log_df(trace_label, "choices_df", None)

    return choices_df


def _interaction_sample(
    state: workflow.State,
    choosers,
    alternatives,
    spec,
    sample_size,
    alt_col_name,
    allow_zero_probs=False,
    log_alt_losers=False,
    skims=None,
    locals_d=None,
    trace_label=None,
    zone_layer=None,
    chunk_sizer=None,
    compute_settings: ComputeSettings | None = None,
):
    """
    Run a MNL simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    Parameters are same as for public function interaction_sa,ple

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
        same shape as utilities, one row per chooser and one column per alternative

    positions : series
        choices among alternatives with the chosen alternative represented
        as the integer index of the selected alternative column in probs

    choices : series
        series with the alternative chosen for each chooser
        the index is same as choosers
        and the series value is the alternative df index of chosen alternative

    zone_layer : {'taz', 'maz'}, default 'taz'
        Specify which zone layer of the skims is to be used.  You cannot use the
        'maz' zone layer in a one-zone model, but you can use the 'taz' layer in
        a two- or three-zone model (e.g. for destination pre-sampling).

    compute_settings : ComputeSettings, optional
        Settings to use if compiling with sharrow

    Returns
    -------
    choices_df : pandas.DataFrame

        A DataFrame where index should match the index of the choosers DataFrame
        and columns alt_col_name, prob, rand, pick_count

        prob: float
            the probability of the chosen alternative
        rand: float
            the rand that did the choosing
        pick_count : int
            number of duplicate picks for chooser, alt
    """

    have_trace_targets = state.tracing.has_trace_targets(choosers)
    trace_ids = None
    trace_rows = None
    num_choosers = len(choosers.index)

    assert num_choosers > 0

    if have_trace_targets:
        state.tracing.trace_df(
            choosers, tracing.extend_trace_label(trace_label, "choosers")
        )
        state.tracing.trace_df(
            alternatives,
            tracing.extend_trace_label(trace_label, "alternatives"),
            slicer="NONE",
            transpose=False,
        )

    if len(spec.columns) > 1:
        raise RuntimeError("spec must have only one column")

    # if using skims, copy index into the dataframe, so it will be
    # available as the "destination" for set_skim_wrapper_targets
    if skims is not None and alternatives.index.name not in alternatives:
        alternatives = alternatives.copy()
        alternatives[alternatives.index.name] = alternatives.index

    chooser_index_id = interaction_simulate.ALT_CHOOSER_ID if log_alt_losers else None

    sharrow_enabled = state.settings.sharrow
    if compute_settings is None:
        compute_settings = ComputeSettings()
    if compute_settings.sharrow_skip:
        sharrow_enabled = False

    # - cross join choosers and alternatives (cartesian product)
    # for every chooser, there will be a row for each alternative
    # index values (non-unique) are from alternatives df
    alternative_count = alternatives.shape[0]

    interaction_utilities = None
    interaction_utilities_sh = None

    if compute_settings is None:
        compute_settings = ComputeSettings()

    # drop variables before the interaction dataframe is created

    # check if tracing is enabled and if we have trace targets
    # if not estimation mode, drop unused columns
    if (not have_trace_targets) and (compute_settings.drop_unused_columns):

        choosers = util.drop_unused_columns(
            choosers,
            spec,
            locals_d,
            custom_chooser=None,
            sharrow_enabled=sharrow_enabled,
            additional_columns=compute_settings.protect_columns,
        )

        alternatives = util.drop_unused_columns(
            alternatives,
            spec,
            locals_d,
            custom_chooser=None,
            sharrow_enabled=sharrow_enabled,
            additional_columns=["tdd"] + compute_settings.protect_columns,
        )

    if sharrow_enabled:
        (
            interaction_utilities,
            trace_eval_results,
        ) = interaction_simulate.eval_interaction_utilities(
            state,
            spec,
            choosers,
            locals_d,
            trace_label,
            trace_rows,
            estimator=None,
            log_alt_losers=log_alt_losers,
            extra_data=alternatives,
            zone_layer=zone_layer,
            compute_settings=compute_settings,
        )
        chunk_sizer.log_df(trace_label, "interaction_utilities", interaction_utilities)
        if sharrow_enabled == "test" or True:
            interaction_utilities_sh, trace_eval_results_sh = (
                interaction_utilities,
                trace_eval_results,
            )
    if not sharrow_enabled or (sharrow_enabled == "test"):
        interaction_df = logit.interaction_dataset(
            state,
            choosers,
            alternatives,
            sample_size=alternative_count,
            chooser_index_id=chooser_index_id,
        )

        chunk_sizer.log_df(trace_label, "interaction_df", interaction_df)

        assert alternative_count == len(interaction_df.index) / len(choosers.index)

        if skims is not None:
            simulate.set_skim_wrapper_targets(interaction_df, skims)

        # evaluate expressions from the spec multiply by coefficients and sum
        # spec is df with one row per spec expression and one col with utility coefficient
        # column names of interaction_df match spec index values
        # utilities has utility value for element in the cross product of choosers and alternatives
        # interaction_utilities is a df with one utility column and one row per row in interaction_df
        if have_trace_targets:
            trace_rows, trace_ids = state.tracing.interaction_trace_rows(
                interaction_df, choosers, alternative_count
            )

            state.tracing.trace_df(
                interaction_df[trace_rows],
                tracing.extend_trace_label(trace_label, "interaction_df"),
                slicer="NONE",
                transpose=False,
            )
        else:
            trace_rows = trace_ids = None

        # interaction_utilities is a df with one utility column and one row per interaction_df row
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
            estimator=None,
            log_alt_losers=log_alt_losers,
            zone_layer=zone_layer,
            compute_settings=ComputeSettings(sharrow_skip=True),
        )
        chunk_sizer.log_df(trace_label, "interaction_utilities", interaction_utilities)

        # ########### HWM - high water mark (point of max observed memory usage)

        del interaction_df
        chunk_sizer.log_df(trace_label, "interaction_df", None)

    if sharrow_enabled == "test":
        try:
            if interaction_utilities_sh is not None:
                np.testing.assert_allclose(
                    interaction_utilities_sh.values.reshape(
                        interaction_utilities.values.shape
                    ),
                    interaction_utilities.values,
                    rtol=1e-2,
                    atol=1e-6,
                    err_msg="utility not aligned",
                    verbose=True,
                )
        except AssertionError as err:
            print(err)
            misses = np.where(
                ~np.isclose(
                    interaction_utilities_sh.values,
                    interaction_utilities.values,
                    rtol=1e-2,
                    atol=1e-6,
                )
            )
            _sh_util_miss1 = interaction_utilities_sh.values[
                tuple(m[0] for m in misses)
            ]
            _u_miss1 = interaction_utilities.values[tuple(m[0] for m in misses)]
            diff = _sh_util_miss1 - _u_miss1
            if len(misses[0]) > interaction_utilities_sh.values.size * 0.01:
                print("big problem")
                print(misses)
                if "nan location mismatch" in str(err):
                    print("nan location mismatch interaction_utilities_sh")
                    print(np.where(np.isnan(interaction_utilities_sh.values)))
                    print("nan location mismatch interaction_utilities legacy")
                    print(np.where(np.isnan(interaction_utilities.values)))
                print("misses =>", misses)
                j = 0
                while j < len(misses[0]):
                    print(
                        f"miss {j} {tuple(m[j] for m in misses)}:",
                        interaction_utilities_sh.values[tuple(m[j] for m in misses)],
                        "!=",
                        interaction_utilities.values[tuple(m[j] for m in misses)],
                    )
                    j += 1
                    if j > 10:
                        break
                raise

    if have_trace_targets and trace_ids is not None:
        state.tracing.trace_interaction_eval_results(
            trace_eval_results,
            trace_ids,
            tracing.extend_trace_label(trace_label, "eval"),
        )

    if have_trace_targets and trace_rows is not None:
        try:
            state.tracing.trace_df(
                interaction_utilities[trace_rows],
                tracing.extend_trace_label(trace_label, "interaction_utilities"),
                slicer="NONE",
                transpose=False,
            )
        except ValueError:
            pass

    state.tracing.dump_df(
        DUMP, interaction_utilities, trace_label, "interaction_utilities"
    )

    # reshape utilities (one utility column and one row per row in interaction_utilities)
    # to a dataframe with one row per chooser and one column per alternative
    utilities = pd.DataFrame(
        interaction_utilities.values.reshape(len(choosers), alternative_count),
        index=choosers.index,
    )
    chunk_sizer.log_df(trace_label, "utilities", utilities)

    del interaction_utilities
    chunk_sizer.log_df(trace_label, "interaction_utilities", None)

    if have_trace_targets:
        state.tracing.trace_df(
            utilities,
            tracing.extend_trace_label(trace_label, "utils"),
            column_labels=["alternative", "utility"],
        )

    state.tracing.dump_df(DUMP, utilities, trace_label, "utilities")

    # convert to probabilities (utilities exponentiated and normalized to probs)
    # probs is same shape as utilities, one row per chooser and one column for alternative
    probs = logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=allow_zero_probs,
        trace_label=trace_label,
        trace_choosers=choosers,
        overflow_protection=not allow_zero_probs,
    )
    chunk_sizer.log_df(trace_label, "probs", probs)

    del utilities
    chunk_sizer.log_df(trace_label, "utilities", None)

    if have_trace_targets:
        state.tracing.trace_df(
            probs,
            tracing.extend_trace_label(trace_label, "probs"),
            column_labels=["alternative", "probability"],
        )

    if sample_size == 0:
        # FIXME return full alternative set rather than sample
        logger.info(
            "Estimation mode for %s using unsampled alternatives" % (trace_label,)
        )

        index_name = probs.index.name
        choices_df = (
            pd.melt(probs.reset_index(), id_vars=[index_name])
            .sort_values(by=index_name, kind="mergesort")
            .set_index(index_name)
            .rename(columns={"value": "prob"})
            .drop(columns="variable")
        )

        choices_df["pick_count"] = 1
        choices_df.insert(
            0, alt_col_name, np.tile(alternatives.index.values, len(choosers.index))
        )

        return choices_df
    else:
        choices_df = make_sample_choices(
            state,
            choosers,
            probs,
            alternatives,
            sample_size,
            alternative_count,
            alt_col_name,
            allow_zero_probs=allow_zero_probs,
            trace_label=trace_label,
            chunk_sizer=chunk_sizer,
        )

    chunk_sizer.log_df(trace_label, "choices_df", choices_df)

    if estimation.manager.enabled and sample_size > 0:
        # we need to ensure chosen alternative is included in the sample
        survey_choices = estimation.manager.get_survey_destination_choices(
            state, choosers, trace_label
        )
        if survey_choices is not None:
            assert (
                survey_choices.index == choosers.index
            ).all(), "survey_choices and choosers must have the same index"
            survey_choices.name = alt_col_name
            survey_choices = survey_choices.dropna().astype(
                choices_df[alt_col_name].dtype
            )

            # merge all survey choices onto choices_df
            probs_df = probs.reset_index().melt(
                id_vars=[choosers.index.name],
                var_name=alt_col_name,
                value_name="prob",
            )
            # probs are numbered 0..n-1 so we need to map back to alt ids
            zone_map = pd.Series(alternatives.index).to_dict()
            probs_df[alt_col_name] = probs_df[alt_col_name].map(zone_map)

            survey_choices = pd.merge(
                survey_choices,
                probs_df,
                on=[choosers.index.name, alt_col_name],
                how="left",
            )
            survey_choices["rand"] = 0
            survey_choices["prob"].fillna(0, inplace=True)
            choices_df = pd.concat([choices_df, survey_choices], ignore_index=True)
            choices_df.sort_values(by=[choosers.index.name], inplace=True)

    del probs
    chunk_sizer.log_df(trace_label, "probs", None)

    # pick_count and pick_dup
    # pick_count is number of duplicate picks
    # pick_dup flag is True for all but first of duplicates
    pick_group = choices_df.groupby([choosers.index.name, alt_col_name])

    # number each item in each group from 0 to the length of that group - 1.
    choices_df["pick_count"] = pick_group.cumcount(ascending=True)
    # flag duplicate rows after first
    choices_df["pick_dup"] = choices_df["pick_count"] > 0
    # add reverse cumcount to get total pick_count (conveniently faster than groupby.count + merge)
    choices_df["pick_count"] += pick_group.cumcount(ascending=False) + 1

    # drop the duplicates
    choices_df = choices_df[~choices_df["pick_dup"]]
    del choices_df["pick_dup"]
    chunk_sizer.log_df(trace_label, "choices_df", choices_df)

    # set index after groupby so we can trace on it
    choices_df.set_index(choosers.index.name, inplace=True)

    state.tracing.dump_df(DUMP, choices_df, trace_label, "choices_df")

    if have_trace_targets:
        state.tracing.trace_df(
            choices_df,
            tracing.extend_trace_label(trace_label, "sampled_alternatives"),
            transpose=False,
            column_labels=["sample_alt", "alternative"],
        )

    # don't need this after tracing
    del choices_df["rand"]
    chunk_sizer.log_df(trace_label, "choices_df", choices_df)

    # - NARROW
    choices_df["prob"] = choices_df["prob"].astype(np.float32)
    assert (choices_df["pick_count"].max() < 4294967295) or (choices_df.empty)
    choices_df["pick_count"] = choices_df["pick_count"].astype(np.uint32)

    return choices_df


def interaction_sample(
    state: workflow.State,
    choosers: pd.DataFrame,
    alternatives: pd.DataFrame,
    spec: pd.DataFrame,
    sample_size: int,
    alt_col_name: str,
    allow_zero_probs: bool = False,
    log_alt_losers: bool = False,
    skims: SkimWrapper | DatasetWrapper | None = None,
    locals_d=None,
    chunk_size: int = 0,
    chunk_tag: str | None = None,
    trace_label: str | None = None,
    zone_layer: str | None = None,
    explicit_chunk_size: float = 0,
    compute_settings: ComputeSettings | None = None,
):
    """
    Run a simulation in the situation in which alternatives must
    be merged with choosers because there are interaction terms or
    because alternatives are being sampled.

    optionally (if chunk_size > 0) iterates over choosers in chunk_size chunks

    Parameters
    ----------
    state : State
    choosers : pandas.DataFrame
        DataFrame of choosers
    alternatives : pandas.DataFrame
        DataFrame of alternatives - will be merged with choosers and sampled
    spec : pandas.DataFrame
        A Pandas DataFrame that gives the specification of the variables to
        compute and the coefficients for each variable.
        Variable specifications must be in the table index and the
        table should have only one column of coefficients.
    sample_size : int, optional
        Sample alternatives with sample of given size.  By default is None,
        which does not sample alternatives.
    alt_col_name: str
        name to give the sampled_alternative column
    skims : SkimWrapper or DatasetWrapper or None
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
    zone_layer : {'taz', 'maz'}, default 'taz'
        Specify which zone layer of the skims is to be used.  You cannot use the
        'maz' zone layer in a one-zone model, but you can use the 'taz' layer in
        a two- or three-zone model (e.g. for destination pre-sampling).
    explicit_chunk_size : float, optional
        If > 0, specifies the chunk size to use when chunking the interaction
        simulation. If < 1, specifies the fraction of the total number of choosers.

    Returns
    -------
    choices_df : pandas.DataFrame

        A DataFrame where index should match the index of the choosers DataFrame
        (except with sample_size rows for each choser row, one row for each alt sample)
        and columns alt_col_name, prob, rand, pick_count

        <alt_col_name>:
            alt identifier from alternatives[<alt_col_name>
        prob: float
            the probability of the chosen alternative
        pick_count : int
            number of duplicate picks for chooser, alt
    """

    trace_label = tracing.extend_trace_label(trace_label, "interaction_sample")
    chunk_tag = chunk_tag or trace_label

    # we return alternatives ordered in (index, alt_col_name)
    # if choosers index is not ordered, it is probably a mistake, since the alts wont line up
    assert alt_col_name is not None
    if not choosers.index.is_monotonic_increasing:
        assert choosers.index.is_monotonic_increasing

    # FIXME - legacy logic - not sure this is needed or even correct?
    sample_size = min(sample_size, len(alternatives.index))

    result_list = []
    for (
        i,
        chooser_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers(
        state, choosers, trace_label, chunk_tag, explicit_chunk_size=explicit_chunk_size
    ):
        choices = _interaction_sample(
            state,
            chooser_chunk,
            alternatives,
            spec=spec,
            sample_size=sample_size,
            alt_col_name=alt_col_name,
            allow_zero_probs=allow_zero_probs,
            log_alt_losers=log_alt_losers,
            skims=skims,
            locals_d=locals_d,
            trace_label=chunk_trace_label,
            zone_layer=zone_layer,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

        if choices.shape[0] > 0:
            # might not be any if allow_zero_probs
            result_list.append(choices)

            chunk_sizer.log_df(trace_label, f"result_list", result_list)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    if len(result_list) > 1:
        choices = pd.concat(result_list)

    assert allow_zero_probs or (
        len(choosers.index) == len(np.unique(choices.index.values))
    )

    # keep alts in canonical order so choices based on their probs are stable across runs
    choices = choices.sort_values(by=alt_col_name).sort_index(kind="mergesort")

    return choices
