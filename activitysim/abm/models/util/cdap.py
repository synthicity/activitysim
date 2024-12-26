from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import itertools
import logging

import numpy as np
import pandas as pd

from activitysim.core import chunk, logit, simulate, tracing, workflow
from activitysim.core.configuration.base import ComputeSettings

logger = logging.getLogger(__name__)

# FIXME - this allows us to turn some dev debug table dump code on and off - eventually remove?
# DUMP = False

_persons_index_ = "person_id"
_hh_index_ = "household_id"
_hh_size_ = "hhsize"

_hh_id_ = "household_id"
_ptype_ = "ptype"
_age_ = "age"

# For clarity, the named constant MAX_HHSIZE refers to the cdap 5 person threshold figure.
MAX_HHSIZE = 5

MAX_INTERACTION_CARDINALITY = 3


def set_hh_index(df):
    # index on household_id, not person_id
    df.set_index(_hh_id_, inplace=True)
    df.index.name = _hh_index_


def add_pn(col, pnum):
    """
    return the canonical column name for the indiv_util column or columns
    in merged hh_chooser df for individual with cdap_rank pnum

    e.g. M_p1, ptype_p2 but leave _hh_id_ column unchanged
    """
    if type(col) is str:
        return col if col == _hh_id_ else "%s_p%s" % (col, pnum)
    elif isinstance(col, (list, tuple)):
        return [c if c == _hh_id_ else "%s_p%s" % (c, pnum) for c in col]
    else:
        raise RuntimeError("add_pn col not list or str")


def assign_cdap_rank(
    state: workflow.State | None,
    persons,
    person_type_map,
    trace_hh_id=None,
    trace_label=None,
):
    """
    Assign an integer index, cdap_rank, to each household member. (Starting with 1, not 0)

    Modifies persons df in place

    The cdap_rank order is important, because cdap only assigns activities to the first
    MAX_HHSIZE persons in each household.

    This will preferentially be two working adults and the three youngest children.

    Rank is assigned starting at 1. This necessitates some care indexing, but is preferred as
    it follows the convention of 1-based pnums in expression files.

    According to the documentation of reOrderPersonsForCdap in mtctm2.abm.ctramp
    HouseholdCoordinatedDailyActivityPatternModel:

    "Method reorders the persons in the household for use with the CDAP model,
    which only explicitly models the interaction of five persons in a HH. Priority
    in the reordering is first given to full time workers (up to two), then to
    part time workers (up to two workers, of any type), then to children (youngest
    to oldest, up to three). If the method is called for a household with less
    than 5 people, the cdapPersonArray is the same as the person array."

    We diverge from the above description in that a cdap_rank is assigned to all persons,
    including 'extra' household members, whose activity is assigned subsequently.
    The pair _hh_id_, cdap_rank will uniquely identify each household member.

    Parameters
    ----------
    persons : pandas.DataFrame
        Table of persons data. Must contain columns _hh_size_, _hh_id_, _ptype_, _age_

    Returns
    -------
    cdap_rank : pandas.Series
        integer cdap_rank of every person, indexed on _persons_index_
    """

    # transient categories used to categorize persons in cdap_rank before assigning final rank
    RANK_WORKER = 1
    RANK_CHILD = 2
    RANK_BACKFILL = 3
    RANK_UNASSIGNED = 9
    persons["cdap_rank"] = RANK_UNASSIGNED

    # choose up to 2 workers, preferring full over part, older over younger
    workers = (
        persons.loc[
            persons[_ptype_].isin(person_type_map["WORKER"]), [_hh_id_, _ptype_]
        ]
        .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True])
        .groupby(_hh_id_)
        .head(2)
    )
    # tag the selected workers
    persons.loc[workers.index, "cdap_rank"] = RANK_WORKER
    del workers

    # choose up to 3, preferring youngest
    children = (
        persons.loc[
            persons[_ptype_].isin(person_type_map["CHILD"]), [_hh_id_, _ptype_, _age_]
        ]
        .sort_values(by=[_hh_id_, _ptype_], ascending=[True, True])
        .groupby(_hh_id_)
        .head(3)
    )
    # tag the selected children
    persons.loc[children.index, "cdap_rank"] = RANK_CHILD
    del children

    # choose up to MAX_HHSIZE, preferring anyone already chosen
    # others = \
    #     persons[[_hh_id_, 'cdap_rank']]\
    #     .sort_values(by=[_hh_id_, 'cdap_rank'], ascending=[True, True])\
    #     .groupby(_hh_id_).head(MAX_HHSIZE)

    # choose up to MAX_HHSIZE, choosing randomly
    others = persons[[_hh_id_, "cdap_rank"]].copy()
    if state is None:
        # typically in estimation, no state is available, just use stable but simple random
        others["random_order"] = np.random.default_rng(seed=0).uniform(size=len(others))
    else:
        others["random_order"] = state.get_rn_generator().random_for_df(persons)
    others = (
        others.sort_values(by=[_hh_id_, "random_order"], ascending=[True, True])
        .groupby(_hh_id_)
        .head(MAX_HHSIZE)
    )

    # tag the backfilled persons
    persons.loc[
        others[others.cdap_rank == RANK_UNASSIGNED].index, "cdap_rank"
    ] = RANK_BACKFILL
    del others

    # assign person number in cdapPersonArray preference order
    # i.e. convert cdap_rank from category to index in order of category rank within household
    # groupby rank() is slow, so we compute rank artisanally
    # save time by sorting only the columns we need (persons is big, and sort moves data)
    p = persons[[_hh_id_, "cdap_rank", _age_]].sort_values(
        by=[_hh_id_, "cdap_rank", _age_], ascending=[True, True, True]
    )
    rank = p.groupby(_hh_id_).size().map(range)
    rank = [item + 1 for sublist in rank for item in sublist]
    p["cdap_rank"] = rank
    persons["cdap_rank"] = p["cdap_rank"]  # assignment aligns on index values

    # if DUMP:
    #     state.tracing.trace_df(persons, '%s.DUMP.cdap_person_array' % trace_label,
    #                      transpose=False, slicer='NONE')

    if trace_hh_id and state is not None:
        state.tracing.trace_df(persons, "%s.cdap_rank" % trace_label)

    return persons["cdap_rank"]


def individual_utilities(
    state: workflow.State,
    persons,
    cdap_indiv_spec,
    locals_d,
    trace_hh_id=None,
    trace_label=None,
    *,
    chunk_sizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Calculate CDAP utilities for all individuals.

    Parameters
    ----------
    persons : pandas.DataFrame
        DataFrame of individual persons data.
    cdap_indiv_spec : pandas.DataFrame
        CDAP spec applied to individuals.

    Returns
    -------
    utilities : pandas.DataFrame
        Will have index of `persons` and columns for each of the alternatives.
        plus some 'useful columns' [_hh_id_, _ptype_, 'cdap_rank', _hh_size_]

    """

    # calculate single person utilities
    indiv_utils = simulate.eval_utilities(
        state,
        cdap_indiv_spec,
        persons,
        locals_d,
        trace_label=trace_label,
        chunk_sizer=chunk_sizer,
        compute_settings=compute_settings,
    )

    # add columns from persons to facilitate building household interactions
    useful_columns = [_hh_id_, _ptype_, "cdap_rank", _hh_size_]
    indiv_utils[useful_columns] = persons[useful_columns]

    # add attributes for joint tour utility
    from activitysim.abm.models.cdap import CdapSettings

    model_settings = CdapSettings.read_settings_file(state.filesystem, "cdap.yaml")
    additional_useful_columns = model_settings.JOINT_TOUR_USEFUL_COLUMNS
    if additional_useful_columns is not None:
        indiv_utils[additional_useful_columns] = persons[additional_useful_columns]

    if trace_hh_id:
        state.tracing.trace_df(
            indiv_utils,
            "%s.indiv_utils" % trace_label,
            column_labels=["activity", "person"],
        )

    return indiv_utils


def preprocess_interaction_coefficients(interaction_coefficients):
    """
    The input cdap_interaction_coefficients.csv file has three columns:

    activity
        A single character activity type name (M, N, or H)

    interaction_ptypes
        List of ptypes in the interaction (in order of increasing ptype)
        Stars `(***)` instead of ptypes means the interaction applies to all ptypes in that size hh.

    coefficient
        The coefficient to apply for all hh interactions for this activity and set of ptypes

    To facilitate building the spec for a given hh ssize, we add two additional columns:

    cardinality
        the number of persons in the interaction (e.g. 3 for a 3-way interaction)

    slug
        a human friendly efficient name so we can dump a readable spec trace file for debugging
        this slug is then replaced with the numerical coefficient value prior to evaluation
    """

    # make a copy
    coefficients = interaction_coefficients.copy()

    if not coefficients["activity"].isin(["M", "N", "H"]).all():
        msg = (
            "Error in cdap_interaction_coefficients at row %s. Expect only M, N, or H!"
            % coefficients[~coefficients["activity"].isin(["M", "N", "H"])].index.values
        )
        raise RuntimeError(msg)

    coefficients["cardinality"] = (
        coefficients["interaction_ptypes"].astype(str).str.len()
    )

    wildcards = coefficients.interaction_ptypes == coefficients.cardinality.map(
        lambda x: x * "*"
    )
    coefficients.loc[wildcards, "interaction_ptypes"] = ""

    coefficients["slug"] = coefficients["activity"] * coefficients[
        "cardinality"
    ] + coefficients["interaction_ptypes"].astype(str)

    return coefficients


def cached_spec_name(hhsize):
    return "cdap_spec_%s" % hhsize


def cached_joint_spec_name(hhsize):
    return "cdap_joint_spec_%s" % hhsize


def get_cached_spec(state: workflow.State, hhsize):

    spec_name = cached_spec_name(hhsize)

    spec = state.get_injectable(spec_name, None)
    if spec is not None:
        logger.debug("build_cdap_spec returning cached injectable spec %s", spec_name)
        return spec

    # this is problematic for multiprocessing and since we delete csv files in output_dir
    # at the start of every run, doesn't provide any benefit in single-processing as the
    # cached spec will be available as an injectable to subsequent chunks

    # # try data dir
    # if os.path.exists(state.get_output_file_path(spec_name)):
    #     spec_path = state.get_output_file_path(spec_name)
    #     logger.info("build_cdap_spec reading cached spec %s from %s", spec_name, spec_path)
    #     return pd.read_csv(spec_path, index_col='Expression')

    return None


def get_cached_joint_spec(state: workflow.State, hhsize):

    spec_name = cached_joint_spec_name(hhsize)

    spec = state.get_injectable(spec_name, None)
    if spec is not None:
        logger.debug(
            "build_cdap_joint_spec returning cached injectable spec %s", spec_name
        )
        return spec

    return None


def cache_spec(state: workflow.State, hhsize, spec):
    spec_name = cached_spec_name(hhsize)
    # cache as injectable
    state.add_injectable(spec_name, spec)


def cache_joint_spec(state: workflow.State, hhsize, spec):
    spec_name = cached_joint_spec_name(hhsize)
    # cache as injectable
    state.add_injectable(spec_name, spec)


def build_cdap_spec(
    state: workflow.State,
    interaction_coefficients,
    hhsize,
    trace_spec=False,
    trace_label=None,
    cache=True,
    joint_tour_alt=False,
):
    """
    Build a spec file for computing utilities of alternative household member interaction patterns
    for households of specified size.

    We generate this spec automatically from a table of rules and coefficients because the
    interaction rules are fairly simple and can be expressed compactly whereas
    there is a lot of redundancy between the spec files for different household sizes, as well as
    in the vectorized expression of the interaction alternatives within the spec file itself

    interaction_coefficients has five columns:
        activity
            A single character activity type name (M, N, or H)
        interaction_ptypes
            List of ptypes in the interaction (in order of increasing ptype) or empty for wildcards
            (meaning that the interaction applies to all ptypes in that size hh)
        cardinality
            the number of persons in the interaction (e.g. 3 for a 3-way interaction)
        slug
            a human friendly efficient name so we can dump a readable spec trace file for debugging
            this slug is replaced with the numerical coefficient value after we dump the trace file
        coefficient
            The coefficient to apply for all hh interactions for this activity and set of ptypes

    The generated spec will have the eval expression in the index, and a utility column for each
    alternative (e.g. ['HH', 'HM', 'HN', 'MH', 'MM', 'MN', 'NH', 'NM', 'NN'] for hhsize 2)

    In order to be able to dump the spec in a human-friendly fashion to facilitate debugging the
    cdap_interaction_coefficients table, we first populate utility columns in the spec file
    with the coefficient slugs, dump the spec file, and then replace the slugs with coefficients.

    Parameters
    ----------
    interaction_coefficients : pandas.DataFrame
        Rules and coefficients for generating interaction specs for different household sizes
    hhsize : int
        household size for which the spec should be built.

    Returns
    -------
    spec: pandas.DataFrame

    """

    t0 = tracing.print_elapsed_time()

    # if DUMP:
    #     # dump the interaction_coefficients table because it has been preprocessed
    #     state.tracing.trace_df(interaction_coefficients,
    #                      '%s.hhsize%d_interaction_coefficients' % (trace_label, hhsize),
    #                      transpose=False, slicer='NONE')

    # cdap spec is same for all households of MAX_HHSIZE and greater
    hhsize = min(hhsize, MAX_HHSIZE)

    if cache:
        spec = get_cached_spec(state, hhsize)
        if spec is not None:
            return spec

    expression_name = "Expression"

    # generate a list of activity pattern alternatives for this hhsize
    # e.g. ['HH', 'HM', 'HN', 'MH', 'MM', 'MN', 'NH', 'NM', 'NN'] for hhsize=2
    alternatives = ["".join(tup) for tup in itertools.product("HMN", repeat=hhsize)]

    if joint_tour_alt:
        joint_alternatives = [
            "".join(tup) + "J"
            for tup in itertools.product("HMN", repeat=hhsize)
            if tup.count("M") + tup.count("N") >= 2
        ]
        alternatives = alternatives + joint_alternatives

    # spec df has expression column plus a column for each alternative
    spec = pd.DataFrame(columns=[expression_name] + alternatives)

    # Before processing the interaction_coefficients, we add add rows to the spec to carry
    # the alternative utilities previously computed for each individual into all hh alternative
    # columns in which the individual assigned that alternative. The Expression column contains
    # the name of the choosers column with that individuals utility for the individual alternative
    # and the hh alternative columns that should receive that utility are given a value of 1
    # e.g. M_p1 is a column in choosers with the individual utility to person p1 of alternative M
    #   Expression   MM   MN   MH   NM   NN   NH   HM   HN   HH
    #         M_p1  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
    #         N_p1  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0
    for pnum in range(1, hhsize + 1):
        for activity in ["M", "N", "H"]:
            new_row_index = len(spec)
            spec.loc[new_row_index, expression_name] = add_pn(activity, pnum)

            # list of alternative columns where person pnum has expression activity
            # e.g. for M_p1 we want the columns where activity M is in position p1
            alternative_columns = [
                alt for alt in alternatives if alt[pnum - 1] == activity
            ]
            spec.loc[new_row_index, alternative_columns] = 1

    # ignore rows whose cardinality exceeds hhsize
    relevant_rows = interaction_coefficients.cardinality <= hhsize

    # for each row in the interaction_coefficients table
    for row in interaction_coefficients[relevant_rows].itertuples():
        # if it is a wildcard all_people interaction
        if not row.interaction_ptypes:
            # wildcard interactions only apply if the interaction includes all household members
            # this will be the case if the cardinality of the wildcard equals the hhsize
            # conveniently, the slug is given the name of the alternative column (e.g. HHHH)

            # conveniently, for wildcards, the slug has been assigned the name of the alternative
            # (e.g. HHHH) that it applies to, since the interaction includes all household members
            # and there are no ptypes to append to it

            # FIXME - should we be doing this for greater than HH_MAXSIZE households?
            if row.slug in alternatives:
                spec.loc[len(spec), [expression_name, row.slug]] = ["1", row.slug]

            continue

        if not (0 <= row.cardinality <= MAX_INTERACTION_CARDINALITY):
            raise RuntimeError(
                "Bad row cardinality %d for %s" % (row.cardinality, row.slug)
            )

        # for all other interaction rules, we need to generate a row in the spec for each
        # possible combination of interacting persons
        # e.g. for (1, 2), (1,3), (2,3) for a coefficient with cardinality 2 in hhsize 3
        for tup in itertools.combinations(list(range(1, hhsize + 1)), row.cardinality):
            # determine the name of the chooser column with the ptypes for this interaction
            if row.cardinality == 1:
                interaction_column = "ptype_p%d" % tup[0]
            else:
                # column named (e.g.) p1_p3 for an interaction between p1 and p3
                interaction_column = "_".join(["p%s" % pnum for pnum in tup])

            # build expression that evaluates True iff the interaction is between specified ptypes
            # (e.g.) p1_p3==13 for an interaction between p1 and p3 of ptypes 1 and 3 (or 3 and1 )
            expression = "%s==%s" % (interaction_column, row.interaction_ptypes)

            # create list of columns with names matching activity for each of the persons in tup
            # e.g. ['MMM', 'MMN', 'MMH'] for an interaction between p1 and p3 with activity 'M'
            # alternative_columns = \
            #     filter(lambda alt: all([alt[p - 1] == row.activity for p in tup]), alternatives)
            alternative_columns = [
                alt
                for alt in alternatives
                if all([alt[p - 1] == row.activity for p in tup])
            ]

            # a row for this interaction may already exist,
            # e.g. if there are rules for both HH13 and MM13, we don't need to add rows for both
            # since they are triggered by the same expressions (e.g. p1_p2==13, p1_p3=13,...)
            existing_row_index = spec[expression_name] == expression
            if (existing_row_index).any():
                # if the rows exist, simply update the appropriate alternative columns in spec
                spec.loc[existing_row_index, alternative_columns] = row.slug
                spec.loc[existing_row_index, expression_name] = expression
            else:
                # otherwise, add a new row to spec
                new_row_index = len(spec)
                spec.loc[new_row_index, alternative_columns] = row.slug
                spec.loc[new_row_index, expression_name] = expression

    # eval expression goes in the index
    spec.set_index(expression_name, inplace=True)

    simulate.uniquify_spec_index(spec)

    if trace_spec:
        state.tracing.trace_df(
            spec,
            "%s.hhsize%d_spec" % (trace_label, hhsize),
            transpose=False,
            slicer="NONE",
        )

    # replace slug with coefficient
    d = interaction_coefficients.set_index("slug")["coefficient"].to_dict()
    for c in spec.columns:
        spec[c] = spec[c].map(lambda x: d.get(x, x or 0.0)).fillna(0)

    if trace_spec:
        state.tracing.trace_df(
            spec,
            "%s.hhsize%d_spec_patched" % (trace_label, hhsize),
            transpose=False,
            slicer="NONE",
        )

    if cache:
        cache_spec(state, hhsize, spec)

    t0 = tracing.print_elapsed_time("build_cdap_spec hh_size %s" % hhsize, t0)

    return spec


def build_cdap_joint_spec(
    state: workflow.State,
    joint_tour_coefficients,
    hhsize,
    trace_spec=False,
    trace_label=None,
    cache=True,
):
    """
    Build a spec file for computing joint tour utilities of alternative household member for households of specified size.
    We generate this spec automatically from a table of rules and coefficients because the
    interaction rules are fairly simple and can be expressed compactly whereas
    there is a lot of redundancy between the spec files for different household sizes, as well as
    in the vectorized expression of the interaction alternatives within the spec file itself
    joint_tour_coefficients has five columns:
        label
            label of the expression
        description
            description of the expression
        dependency
            if the expression is dependent on alternative, and which alternative is it dependent on
            (e.g. M_px, N_px, H_px)
        expression
            expression of the utility term
        coefficient
            The coefficient to apply for the alternative
    The generated spec will have the eval expression in the index, and a utility column for each
    alternative (e.g. ['HH', 'HM', 'HN', 'MH', 'MM', 'MN', 'NH', 'NM', 'NN', 'MMJ', 'MNJ', 'NMJ', 'NNJ'] for hhsize 2 with joint alts)
    Parameters
    ----------
    joint_tour_coefficients : pandas.DataFrame
        Rules and coefficients for generating joint tour specs for different household sizes
    hhsize : int
        household size for which the spec should be built.
    Returns
    -------
    spec: pandas.DataFrame
    """

    t0 = tracing.print_elapsed_time()

    # cdap joint spec is same for all households of MAX_HHSIZE and greater
    hhsize = min(hhsize, MAX_HHSIZE)

    if cache:
        spec = get_cached_joint_spec(state, hhsize)
        if spec is not None:
            return spec

    expression_name = "Expression"

    # generate a list of activity pattern alternatives for this hhsize
    # e.g. ['HH', 'HM', 'HN', 'MH', 'MM', 'MN', 'NH', 'NM', 'NN'] for hhsize=2
    alternatives = ["".join(tup) for tup in itertools.product("HMN", repeat=hhsize)]

    joint_alternatives = [
        "".join(tup) + "J"
        for tup in itertools.product("HMN", repeat=hhsize)
        if tup.count("M") + tup.count("N") >= 2
    ]
    alternatives = alternatives + joint_alternatives

    # spec df has expression column plus a column for each alternative
    spec = pd.DataFrame(columns=[expression_name] + alternatives)

    # Before processing the interaction_coefficients, we add add rows to the spec to carry
    # the alternative utilities previously computed for each individual into all hh alternative
    # columns in which the individual assigned that alternative. The Expression column contains
    # the name of the choosers column with that individuals utility for the individual alternative
    # and the hh alternative columns that should receive that utility are given a value of 1
    # e.g. M_p1 is a column in choosers with the individual utility to person p1 of alternative M
    #   Expression   MM   MN   MH   NM   NN   NH   HM   HN   HH
    #         M_p1  1.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
    #         N_p1  0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0
    for pnum in range(1, hhsize + 1):
        for activity in ["M", "N", "H"]:

            new_row_index = len(spec)
            spec.loc[new_row_index, expression_name] = add_pn(activity, pnum)

            # list of alternative columns where person pnum has expression activity
            # e.g. for M_p1 we want the columns where activity M is in position p1
            alternative_columns = [
                alt for alt in alternatives if alt[pnum - 1] == activity
            ]
            spec.loc[new_row_index, alternative_columns] = 1

    # for each row in the joint util table
    for row in joint_tour_coefficients.itertuples():

        # if there is no dependencies
        if row.dependency is np.nan:
            expression = row.Expression
            # add a new row to spec
            new_row_index = len(spec)
            spec.loc[new_row_index, expression_name] = expression
            spec.loc[new_row_index, alternatives] = row.coefficient
        # if there is dependencies
        else:
            dependency_name = row.dependency
            expression = row.Expression
            coefficient = row.coefficient
            if dependency_name in ["M_px", "N_px", "H_px"]:
                if "_pxprod" in expression:
                    prod_conds = [j.strip() for j in row.Expression.split("|")]
                    expanded_expressions = [
                        tup
                        for tup in itertools.product(
                            range(len(prod_conds)), repeat=hhsize
                        )
                    ]
                    for expression_tup in expanded_expressions:
                        expression_list = []
                        dependency_list = []
                        for counter in range(len(expression_tup)):
                            expression_list.append(
                                prod_conds[expression_tup[counter]].replace(
                                    "xprod", str(counter + 1)
                                )
                            )
                            if expression_tup[counter] == 0:
                                dependency_list.append(
                                    dependency_name.replace("x", str(counter + 1))
                                )

                        expression_value = "&".join(expression_list)
                        dependency_value = pd.Series(
                            np.ones(len(alternatives)), index=alternatives
                        )
                        if len(dependency_list) > 0:
                            for dependency in dependency_list:
                                # temp = spec.loc[spec[expression_name]==dependency, alternatives].squeeze().fillna(0)
                                dependency_value *= (
                                    spec.loc[
                                        spec[expression_name] == dependency,
                                        alternatives,
                                    ]
                                    .squeeze()
                                    .fillna(0)
                                )

                        # add a new row to spec
                        new_row_index = len(spec)
                        spec.loc[new_row_index] = dependency_value
                        spec.loc[new_row_index, expression_name] = expression_value
                        spec.loc[new_row_index, alternatives] = (
                            spec.loc[new_row_index, alternatives] * coefficient
                        )

                elif "_px" in expression:
                    for pnum in range(1, hhsize + 1):
                        dependency_name = row.dependency.replace("x", str(pnum))
                        expression = row.Expression.replace("x", str(pnum))

                        # add a new row to spec
                        new_row_index = len(spec)
                        spec.loc[new_row_index] = spec.loc[
                            spec[expression_name] == dependency_name
                        ].squeeze()
                        spec.loc[new_row_index, expression_name] = expression
                        spec.loc[new_row_index, alternatives] = (
                            spec.loc[new_row_index, alternatives] * coefficient
                        )

    # drop dependency rows
    spec = spec[~spec[expression_name].str.startswith(("M_p", "N_p", "H_p"))]

    # eval expression goes in the index
    spec.set_index(expression_name, inplace=True)

    for c in spec.columns:
        spec[c] = spec[c].fillna(0)

    simulate.uniquify_spec_index(spec)

    # make non-joint alts 0
    for c in alternatives:
        if c.endswith("J"):
            continue
        else:
            spec[c] = 0

    if trace_spec:
        state.tracing.trace_df(
            spec,
            "%s.hhsize%d_joint_spec" % (trace_label, hhsize),
            transpose=False,
            slicer="NONE",
        )

    if trace_spec:
        state.tracing.trace_df(
            spec,
            "%s.hhsize%d_joint_spec_patched" % (trace_label, hhsize),
            transpose=False,
            slicer="NONE",
        )

    if cache:
        cache_joint_spec(state, hhsize, spec)

    t0 = tracing.print_elapsed_time("build_cdap_joint_spec hh_size %s" % hhsize, t0)

    return spec


def add_interaction_column(choosers, p_tup):
    """
    Add an interaction column in place to choosers, listing the ptypes of the persons in p_tup

    The name of the interaction column will be determined by the cdap_ranks from p_tup,
    and the rows in the column contain the ptypes of those persons in that household row.

    For instance, for p_tup = (1,3) choosers interaction column name will be 'p1_p3'

    For a household where person 1 is part-time worker (ptype=2) and person 3 is infant (ptype 8)
    the corresponding row value interaction code will be 28

    We take advantage of the fact that interactions are symmetrical to simplify spec expressions:
    We name the interaction_column in increasing pnum (cdap_rank) order (p1_p2 and not p3_p1)
    And we format row values in increasing ptype order (28 and not 82)
    This simplifies the spec expressions as we don't have to test for p1_p3 == 28 | p1_p3 == 82

    Parameters
    ----------
    choosers : pandas.DataFrame
        household choosers, indexed on _hh_index_
        choosers should contain columns ptype_p1, ptype_p2 for each cdap_rank person in hh

    p_tup : int tuple
        tuple specifying the cdap_ranks for the interaction column
        p_tup = (1,3) means persons with cdap_rank 1 and 3

    Returns
    -------

    """

    # Since ptypes are always between 1 and 8, we represent the interaction as an integer (24)
    # rather than as a string ('24')
    # FIXME - check that coding interactions as integers is in fact faster then coding as strings

    if p_tup != tuple(sorted(p_tup)):
        raise RuntimeError("add_interaction_column tuple not sorted" % p_tup)

    # FIXME - this could be made more elegant and efficient
    # I couldn't figure out a good way to do this in pandas, but we want to do something like:
    # choosers['p1_p3'] = choosers['ptype_p1'].astype(str) + choosers['ptype_p3'].astype(str)

    dest_col = "_".join(["p%s" % pnum for pnum in p_tup])

    # build a string concatenating the ptypes of the persons in the order they appear in p_tup
    choosers[dest_col] = choosers[add_pn("ptype", p_tup[0])].astype(str)
    for pnum in p_tup[1:]:
        choosers[dest_col] = choosers[dest_col] + choosers[
            add_pn("ptype", pnum)
        ].astype(str)

    # sort the list of ptypes so it is in increasing ptype order, then convert to int
    choosers[dest_col] = (
        choosers[dest_col].apply(lambda x: "".join(sorted(x))).astype(int)
    )


def hh_choosers(state: workflow.State, indiv_utils, hhsize):
    """
    Build a chooser table for calculating house utilities for all households of specified hhsize

    The choosers table will have one row per household with columns containing the indiv_utils
    for all non-extra (i.e. cdap_rank <- MAX_HHSIZE) persons. That makes 3 columns for each
    individual. e.g. the utilities of person with cdap_rank 1 will be included as M_p1, N_p1, H_p1

    The chooser table will also contain interaction columns for all possible interactions involving
    from 2 to 3 persons (actually MAX_INTERACTION_CARDINALITY, which is currently 3).

    The interaction columns list the ptypes of the persons in the interaction set, sorted by ptype.
    For instance the interaction between persons with cdap_rank 1 and three and ptypes will
    be listed in a column named 'p1_p3' and for a household where persons p1 and p3 are 2 and 4
    will a row value of 24 in the p1_p3 column.

    Parameters
    ----------
    indiv_utils : pandas.DataFrame
        CDAP utilities for each individual, ignoring interactions.
        ind_utils has index of _persons_index_ and a column for each alternative
        i.e. three columns 'M' (Mandatory), 'N' (NonMandatory), 'H' (Home)

    hhsize : int
        household size for which the choosers table should be built. Households with more than
        MAX_HHSIZE members will be included with MAX_HHSIZE choosers since the are handled the
        same, and the activities of the extra members are assigned afterwards

    Returns
    -------
    choosers : pandas.DataFrame
        choosers households of hhsize with activity utility columns interaction columns
        for all (non-extra) household members
    """

    # we want to merge the ptype and M, N, and H utilities for each individual in the household
    merge_cols = [_hh_id_, _ptype_, "M", "N", "H"]

    # add attributes for joint tour utility
    from activitysim.abm.models.cdap import CdapSettings

    model_settings = CdapSettings.read_settings_file(state.filesystem, "cdap.yaml")
    additional_merge_cols = model_settings.JOINT_TOUR_USEFUL_COLUMNS
    if additional_merge_cols is not None:
        merge_cols.extend(additional_merge_cols)

    if hhsize > MAX_HHSIZE:
        raise RuntimeError("hh_choosers hhsize > MAX_HHSIZE")

    if hhsize < MAX_HHSIZE:
        include_households = indiv_utils[_hh_size_] == hhsize
    else:
        # we want to include larger households along with MAX_HHSIZE households
        include_households = indiv_utils[_hh_size_] >= MAX_HHSIZE

    # start with all the individuals with cdap_rank of 1 (thus there will be one row per household)
    choosers = indiv_utils.loc[
        include_households & (indiv_utils["cdap_rank"] == 1), merge_cols
    ]
    # rename columns, adding pn suffix (e.g. ptype_p1, M_p1) to all columns except hh_id
    choosers.columns = add_pn(merge_cols, 1)

    # for each of the higher cdap_ranks
    for pnum in range(2, hhsize + 1):
        # df with merge columns for indiv with cdap_rank of pnum
        rhs = indiv_utils.loc[
            include_households & (indiv_utils["cdap_rank"] == pnum), merge_cols
        ]
        # rename columns, adding pn suffix (e.g. ptype_p1, M_p1) to all columns except hh_id
        rhs.columns = add_pn(merge_cols, pnum)

        # merge this cdap_rank into choosers
        choosers = pd.merge(left=choosers, right=rhs, on=_hh_id_)

    # we set index to _hh_id_ choosers has one row per household
    set_hh_index(choosers)

    # coerce utilities to float (merge apparently makes column type objects)
    for pnum in range(1, hhsize + 1):
        pn_cols = add_pn(["M", "N", "H"], pnum)
        choosers[pn_cols] = choosers[pn_cols].astype(float)

    # add interaction columns for all 2 and 3 person interactions
    for i in range(2, min(hhsize, MAX_INTERACTION_CARDINALITY) + 1):
        for tup in itertools.combinations(list(range(1, hhsize + 1)), i):
            add_interaction_column(choosers, tup)

    # add hhsize
    choosers["hhsize"] = hhsize

    return choosers


def household_activity_choices(
    state: workflow.State,
    indiv_utils,
    interaction_coefficients,
    hhsize,
    trace_hh_id=None,
    trace_label=None,
    add_joint_tour_utility=False,
    *,
    chunk_sizer,
    compute_settings: ComputeSettings | None = None,
):
    """
    Calculate household utilities for each activity pattern alternative for households of hhsize
    The resulting activity pattern for each household will be coded as a string of activity codes.
    e.g. 'MNHH' for a 4 person household with activities Mandatory, NonMandatory, Home, Home

    Parameters
    ----------
    indiv_utils : pandas.DataFrame
        CDAP utilities for each individual, ignoring interactions
        ind_utils has index of _persons_index_ and a column for each alternative
        i.e. three columns 'M' (Mandatory), 'N' (NonMandatory), 'H' (Home)

    interaction_coefficients : pandas.DataFrame
        Rules and coefficients for generating interaction specs for different household sizes

    hhsize : int
        the size of household for which activity perttern should be calculated (1..MAX_HHSIZE)

    Returns
    -------
    choices : pandas.Series
        the chosen cdap activity pattern for each household represented as a string (e.g. 'MNH')
        with same index (_hh_index_) as utils

    """

    if hhsize == 1:
        # for 1 person households, there are no interactions to account for
        # and the household utils are the same as the individual utils
        choosers = vars = None
        # extract the individual utilities for individuals from hhsize 1 households
        utils = indiv_utils.loc[indiv_utils[_hh_size_] == 1, [_hh_id_, "M", "N", "H"]]
        # index on household_id, not person_id
        set_hh_index(utils)
    else:
        choosers = hh_choosers(state, indiv_utils, hhsize=hhsize)

        spec = build_cdap_spec(
            state,
            interaction_coefficients,
            hhsize,
            trace_spec=(trace_hh_id in choosers.index),
            trace_label=trace_label,
            joint_tour_alt=add_joint_tour_utility,
        )

        utils = simulate.eval_utilities(
            state,
            spec,
            choosers,
            trace_label=trace_label,
            have_trace_targets=(trace_hh_id in choosers.index),
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

    if len(utils.index) == 0:
        return pd.Series(dtype="float64")

    # calculate joint tour utility
    if add_joint_tour_utility & (hhsize > 1):
        # calculate joint utils
        joint_tour_spec = build_cdap_joint_spec(
            state,
            interaction_coefficients,
            hhsize,
            trace_spec=(trace_hh_id in choosers.index),
            trace_label=tracing.extend_trace_label(trace_label, "joint"),
        )

        joint_tour_utils = simulate.eval_utilities(
            state,
            joint_tour_spec,
            choosers,
            trace_label=tracing.extend_trace_label(trace_label, "joint"),
            have_trace_targets=(trace_hh_id in choosers.index),
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

        # add joint util to util
        utils = utils.add(joint_tour_utils)

    probs = logit.utils_to_probs(state, utils, trace_label=trace_label)

    # select an activity pattern alternative for each household based on probability
    # result is a series indexed on _hh_index_ with the (0 based) index of the column from probs
    idx_choices, rands = logit.make_choices(state, probs, trace_label=trace_label)

    # convert choice expressed as index into alternative name from util column label
    choices = pd.Series(utils.columns[idx_choices].values, index=utils.index)

    if trace_hh_id:
        if hhsize > 1:
            state.tracing.trace_df(
                choosers,
                "%s.hhsize%d_choosers" % (trace_label, hhsize),
                column_labels=["expression", "person"],
            )

        state.tracing.trace_df(
            utils,
            "%s.hhsize%d_utils" % (trace_label, hhsize),
            column_labels=["expression", "household"],
        )
        state.tracing.trace_df(
            probs,
            "%s.hhsize%d_probs" % (trace_label, hhsize),
            column_labels=["expression", "household"],
        )
        state.tracing.trace_df(
            choices,
            "%s.hhsize%d_activity_choices" % (trace_label, hhsize),
            column_labels=["expression", "household"],
        )
        state.tracing.trace_df(
            rands, "%s.hhsize%d_rands" % (trace_label, hhsize), columns=[None, "rand"]
        )

    return choices


def unpack_cdap_indiv_activity_choices(persons, hh_choices, trace_hh_id, trace_label):
    """
    Unpack the household activity choice list into choices for each (non-extra) household member

    Parameters
    ----------
    persons : pandas.DataFrame
        Table of persons data indexed on _persons_index_
        We expect, at least, columns [_hh_id_, 'cdap_rank']
    hh_choices : pandas.Series
        household activity pattern is encoded as a string (of length hhsize) of activity codes
        e.g. 'MNHH' for a 4 person household with activities Mandatory, NonMandatory, Home, Home

    Returns
    -------
    cdap_indiv_activity_choices : pandas.Series
        series contains one activity per individual hh member, indexed on _persons_index_
    """

    cdap_indivs = persons["cdap_rank"] <= MAX_HHSIZE

    indiv_activity = pd.merge(
        left=persons.loc[cdap_indivs, [_hh_id_, "cdap_rank"]],
        right=hh_choices.to_frame(name="hh_choices"),
        left_on=_hh_id_,
        right_index=True,
    )

    # resulting dataframe has columns _hh_id_,'cdap_rank', hh_choices indexed on _persons_index_

    indiv_activity["cdap_activity"] = ""

    # for each cdap_rank (1..5)
    for i in range(MAX_HHSIZE):
        pnum_i = indiv_activity["cdap_rank"] == i + 1
        indiv_activity.loc[pnum_i, ["cdap_activity"]] = indiv_activity[pnum_i][
            "hh_choices"
        ].str[i]

    cdap_indiv_activity_choices = indiv_activity["cdap_activity"]

    # if DUMP:
    #     state.tracing.trace_df(cdap_indiv_activity_choices,
    #                      '%s.DUMP.cdap_indiv_activity_choices' % trace_label,
    #                      transpose=False, slicer='NONE')

    return cdap_indiv_activity_choices


def extra_hh_member_choices(
    state: workflow.State,
    persons,
    cdap_fixed_relative_proportions: pd.DataFrame,
    locals_d,
    trace_hh_id,
    trace_label,
):
    """
    Generate the activity choices for the 'extra' household members who weren't handled by cdap

    Following the CTRAMP HouseholdCoordinatedDailyActivityPatternModel, "a separate,
    simple cross-sectional distribution is looked up for the remaining household members"

    The cdap_fixed_relative_proportions spec is handled like an activitysim logit utility spec,
    EXCEPT that the values computed are relative proportions, not utilities
    (i.e. values are not exponentiated before being normalized to probabilities summing to 1.0)

    Parameters
    ----------
    persons : pandas.DataFrame
        Table of persons data indexed on _persons_index_
         We expect, at least, columns [_hh_id_, _ptype_]
    cdap_fixed_relative_proportions
        spec to compute/specify the relative proportions of each activity (M, N, H)
        that should be used to choose activities for additional household members
        not handled by CDAP.
    locals_d : Dict
        dictionary of local variables that eval_variables adds to the environment
        for an evaluation of an expression that begins with @

    Returns
    -------
    choices : pandas.Series
        list of alternatives chosen for all extra members, indexed by _persons_index_
    """

    trace_label = tracing.extend_trace_label(trace_label, "extra_hh_member_choices")

    # extra household members have cdap_ran > MAX_HHSIZE
    choosers = persons[persons["cdap_rank"] > MAX_HHSIZE]

    if len(choosers.index) == 0:
        return pd.Series(dtype="float64")

    # eval the expression file
    values = simulate.eval_variables(
        state, cdap_fixed_relative_proportions.index, choosers, locals_d
    )

    # cdap_fixed_relative_proportions computes relative proportions by ptype, not utilities
    proportions = values.dot(cdap_fixed_relative_proportions)

    # convert relative proportions to probability
    probs = proportions.div(proportions.sum(axis=1), axis=0)

    # select an activity pattern alternative for each person based on probability
    # idx_choices is a series (indexed on _persons_index_ ) with the chosen alternative represented
    # as the integer (0 based) index of the chosen column from probs
    idx_choices, rands = logit.make_choices(state, probs, trace_label=trace_label)

    # convert choice from column index to activity name
    choices = pd.Series(probs.columns[idx_choices].values, index=probs.index)

    # if DUMP:
    #     state.tracing.trace_df(proportions, '%s.DUMP.extra_proportions' % trace_label,
    #                      transpose=False, slicer='NONE')
    #     state.tracing.trace_df(probs, '%s.DUMP.extra_probs' % trace_label,
    #                      transpose=False, slicer='NONE')
    #     state.tracing.trace_df(choices, '%s.DUMP.extra_choices' % trace_label,
    #                      transpose=False,
    #                      slicer='NONE')

    if trace_hh_id:
        state.tracing.trace_df(
            proportions,
            "%s.extra_hh_member_choices_proportions" % trace_label,
            column_labels=["expression", "person"],
        )
        state.tracing.trace_df(
            probs,
            "%s.extra_hh_member_choices_probs" % trace_label,
            column_labels=["expression", "person"],
        )
        state.tracing.trace_df(
            choices,
            "%s.extra_hh_member_choices_choices" % trace_label,
            column_labels=["expression", "person"],
        )
        state.tracing.trace_df(
            rands,
            "%s.extra_hh_member_choices_rands" % trace_label,
            columns=[None, "rand"],
        )

    return choices


def _run_cdap(
    state: workflow.State,
    persons,
    person_type_map,
    cdap_indiv_spec,
    interaction_coefficients,
    cdap_fixed_relative_proportions,
    locals_d,
    trace_hh_id,
    trace_label,
    add_joint_tour_utility,
    *,
    chunk_sizer,
    compute_settings: ComputeSettings | None = None,
) -> pd.DataFrame | tuple:
    """
    Implements core run_cdap functionality on persons df (or chunked subset thereof)
    Aside from chunking of persons df, params are passed through from run_cdap unchanged

    Returns pandas Dataframe with two columns:
        cdap_activity : str
            activity for that person expressed as 'M', 'N', 'H'
        cdap_rank : int
            activities for persons with cdap_rank <= MAX_HHSIZE are determined by cdap
            'extra' household members activities are assigned by cdap_fixed_relative_proportions
    """

    # assign integer cdap_rank to each household member
    # persons with cdap_rank 1..MAX_HHSIZE will be have their activities chose by CDAP model
    # extra household members, will have activities assigned by in fixed proportions
    assign_cdap_rank(state, persons, person_type_map, trace_hh_id, trace_label)
    chunk_sizer.log_df(trace_label, "persons", persons)

    # Calculate CDAP utilities for each individual, ignoring interactions
    # ind_utils has index of 'person_id' and a column for each alternative
    # i.e. three columns 'M' (Mandatory), 'N' (NonMandatory), 'H' (Home)
    indiv_utils = individual_utilities(
        state,
        persons[persons.cdap_rank <= MAX_HHSIZE],
        cdap_indiv_spec,
        locals_d,
        trace_hh_id,
        trace_label,
        chunk_sizer=chunk_sizer,
        compute_settings=compute_settings,
    )
    chunk_sizer.log_df(trace_label, "indiv_utils", indiv_utils)

    # compute interaction utilities, probabilities, and hh activity pattern choices
    # for each size household separately in turn up to MAX_HHSIZE
    hh_choices_list = []
    for hhsize in range(1, MAX_HHSIZE + 1):
        choices = household_activity_choices(
            state,
            indiv_utils,
            interaction_coefficients,
            hhsize=hhsize,
            trace_hh_id=trace_hh_id,
            trace_label=trace_label,
            add_joint_tour_utility=add_joint_tour_utility,
            chunk_sizer=chunk_sizer,
            compute_settings=compute_settings,
        )

        hh_choices_list.append(choices)

    del indiv_utils
    chunk_sizer.log_df(trace_label, "indiv_utils", None)

    # concat all the household choices into a single series indexed on _hh_index_
    hh_activity_choices = pd.concat(hh_choices_list)
    chunk_sizer.log_df(trace_label, "hh_activity_choices", hh_activity_choices)

    # unpack the household activity choice list into choices for each (non-extra) household member
    # resulting series contains one activity per individual hh member, indexed on _persons_index_
    cdap_person_choices = unpack_cdap_indiv_activity_choices(
        persons, hh_activity_choices, trace_hh_id, trace_label
    )

    # assign activities to extra household members (with cdap_rank > MAX_HHSIZE)
    # resulting series contains one activity per individual hh member, indexed on _persons_index_
    extra_person_choices = extra_hh_member_choices(
        state,
        persons,
        cdap_fixed_relative_proportions,
        locals_d,
        trace_hh_id,
        trace_label,
    )

    # concat cdap and extra persoin choices into a single series
    # this series will be the same length as the persons dataframe and be indexed on _persons_index_

    person_choices = pd.concat([cdap_person_choices, extra_person_choices])

    persons["cdap_activity"] = person_choices
    chunk_sizer.log_df(trace_label, "persons", persons)

    # return household joint tour flag
    if add_joint_tour_utility:
        hh_activity_choices = hh_activity_choices.to_frame(name="hh_choices")
        hh_activity_choices["has_joint_tour"] = hh_activity_choices["hh_choices"].apply(
            lambda x: 1 if "J" in x else 0
        )

    # if DUMP:
    #     state.tracing.trace_df(hh_activity_choices, '%s.DUMP.hh_activity_choices' % trace_label,
    #                      transpose=False, slicer='NONE')
    #     state.tracing.trace_df(cdap_results, '%s.DUMP.cdap_results' % trace_label,
    #                      transpose=False, slicer='NONE')

    result = persons[["cdap_rank", "cdap_activity"]]

    del persons
    chunk_sizer.log_df(trace_label, "persons", None)

    if add_joint_tour_utility:
        return result, hh_activity_choices["has_joint_tour"]
    else:
        return result


def run_cdap(
    state: workflow.State,
    persons,
    person_type_map,
    cdap_indiv_spec,
    cdap_interaction_coefficients,
    cdap_fixed_relative_proportions,
    locals_d,
    chunk_size=0,
    trace_hh_id=None,
    trace_label=None,
    add_joint_tour_utility=False,
    compute_settings: ComputeSettings | None = None,
):
    """
    Choose individual activity patterns for persons.

    Parameters
    ----------
    persons : pandas.DataFrame
        Table of persons data. Must contain at least a household ID, household size,
        person type category, and age, plus any columns used in cdap_indiv_spec
    cdap_indiv_spec : pandas.DataFrame
        CDAP spec for individuals without taking any interactions into account.
    cdap_interaction_coefficients : pandas.DataFrame
        Rules and coefficients for generating interaction specs for different household sizes
    cdap_fixed_relative_proportions : pandas.DataFrame
        Spec to for the relative proportions of each activity (M, N, H)
        to choose activities for additional household members not handled by CDAP
    locals_d : Dict
        This is a dictionary of local variables that will be the environment
        for an evaluation of an expression that begins with @
        in either the cdap_indiv_spec or cdap_fixed_relative_proportions expression files
    chunk_size: int
        Chunk size or 0 for no chunking
    trace_hh_id : int
        hh_id to trace or None if no hh tracing
    trace_label : str
        label for tracing or None if no tracing
    add_joint_tour_utility : Bool
        cdap model include joint tour utility or not

    Returns
    -------
    choices : pandas.DataFrame

        dataframe is indexed on _persons_index_ and has two columns:

        cdap_activity : str
            activity for that person expressed as 'M', 'N', 'H'
    """

    trace_label = tracing.extend_trace_label(trace_label, "cdap")

    cdap_results = hh_choice_results = None
    result_list = []
    # segment by person type and pick the right spec for each person type
    for (
        i,
        persons_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers_by_chunk_id(state, persons, trace_label):
        if add_joint_tour_utility:
            cdap_results, hh_choice_results = _run_cdap(
                state,
                persons_chunk,
                person_type_map,
                cdap_indiv_spec,
                cdap_interaction_coefficients,
                cdap_fixed_relative_proportions,
                locals_d,
                trace_hh_id,
                chunk_trace_label,
                add_joint_tour_utility,
                chunk_sizer=chunk_sizer,
                compute_settings=compute_settings,
            )
        else:
            cdap_results = _run_cdap(
                state,
                persons_chunk,
                person_type_map,
                cdap_indiv_spec,
                cdap_interaction_coefficients,
                cdap_fixed_relative_proportions,
                locals_d,
                trace_hh_id,
                chunk_trace_label,
                add_joint_tour_utility,
                chunk_sizer=chunk_sizer,
                compute_settings=compute_settings,
            )

        result_list.append(cdap_results)

        chunk_sizer.log_df(trace_label, "result_list", result_list)

    # FIXME: this will require 2X RAM
    # if necessary, could append to hdf5 store on disk:
    # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
    if len(result_list) > 1:
        cdap_results = pd.concat(result_list)

    if trace_hh_id:
        state.tracing.trace_df(
            cdap_results,
            label="cdap",
            columns=["cdap_rank", "cdap_activity"],
            warn_if_empty=True,
        )

    # return choices column as series
    if add_joint_tour_utility:
        return cdap_results["cdap_activity"], hh_choice_results
    else:
        # return choices column as series
        return cdap_results["cdap_activity"]
