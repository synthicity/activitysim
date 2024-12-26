from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from larch import Model, P, X  # noqa: F401

# from larch.log import logger_name
# from larch.model.abstract_model import AbstractChoiceModel
from larch.model.tree import NestingTree
from larch.util import Dict  # noqa: F401

_logger = logging.getLogger("larch")


def cv_to_ca(alt_values, dtype="float64", required_labels=None):
    """
    Convert a choosers-variables DataFrame to an idca DataFrame.

    Parameters
    ----------
    alt_values : pandas.DataFrame
        This DataFrame should be in choosers-variables format,
        with one row per chooser and variable, and one column per
        alternative.  The id's for the choosers and variables
        must be in that order, in a two-level MultiIndex.
    dtype : dtype
        Convert the incoming data to this type.  Set to None to
        skip data conversion.
    required_labels : Collection, optional
        If given, any columns in the output that are not required
        will be pre-emptively dropped.

    Returns
    -------
    pandas.DataFrame
        The resulting DataFrame is transformed into Larch's
        idca format, with one row per chooser (case) and
        alternative, and one column per variable.
    """

    # Read the source file, converting to a tall stack of
    # data with a 3 level multiindex
    x_ca_tall = alt_values.stack()

    # Set the last level index name to 'altid'
    x_ca_tall.index.rename("altid", level=-1, inplace=True)
    c_, v_, a_ = x_ca_tall.index.names

    # case and alt id's should be integers.
    # Easier to manipulate dtypes of a multiindex if we just convert
    # it to columns first.  If this ends up being a performance issue
    # we can look at optimizing this in the future.
    x_ca_tall = x_ca_tall.reset_index()
    x_ca_tall[c_] = x_ca_tall[c_].astype(np.int64)
    x_ca_tall[a_] = x_ca_tall[a_].astype(np.int64)
    x_ca_tall = x_ca_tall.set_index([c_, v_, a_])

    if dtype is not None:
        # Convert instances where data is 'False' or 'True' to numbers
        x_ca_tall[x_ca_tall == "False"] = 0
        x_ca_tall[x_ca_tall == "True"] = 1

        if required_labels is None:
            # Convert data to float64 to optimize computation speed in larch
            x_ca_tall = x_ca_tall.astype(dtype)

    # Unstack the variables dimension
    x_ca = (
        x_ca_tall.reset_index()
        .drop_duplicates()
        .set_index(x_ca_tall.index.names)
        .unstack(1)
    )

    # Code above added a dummy top level to columns, remove it here.
    x_ca.columns = x_ca.columns.droplevel(0)

    if required_labels is not None:
        reqrd = set(required_labels.to_numpy())
        drop_cols = [c for c in x_ca.columns if c not in reqrd]
        x_ca = x_ca.drop(drop_cols, axis=1).astype(dtype)

    return x_ca


def prevent_overlapping_column_names(x_ca, x_co):
    """
    Rename columns in idca data to prevent overlapping names.

    Parameters
    ----------
    x_ca, x_co : pandas.DataFrame
        The idca and idco data, respectively

    Returns
    -------
    x_ca, x_co
    """
    renaming = {i: f"{i}_ca" for i in x_ca.columns if i in x_co.columns}
    x_ca.rename(columns=renaming, inplace=True)
    return x_ca, x_co


def str_repr(x):
    if isinstance(x, str):
        return repr(x)
    return x


def linear_utility_from_spec(
    spec, x_col, p_col, ignore_x=(), segment_id=None, x_validator=None, expr_col=None
):
    """
    Create a linear function from a spec DataFrame.

    Parameters
    ----------
    spec : pandas.DataFrame
        A spec for an ActivitySim model.
    x_col: str
        The name of the columns in spec representing the data.
    p_col: str or dict
        The name of the columns in spec representing the parameters.
        Give as a string for a single column, or as a dict to
        have segments on multiple columns. If given as a dict,
        the keys give the names of the columns to use, and the
        values give the identifiers that will need to match the
        loaded `segment_id` value.
    ignore_x : Collection, optional
        Labels in the spec file to ignore.  Typically this
        includes variables that are pre-processed by ActivitySim
        and therefore don't need to be made available in Larch.
    segment_id : str, optional
        The CHOOSER_SEGMENT_COLUMN_NAME identified for ActivitySim.
        This value is ignored if `p_col` is a string, and required
        if `p_col` is a dict.
    x_validator : Container, optional
        A container of valid values for the x_col.  If given, the
        x_col values will be used if they are `in` the x_validator,
        otherwise the value from `expr_col` will be used.
    expr_col : str, optional
        The name of the column to use when the x_col value is not
        in the x_validator.

    Returns
    -------
    LinearFunction_C
    """
    if isinstance(p_col, dict):
        if segment_id is None:
            raise ValueError("segment_id must be given if p_col is a dict")
        partial_utility = {}
        for seg_p_col, segval in p_col.items():
            partial_utility[seg_p_col] = linear_utility_from_spec(
                spec,
                x_col,
                seg_p_col,
                ignore_x,
                x_validator=x_validator,
                expr_col=expr_col,
            ) * X(f"{segment_id}=={str_repr(segval)}")
        return sum(partial_utility.values())
    parts = []
    for i in spec.index:
        _x = spec.loc[i, x_col]
        try:
            _x = _x.strip()
        except AttributeError:
            if np.isnan(_x):
                _x = None
            else:
                raise

        # when a validator is given, use the expression column if the original
        # x value is not in the validator
        if _x is not None and _x not in ignore_x:
            if x_validator is not None and _x not in x_validator:
                _x = spec.loc[i, expr_col]
                if _x.startswith("@"):
                    _x = _x[1:]

        # handle the parameter...
        _p = spec.loc[i, p_col]

        if _x is not None and (_x not in ignore_x) and not pd.isna(_p):
            # process coefficients when they are multiples instead of raw names
            if isinstance(_p, str) and "*" in _p:
                _p_star = [i.strip() for i in _p.split("*")]
                if len(_p_star) == 2:
                    try:
                        _p0 = float(_p_star[0])
                    except ValueError:
                        # first term not a number, maybe the second is
                        try:
                            _p1 = float(_p_star[1])
                        except ValueError:
                            # second term also not a number, it's just a star in a name
                            _P = P(_p)
                        else:
                            # second term is a number, use the multiplier
                            _P = P(_p_star[0]) * _p1
                    else:
                        # first term is a number, ensure the second is not
                        try:
                            _p1 = float(_p_star[1])
                        except ValueError:
                            # second term is not a number, use the multiplier
                            _P = P(_p_star[1]) * _p0
                        else:
                            # both terms are numbers, not allowed
                            raise ValueError(f"parameter is just {_p}, I need a name")
                else:
                    # not handling triple-multiple terms (or worse)
                    _P = P(_p)
            else:
                _P = P(_p)
            parts.append(_P * X(_x))
    return sum(parts)


def dict_of_linear_utility_from_spec(
    spec, x_col, p_col, ignore_x=(), x_validator=None, expr_col=None
):
    """
    Create a linear function from a spec DataFrame.

    Parameters
    ----------
    spec : pandas.DataFrame
        A spec for an ActivitySim model.
    x_col: str
        The name of the columns in spec representing the data.
    p_col: dict
        The name of the columns in spec representing the parameters.
        The keys give the names of the columns to use, and the
        values will become the keys of the output dictionary.
    ignore_x : Collection, optional
        Labels in the spec file to ignore.  Typically this
        includes variables that are pre-processed by ActivitySim
        and therefore don't need to be made available in Larch.
    segment_id : str, optional
        The CHOOSER_SEGMENT_COLUMN_NAME identified for ActivitySim.
        This value is ignored if `p_col` is a string, and required
        if `p_col` is a dict.
    x_validator : Container, optional
        A container of valid values for the x_col.  If given, the
        x_col values will be used if they are `in` the x_validator,
        otherwise the value from `expr_col` will be used.
    expr_col : str, optional
        The name of the column to use when the x_col value is not
        in the x_validator.

    Returns
    -------
    dict
    """
    utils = {}
    for altname, altcode in p_col.items():
        utils[altcode] = linear_utility_from_spec(
            spec,
            x_col,
            altname,
            ignore_x=ignore_x,
            x_validator=x_validator,
            expr_col=expr_col,
        )
    return utils


def explicit_value_parameters_from_spec(spec, p_col, model):
    """
    Define and lock parameters given as fixed values in the spec.

    Parameters
    ----------
    spec : pandas.DataFrame
        A spec for an ActivitySim model.
    p_col : str or dict
        The name of the columns in spec representing the parameters.
        Give as a string for a single column, or as a dict to
        have segments on multiple columns. If given as a dict,
        the keys give the names of the columns to use, and the
        values give the identifiers that will need to match the
        loaded `segment_id` value.  Only the keys are used in this
        function.
    model : larch.Model
        The model to insert fixed value parameters.

    Returns
    -------

    """
    if isinstance(p_col, dict):
        for p_col_ in p_col:
            explicit_value_parameters_from_spec(spec, p_col_, model)
    else:
        for i in spec.itertuples():
            try:
                j = float(getattr(i, p_col))
            except Exception:
                pass
            else:
                model.set_value(
                    getattr(i, p_col),
                    value=j,
                    holdfast=True,
                )


def explicit_value_parameters(model):
    """
    Define and lock parameters given as fixed values.

    Parameters
    ----------
    model : larch.Model
        The model to insert fixed value parameters.

    Returns
    -------

    """
    for i in model.pf.index:
        try:
            j = float(i)
        except Exception:
            pass
        else:
            model.lock_value(i, value=j)


def apply_coefficients(coefficients, model, minimum=None, maximum=None):
    """
    Read the coefficients CSV file to a DataFrame and set model parameters.

    Parameters
    ----------
    coefficients : pandas.DataFrame
        The coefficients table in the ActivitySim data bundle
        for this model.
    model : Model
        Apply coefficient values and constraints to this model.

    """
    if isinstance(model, dict):
        for m in model.values():
            apply_coefficients(coefficients, m)
    else:
        assert isinstance(coefficients, pd.DataFrame)
        assert "value" in coefficients.columns
        if "constrain" not in coefficients.columns:
            import warnings

            warnings.warn(
                "coefficient dataframe missing 'constrain' column, setting all to 'F'"
            )
            coefficients["constrain"] = "F"
        assert coefficients.index.name == "coefficient_name"
        # assert isinstance(model, AbstractChoiceModel)
        explicit_value_parameters(model)
        for i in coefficients.itertuples():
            if i.Index in model.pnames:
                holdfast = i.constrain == "T"
                if holdfast:
                    model.lock_value(i.Index, value=i.value)
                else:
                    model.set_value(
                        i.Index,
                        value=i.value,
                        initvalue=i.value,
                        holdfast=holdfast,
                        minimum=minimum,
                        maximum=maximum,
                    )


def apply_coef_template(linear_utility, template_col, condition=None):
    """
    Apply a coefficient template over a linear utility function.

    Parameters
    ----------
    linear_utility : LinearFunction_C
    template_col : Mapping
    condition : any

    Returns
    -------
    LinearFunction_C
    """
    result = sum(
        P("*".join(template_col.get(ip, ip) for ip in i.param.split("*")))
        * i.data
        * i.scale
        for i in linear_utility
    )
    if condition is not None:
        result = result * condition
    return result


def construct_nesting_tree(alternatives, nesting_settings):
    """
    Construct a NestingTree from ActivitySim settings.

    Parameters
    ----------
    alternatives : Mapping or Sequence
        If given as a Mapping (dict), the keys are the alternative names
        as strings, and the values are alternative code numbers to use
        in larch.  If given as a Sequence, the values are the alternative
        names, and unique sequential codes will be created starting from 1.
    nesting_settings : Mapping
        The 'NESTS' section of the ActivitySim config file.

    Returns
    -------
    NestingTree
    """
    if not isinstance(alternatives, Mapping):
        alt_names = list(alternatives)
        alt_codes = np.arange(1, len(alt_names) + 1)
        alternatives = dict(zip(alt_names, alt_codes))

    tree = NestingTree()
    nest_names_to_codes = alternatives.copy()
    nest_names_to_codes["root"] = 0
    for alt_name, alt_code in alternatives.items():
        tree.add_node(alt_code, name=alt_name)

    def make_nest(cfg, parent_code=0):
        nonlocal nest_names_to_codes
        if cfg["name"] != "root":
            if cfg["name"] not in nest_names_to_codes:
                n = tree.new_node(
                    name=cfg["name"],
                    parameter=str(cfg["coefficient"]),
                    parent=parent_code,
                )
                nest_names_to_codes[cfg["name"]] = n
            else:
                tree.add_edge(parent_code, nest_names_to_codes[cfg["name"]])
        for a in cfg["alternatives"]:
            if isinstance(a, str):
                tree.add_edge(nest_names_to_codes[cfg["name"]], nest_names_to_codes[a])
            else:
                make_nest(a, parent_code=nest_names_to_codes[cfg["name"]])

    if nesting_settings:
        make_nest(nesting_settings)

    return tree


def remove_apostrophes(df, from_columns=None):
    """
    Remove apostrophes from columns names and from data in given columns.

    Also strips leading and trailing whitespace.

    This function operates in-place on DataFrames.

    Parameters
    ----------
    df : pandas.DataFrame
    from_columns : Collection, optional

    Returns
    -------
    pandas.DataFrame
    """
    df.columns = df.columns.str.replace("'", "")
    df.columns = df.columns.str.strip()
    if from_columns:
        for c in from_columns:
            df[c] = df[c].str.replace("'", "")
    return df


def clean_values(
    values,
    choice_col="override_choice",
    alt_names_to_codes=None,
    choice_code="override_choice_code",
):
    """

    Parameters
    ----------
    values : pd.DataFrame
    alt_names : Collection
    override_choice : str
        The columns of `values` containing the observed choices.
    alt_names_to_codes : Mapping, optional
        If the `override_choice` column contains alternative names,
        use this mapping to convert the names into alternative code
        numbers.
    choice_code : str, default 'override_choice_code'
        The name of the observed choice code number column that will
        be added to `values`.

    Returns
    -------
    pd.DataFrame
    """
    values = remove_apostrophes(values)
    values.fillna(0, inplace=True)
    # Retain only choosers with valid observed choice
    values = values[values[choice_col].isin(list(alt_names_to_codes.keys()))]
    # Convert choices to code numbers
    if alt_names_to_codes is not None:
        values[choice_code] = values[choice_col].map(dict(alt_names_to_codes))
    else:
        values[choice_code] = values[choice_col]
    return values


def update_coefficients(
    model, data, result_dir=Path("."), output_file=None, relabel_coef=None
):
    if isinstance(data, pd.DataFrame):
        coefficients = data.copy()
    else:
        coefficients = data.coefficients.copy()
    if relabel_coef is not None and len(relabel_coef):
        for j in coefficients.index:
            if j in model.pf.index:
                coefficients.loc[j, "value"] = model.pf.loc[j, "value"]
            else:
                j_ = relabel_coef.get(j, None)
                if j_ is not None and j_ in model.pf.index:
                    coefficients.loc[j, "value"] = model.pf.loc[j_, "value"]
    else:
        est_names = [j for j in coefficients.index if j in model.pf.index]
        coefficients.loc[est_names, "value"] = model.pf.loc[est_names, "value"]
    if output_file is not None:
        os.makedirs(result_dir, exist_ok=True)
        coefficients.reset_index().to_csv(
            result_dir / output_file,
            index=False,
        )
    return coefficients
