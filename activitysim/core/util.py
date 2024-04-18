# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import argparse
import collections
import itertools
import logging
import numbers
import os
from collections.abc import Iterable
from operator import itemgetter
from pathlib import Path
from typing import Optional, TypeVar

import cytoolz as tz
import cytoolz.curried
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def si_units(x, kind="B", digits=3, shift=1000):
    #       nano micro milli    kilo mega giga tera peta exa  zeta yotta
    tiers = ["n", "µ", "m", "", "K", "M", "G", "T", "P", "E", "Z", "Y"]

    tier = 3
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x > 0:
        while x > shift and tier < len(tiers):
            x /= shift
            tier += 1
        while x < 1 and tier >= 0:
            x *= shift
            tier -= 1
    return f"{sign}{round(x,digits)} {tiers[tier]}{kind}"


def GB(bytes):
    return si_units(bytes, kind="B", digits=1)


def SEC(seconds):
    return si_units(seconds, kind="s", digits=2)


def INT(x):
    # format int as camel case (e.g. 1000000 vecomes '1_000_000')
    negative = x < 0
    x = abs(int(x))
    result = ""
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = "_%03d%s" % (r, result)
    result = "%d%s" % (x, result)

    return f"{'-' if negative else ''}{result}"


def delete_files(file_list, trace_label):
    # delete files in file_list

    file_list = [file_list] if isinstance(file_list, str) else file_list
    for file_path in file_list:
        try:
            if os.path.isfile(file_path):
                logger.debug(f"{trace_label} deleting {file_path}")
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"{trace_label} exception (e) trying to delete {file_path}")


def df_size(df):
    bytes = 0 if df.empty else df.memory_usage(index=True).sum()
    return "%s %s" % (df.shape, GB(bytes))


def iprod(ints):
    """
    Return the product of hte ints in the list or tuple as an unlimited precision python int

    Specifically intended to compute arrray/buffer size for skims where np.proc might overflow for default dtypes.
    (Narrowing rules for np.prod are different on Windows and linux)
    an alternative to the unwieldy: int(np.prod(ints, dtype=np.int64))

    Parameters
    ----------
    ints: list or tuple of ints or int wannabees

    Returns
    -------
    returns python int
    """
    assert len(ints) > 0
    return int(np.prod(ints, dtype=np.int64))


def left_merge_on_index_and_col(left_df, right_df, join_col, target_col):
    """
    like pandas left merge, but join on both index and a specified join_col

    FIXME - for now return a series of ov values from specified right_df target_col

    Parameters
    ----------
    left_df : pandas DataFrame
        index name assumed to be same as that of right_df
    right_df : pandas DataFrame
        index name assumed to be same as that of left_df
    join_col : str
        name of column to join on (in addition to index values)
        should have same name in both dataframes
    target_col : str
        name of column from right_df whose joined values should be returned as series

    Returns
    -------
    target_series : pandas Series
        series of target_col values with same index as left_df
        i.e. values joined to left_df from right_df with index of left_df
    """
    assert left_df.index.name == right_df.index.name

    # want to know name previous index column will have after reset_index
    idx_col = right_df.index.name

    # SELECT target_col FROM full_sample LEFT JOIN unique_sample on idx_col, join_col
    merged = pd.merge(
        left_df[[join_col]].reset_index(),
        right_df[[join_col, target_col]].reset_index(),
        on=[idx_col, join_col],
        how="left",
    )

    merged.set_index(idx_col, inplace=True)

    return merged[target_col]


def reindex(series1, series2):
    """
    This reindexes the first series by the second series.  This is an extremely
    common operation that does not appear to  be in Pandas at this time.
    If anyone knows of an easier way to do this in Pandas, please inform the
    UrbanSim developers.

    The canonical example would be a parcel series which has an index which is
    parcel_ids and a value which you want to fetch, let's say it's land_area.
    Another dataset, let's say of buildings has a series which indicate the
    parcel_ids that the buildings are located on, but which does not have
    land_area.  If you pass parcels.land_area as the first series and
    buildings.parcel_id as the second series, this function returns a series
    which is indexed by buildings and has land_area as values and can be
    added to the buildings dataset.

    In short, this is a join on to a different table using a foreign key
    stored in the current table, but with only one attribute rather than
    for a full dataset.

    This is very similar to the pandas "loc" function or "reindex" function,
    but neither of those functions return the series indexed on the current
    table.  In both of those cases, the series would be indexed on the foreign
    table and would require a second step to change the index.

    Parameters
    ----------
    series1, series2 : pandas.Series

    Returns
    -------
    reindexed : pandas.Series

    """

    result = series1.reindex(series2)
    try:
        result.index = series2.index
    except AttributeError:
        pass
    return result

    # return pd.Series(series1.loc[series2.values].values, index=series2.index)


def reindex_i(series1, series2, dtype=np.int8):
    """
    version of reindex that replaces missing na values and converts to int
    helpful in expression files that compute counts (e.g. num_work_tours)
    """
    return reindex(series1, series2).fillna(0).astype(dtype)


def other_than(groups, bools):
    """
    Construct a Series that has booleans indicating the presence of
    something- or someone-else with a certain property within a group.

    Parameters
    ----------
    groups : pandas.Series
        A column with the same index as `bools` that defines the grouping
        of `bools`. The `bools` Series will be used to index `groups` and
        then the grouped values will be counted.
    bools : pandas.Series
        A boolean Series indicating where the property of interest is present.
        Should have the same index as `groups`.

    Returns
    -------
    others : pandas.Series
        A boolean Series with the same index as `groups` and `bools`
        indicating whether there is something- or something-else within
        a group with some property (as indicated by `bools`).

    """
    counts = groups[bools].value_counts()
    merge_col = groups.to_frame(name="right")
    pipeline = tz.compose(
        tz.curry(pd.Series.fillna, value=False),
        itemgetter("left"),
        tz.curry(
            pd.DataFrame.merge,
            right=merge_col,
            how="right",
            left_index=True,
            right_on="right",
        ),
        tz.curry(pd.Series.to_frame, name="left"),
    )
    gt0 = pipeline(counts > 0)
    gt1 = pipeline(counts > 1)

    return gt1.where(bools, other=gt0)


def quick_loc_df(loc_list, target_df, attribute=None):
    """
    faster replacement for target_df.loc[loc_list] or target_df.loc[loc_list][attribute]

    pandas DataFrame.loc[] indexing doesn't scale for large arrays (e.g. > 1,000,000 elements)

    Parameters
    ----------
    loc_list : list-like (numpy.ndarray, pandas.Int64Index, or pandas.Series)
    target_df : pandas.DataFrame containing column named attribute
    attribute : name of column from loc_list to return (or none for all columns)

    Returns
    -------
        pandas.DataFrame or, if attribbute specified, pandas.Series
    """
    if attribute:
        target_df = target_df[[attribute]]

    df = target_df.reindex(loc_list)

    df.index.name = target_df.index.name

    if attribute:
        # return series
        return df[attribute]
    else:
        # return df
        return df


def quick_loc_series(loc_list, target_series):
    """
    faster replacement for target_series.loc[loc_list]

    pandas Series.loc[] indexing doesn't scale for large arrays (e.g. > 1,000,000 elements)

    Parameters
    ----------
    loc_list : list-like (numpy.ndarray, pandas.Int64Index, or pandas.Series)
    target_series : pandas.Series

    Returns
    -------
        pandas.Series
    """

    left_on = "left"

    if isinstance(loc_list, pd.Index):
        left_df = pd.DataFrame({left_on: loc_list.values})
    elif isinstance(loc_list, pd.Series):
        left_df = loc_list.to_frame(name=left_on)
    elif isinstance(loc_list, np.ndarray) or isinstance(loc_list, list):
        left_df = pd.DataFrame({left_on: loc_list})
    else:
        raise RuntimeError(
            "quick_loc_series loc_list of unexpected type %s" % type(loc_list)
        )

    df = pd.merge(
        left_df,
        target_series.to_frame(name="right"),
        left_on=left_on,
        right_index=True,
        how="left",
    )

    # regression test
    # assert list(df.right) == list(target_series.loc[loc_list])

    return df.right


def assign_in_place(df, df2, downcast_int=False, downcast_float=False):
    """
    update existing row values in df from df2, adding columns to df if they are not there

    Parameters
    ----------
    df : pd.DataFrame
        assignment left-hand-side (dest)
    df2: pd.DataFrame
        assignment right-hand-side (source)
    downcast_int: bool
        if True, downcast int columns if possible
    downcast_float: bool
        if True, downcast float columns if possible
    Returns
    -------

    """

    # expect no rows in df2 that are not in df
    assert len(df2.index.difference(df.index)) == 0

    # update common columns in place
    common_columns = df2.columns.intersection(df.columns)
    if len(common_columns) > 0:
        old_dtypes = [df[c].dtype for c in common_columns]
        df.update(df2)

        # avoid needlessly changing int columns to float
        # this is a hack fix for a bug in pandas.update
        # github.com/pydata/pandas/issues/4094
        for c, old_dtype in zip(common_columns, old_dtypes):
            # if both df and df2 column were same type, but result is not
            if (old_dtype == df2[c].dtype) and (df[c].dtype != old_dtype):
                try:
                    df[c] = df[c].astype(old_dtype)
                except ValueError:
                    logger.warning(
                        "assign_in_place changed dtype %s of column %s to %s"
                        % (old_dtype, c, df[c].dtype)
                    )

            if isinstance(old_dtype, pd.api.types.CategoricalDtype):
                continue

            # if both df and df2 column were ints, but result is not
            if (
                np.issubdtype(old_dtype, np.integer)
                and np.issubdtype(df2[c].dtype, np.integer)
                and not np.issubdtype(df[c].dtype, np.integer)
            ):
                try:
                    df[c] = df[c].astype(old_dtype)
                except ValueError:
                    logger.warning(
                        "assign_in_place changed dtype %s of column %s to %s"
                        % (old_dtype, c, df[c].dtype)
                    )

    # add new columns (in order they appear in df2)
    new_columns = [c for c in df2.columns if c not in df.columns]

    df[new_columns] = df2[new_columns]

    for c in new_columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("category")

    auto_opt_pd_dtypes(df, downcast_int, downcast_float, inplace=True)


def auto_opt_pd_dtypes(
    df_: pd.DataFrame, downcast_int=False, downcast_float=False, inplace=False
) -> Optional[pd.DataFrame]:
    """
    Automatically downcast Number dtypes for minimal possible,
    will not touch other (datetime, str, object, etc)

    Parameters
    ----------
    df_ : pd.DataFrame
        assignment left-hand-side (dest)
    downcast_int: bool
        if True, downcast int columns if possible
    downcast_float: bool
        if True, downcast float columns if possible
    inplace: bool
        if False, will return a copy of input dataset

    Returns
    -------
        `None` if `inplace=True` or dataframe if `inplace=False`

    """
    df = df_ if inplace else df_.copy()

    for col in df.columns:
        dtype = df[col].dtype
        # Skip optimizing floats for precision concerns
        if pd.api.types.is_float_dtype(dtype):
            if not downcast_float:
                continue
            else:
                # there is a bug in pandas to_numeric
                # when convert int and floats gt 16777216
                # https://github.com/pandas-dev/pandas/issues/43693
                # https://github.com/pandas-dev/pandas/issues/23676#issuecomment-438488603
                if df[col].max() >= 16777216:
                    continue
                else:
                    df[col] = pd.to_numeric(df[col], downcast="float")
        # Skip if the column is already categorical
        if pd.api.types.is_categorical_dtype(dtype):
            continue
        # Handle integer types
        if pd.api.types.is_integer_dtype(dtype):
            if not downcast_int:
                continue
            # there is a bug in pandas to_numeric
            # when convert int and floats gt 16777216
            # https://github.com/pandas-dev/pandas/issues/43693
            # https://github.com/pandas-dev/pandas/issues/23676#issuecomment-438488603
            if df[col].max() >= 16777216:
                continue
            else:
                df[col] = pd.to_numeric(df[col], downcast="integer")
                continue
            # Initially thought of using unsigned integers, BUT:
            # There are calculations in asim (e.g., UECs) that expect results in negative values,
            # and operations on two unsigned types will not produce negative values,
            # therefore, we did not use unsigned integers.

    if not inplace:
        return df


def reindex_if_series(values, index):
    if index is not None:
        return values

    if isinstance(values, pd.Series):
        assert len(set(values.index).intersection(index)) == len(index)

        if all(values.index != index):
            return values.reindex(index=index)


def df_from_dict(values, index=None):
    # If value object is a series and has out of order index, reindex it
    values = {k: reindex_if_series(v, index) for k, v in values.items()}

    df = pd.DataFrame.from_dict(values)
    if index is not None:
        df.index = index

    # 2x slower but users less peak RAM
    # df = pd.DataFrame(index = index)
    # for c in values.keys():
    #     df[c] = values[c]
    #     del values[c]

    return df


# for disaggregate accessibilities


def ordered_load(
    stream, Loader=yaml.SafeLoader, object_pairs_hook=collections.OrderedDict
):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def named_product(**d):
    names = d.keys()
    vals = d.values()
    for res in itertools.product(*vals):
        yield dict(zip(names, res))


def recursive_replace(obj, search, replace):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = recursive_replace(v, search, replace)
    if isinstance(obj, list):
        obj = [replace if x == search else x for x in obj]
    if search == obj:
        obj = replace
    return obj


T = TypeVar("T")


def suffix_tables_in_settings(
    model_settings: T,
    suffix: str = "proto_",
    tables: Iterable[str] = ("persons", "households", "tours", "persons_merged"),
) -> T:
    if not isinstance(model_settings, dict):
        model_settings_type = type(model_settings)
        model_settings = model_settings.dict()
    else:
        model_settings_type = None

    for k in tables:
        model_settings = recursive_replace(model_settings, k, suffix + k)

    if model_settings_type is not None:
        model_settings = model_settings_type.model_validate(model_settings)

    return model_settings


def suffix_expressions_df_str(
    df, suffix="proto_", tables=["persons", "households", "tours", "persons_merged"]
):
    for k in tables:
        df["expression"] = df.expression.str.replace(k, suffix + k)
    return df


def parse_suffix_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file name")
    parser.add_argument("-s", "--SUFFIX", "-s", help="suffix to replace root targets")
    parser.add_argument(
        "-r", "--ROOTS", nargs="*", help="roots be suffixed", default=[]
    )
    return parser.parse_args(args.split())


def concat_suffix_dict(args):
    if isinstance(args, BaseModel):
        args = args.dict()
        if "source_file_paths" in args:
            del args["source_file_paths"]
    if isinstance(args, dict):
        args = sum([["--" + k, v] for k, v in args.items()], [])
    if isinstance(args, list):
        args = list(flatten(args))
    return args


def flatten(lst):
    for sublist in lst:
        if isinstance(sublist, list):
            for item in sublist:
                yield item
        else:
            yield sublist


def nearest_node_index(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)


def read_csv(filename):
    """Simple read of a CSV file, much faster than pandas.read_csv"""
    return csv.read_csv(filename).to_pandas()


def to_csv(df, filename, index=False):
    """Simple write of a CSV file, much faster than pandas.DataFrame.to_csv"""
    filename = Path(filename)
    if filename.suffix == ".gz":
        with pa.CompressedOutputStream(filename, "gzip") as out:
            csv.write_csv(pa.Table.from_pandas(df, preserve_index=index), out)
    else:
        csv.write_csv(pa.Table.from_pandas(df, preserve_index=index), filename)


def read_parquet(filename):
    """Simple read of a parquet file"""
    return pq.read_table(filename).to_pandas()


def to_parquet(df, filename, index=False):
    filename = Path(filename)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=index), filename)


def latest_file_modification_time(filenames: Iterable[Path]):
    """Find the most recent file modification time."""
    return max(os.path.getmtime(filename) for filename in filenames)


def oldest_file_modification_time(filenames: Iterable[Path]):
    """Find the least recent file modification time."""
    return min(os.path.getmtime(filename) for filename in filenames)


def zarr_file_modification_time(zarr_dir: Path):
    """Find the most recent file modification time inside a zarr dir."""
    t = 0
    for dirpath, dirnames, filenames in os.walk(zarr_dir):
        if os.path.basename(dirpath).startswith(".git"):
            continue
        for n in range(len(dirnames) - 1, -1, -1):
            if dirnames[n].startswith(".git"):
                dirnames.pop(n)
        for f in filenames:
            if f.startswith(".git") or f == ".DS_Store":
                continue
            finame = Path(os.path.join(dirpath, f))
            file_time = os.path.getmtime(finame)
            if file_time > t:
                t = file_time
    if t == 0:
        raise FileNotFoundError(zarr_dir)
    return t
