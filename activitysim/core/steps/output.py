# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import sys
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as parquet

from activitysim.core import configuration, workflow
from activitysim.core.workflow.checkpoint import CHECKPOINT_NAME
from activitysim.core.estimation import estimation_enabled, EstimationConfig

logger = logging.getLogger(__name__)


@workflow.step
def track_skim_usage(state: workflow.State) -> None:
    """
    write statistics on skim usage (diagnostic to detect loading of un-needed skims)

    FIXME - have not yet implemented a facility to avoid loading of unused skims

    FIXME - if resume_after, this will only reflect skims used after resume

    Parameters
    ----------
    output_dir: str

    """
    pd.options.display.max_columns = 500
    pd.options.display.max_rows = 100

    skim_dict = state.get_injectable("skim_dict")

    mode = "wb" if sys.version_info < (3,) else "w"
    with open(
        state.filesystem.get_output_file_path("skim_usage.txt"), mode
    ) as output_file:
        print("\n### skim_dict usage", file=output_file)
        for key in skim_dict.get_skim_usage():
            print(key, file=output_file)

        try:
            unused = set(k for k in skim_dict.skim_info.base_keys) - set(
                k for k in skim_dict.get_skim_usage()
            )
        except AttributeError:
            base_keys = set(skim_dict.dataset.variables.keys()) - set(
                skim_dict.dataset.coords.keys()
            )
            # using dataset
            unused = base_keys - set(k for k in skim_dict.get_skim_usage())

        for key in unused:
            print(key, file=output_file)


def previous_write_data_dictionary(state: workflow.State, output_dir):
    """
    Write table_name, number of rows, columns, and bytes for each checkpointed table

    Parameters
    ----------
    output_dir: str

    """

    model_settings = state.filesystem.read_model_settings("write_data_dictionary")
    txt_format = model_settings.get("txt_format", "data_dict.txt")
    csv_format = model_settings.get("csv_format", "data_dict.csv")

    if txt_format:
        output_file_path = state.get_output_file_path(txt_format)

        pd.options.display.max_columns = 500
        pd.options.display.max_rows = 100

        output_tables = state.checkpoint.list_tables()

        # write data dictionary for all checkpointed_tables

        with open(output_file_path, "w") as output_file:
            for table_name in output_tables:
                df = state.get_dataframe(table_name)

                print("\n### %s %s" % (table_name, df.shape), file=output_file)
                print("index:", df.index.name, df.index.dtype, file=output_file)
                print(df.dtypes, file=output_file)


@workflow.step
def write_data_dictionary(state: workflow.State) -> None:
    """
    Write table schema for all tables

    model settings
        txt_format: output text file name (default data_dict.txt) or empty to suppress txt output
        csv_format: output csv file name (default data_dict.tcsvxt) or empty to suppress txt output

        schema_tables: list of tables to include in output (defaults to all checkpointed tables)

    for each table, write column names, dtype, and checkpoint added)

    text format writes individual table schemas to a single text file
    csv format writes all tables together with an additional table_name column

    Parameters
    ----------
    output_dir: str

    """

    model_settings = state.filesystem.read_model_settings("write_data_dictionary")
    txt_format = model_settings.get("txt_format", "data_dict.txt")
    csv_format = model_settings.get("csv_format", "data_dict.csv")

    if not (csv_format or txt_format):
        logger.warning(
            f"write_data_dictionary step invoked but neither 'txt_format' nor 'csv_format' specified"
        )
        return

    table_names = state.registered_tables()

    # use table_names list from model_settings, if provided
    schema_tables = model_settings.get("tables", None)
    if schema_tables:
        table_names = [c for c in schema_tables if c in table_names]

    # initialize schema as dict of dataframe[table_name, column_name, dtype, checkpoint]
    schema = dict()
    final_shapes = dict()
    for table_name in table_names:
        try:
            df = state.get_dataframe(table_name)
        except RuntimeError as run_err:
            if run_err.args and "dropped" in run_err.args[0]:
                # if a checkpointed table was dropped, that's not ideal, so we should
                # log a warning about it, but not allow the error to stop execution here
                logger.warning(run_err.args[0])
                # note actually emitting a warnings.warn instead of a logger message will
                # unfortunately cause some of our excessively strict tests to fail
                continue
            else:
                raise

        final_shapes[table_name] = df.shape

        if df.index.name and df.index.name not in df.columns:
            df = df.reset_index()
        info = (
            df.dtypes.astype(str)
            .to_frame("dtype")
            .reset_index()
            .rename(columns={"index": "column_name"})
        )
        info["checkpoint"] = ""

        info.insert(loc=0, column="table_name", value=table_name)
        schema[table_name] = info

    # annotate schema.info with name of checkpoint columns were first seen
    if state.checkpoint.store:
        for _, row in state.checkpoint.get_inventory().iterrows():
            checkpoint_name = row[CHECKPOINT_NAME]

            for table_name in table_names:
                # no change to table in this checkpoint
                if row.get(table_name, None) != checkpoint_name:
                    continue

                # get the checkpointed version of the table
                df = state.checkpoint.load_dataframe(table_name, checkpoint_name)

                if df.index.name and df.index.name not in df.columns:
                    df = df.reset_index()

                info = schema.get(table_name, None)

                if info is not None:
                    # tag any new columns with checkpoint name
                    prev_columns = info[info.checkpoint != ""].column_name.values
                    new_cols = [c for c in df.columns.values if c not in prev_columns]
                    is_new_column_this_checkpoont = info.column_name.isin(new_cols)
                    info.checkpoint = np.where(
                        is_new_column_this_checkpoont, checkpoint_name, info.checkpoint
                    )
                    schema[table_name] = info

    schema_df = pd.concat(schema.values())

    if csv_format:
        schema_df.to_csv(
            state.get_output_file_path(csv_format), header=True, index=False
        )

    if txt_format:
        with open(state.get_output_file_path(txt_format), "w") as output_file:
            # get max schema column widths from omnibus table
            col_width = {c: schema_df[c].str.len().max() + 2 for c in schema_df}

            for table_name in table_names:
                info = schema.get(table_name, None)
                if info is None:
                    continue
                columns_to_print = ["column_name", "dtype", "checkpoint"]
                info = info[columns_to_print].copy()

                # normalize schema columns widths across all table schemas for unified output formatting
                for c in info:
                    info[c] = info[c].str.pad(col_width[c], side="right")
                info.columns = [c.ljust(col_width[c]) for c in info.columns]

                info = info.to_string(index=False)

                print(
                    f"###\n### {table_name} {final_shapes[table_name]}\n###\n",
                    file=output_file,
                )
                print(f"{info}\n", file=output_file)


def find_lowest_level_directories(starting_directory):
    lowest_dirs = list()

    for root, dirs, files in os.walk(starting_directory):
        if not dirs:
            lowest_dirs.append(root)

    return lowest_dirs


def concat_and_write_edb(df_concat_dict, write_dir):
    # concatenate the dataframes and output final file
    for table_name, df_array in df_concat_dict.items():
        df = pd.concat(df_array)

        # sort the dataframe by index
        if df.index.name is not None:
            df = df.sort_index()
        else:
            df = df.sort_values(by=df.columns[0], kind="mergesort", ignore_index=True)

        if table_name.endswith(".csv"):
            df.to_csv(os.path.join(write_dir, table_name), index=False)
        elif table_name.endswith(".parquet"):
            df.to_parquet(os.path.join(write_dir, table_name), index=True)
        elif table_name.endswith(".pkl"):
            df.to_pickle(os.path.join(write_dir, table_name))
        else:
            raise ValueError(f"Unknown file type {table_name}")


def _coalesce_estimation_data_bundles(state):
    """
    In estimation mode, estimation data bundles are written to separate subdirectories for each subprocess.
    This model will go through each subdirectory and move the files to the parent directory.
    This will only occur if the lowest level directory contains the multiprocess step names.
    Only multiprocess step names are used because that's how EDBs are written in estimation mode.
    """

    logger.info("Coalescing Estimation Data Bundles")

    edb_dir = state.filesystem.get_output_dir("estimation_data_bundle")
    estimation_settings = EstimationConfig.read_settings_file(
        state.filesystem, "estimation.yaml", mandatory=True
    )

    lowest_dirs = find_lowest_level_directories(edb_dir)

    multiprocessing_step_names = [
        step.name for step in state.settings.multiprocess_steps
    ]
    lowest_dirs = [
        dir
        for dir in lowest_dirs
        if any(step in dir for step in multiprocessing_step_names)
    ]

    if len(lowest_dirs) == 0:
        logger.info("No estimation data bundles to coalesce")
        return

    prev_edb = None
    df_concat_dict = {}

    # loop through each lowest level directory
    for dir in lowest_dirs:
        logger.debug(f"Coalescing {dir}")
        # get the parent directory
        cur_edb = Path(dir).parent.absolute()
        if prev_edb is None:
            prev_edb = cur_edb

        # check if we have moved onto a new EDB
        is_same_edb = cur_edb == prev_edb

        # if we have moved onto a new EDB, concatenate the dataframes and write the final files
        if (
            (not is_same_edb)
            and (len(df_concat_dict) > 0)
            # and (len(df_concat_dict[list(df_concat_dict.keys())[0]]) > 1)
        ):
            concat_and_write_edb(df_concat_dict, prev_edb)

            # reset edb dir and dictionary
            prev_edb = cur_edb
            df_concat_dict = {}

        for i, file in enumerate(os.listdir(dir)):

            # get the file path
            file_path = os.path.join(dir, file)

            # look for files that are duplicated across subprocesses
            is_coefs_file = file.endswith(".csv") and "coef" in file
            is_settings_file = file.endswith(".yaml")
            is_spec_file = file.endswith(".csv") and ("spec" in file.lower())
            is_landuse_file = file.endswith("_landuse.csv")
            is_size_terms_file = file.endswith("_size_terms.csv")
            is_duplicate_file = (
                is_coefs_file
                or is_spec_file
                or is_settings_file
                or is_landuse_file
                or is_size_terms_file
            )

            if is_duplicate_file and not os.path.exists(os.path.join(cur_edb, file)):
                # copy the file to the parent directory
                shutil.copy(file_path, os.path.join(cur_edb, file))

            if not is_duplicate_file:
                # read file and store in dictionary
                if file.endswith(".csv"):
                    df = pd.read_csv(file_path, low_memory=False)
                elif file.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                elif file.endswith(".pkl"):
                    df = pd.read_pickle(file_path)
                else:
                    raise ValueError(
                        f"Unknown file type found {file}, expect csv, parquet, or pkl"
                    )

                if file in df_concat_dict.keys():
                    df_concat_dict[file].append(df)
                else:
                    df_concat_dict[file] = [df]

        # delete the directory now that we have gone through all the files
        if estimation_settings.DELETE_MP_SUBDIRS:
            shutil.rmtree(dir)

    # need to concatenate the last set of dataframes
    concat_and_write_edb(df_concat_dict, cur_edb)

    return


@workflow.step
def write_tables(state: workflow.State) -> None:
    """
    Write pipeline tables as csv or parquet files (in output directory) as specified
    by output_tables list in settings file. Output to parquet or a single h5 file is
    also supported.

    'h5_store' defaults to False, which means the output will be written out to csv.
    'file_type' defaults to 'csv' but can also be used to specify 'parquet' or 'h5'.
    When 'h5_store' is set to True, 'file_type' is ingored and the outputs are written to h5.

    'output_tables' can specify either a list of output tables to include or to skip
    if no output_tables list is specified, then all checkpointed tables will be written

    To write all output tables EXCEPT the households and persons tables:

    ::

      output_tables:
        action: skip
        tables:
          - households
          - persons

    To write ONLY the households table:

    ::

      output_tables:
        action: include
        tables:
           - households

    To write tables into a single HDF5 store instead of individual CSVs, use the h5_store flag:

    ::

      output_tables:
        h5_store: True
        action: include
        tables:
           - households

    To write tables to parquet files, use the file_type setting:

    ::

      output_tables:
        file_type: parquet
        action: include
        tables:
           - households

    Parameters
    ----------
    output_dir: str

    """

    output_tables_settings = state.settings.output_tables

    if output_tables_settings is None:
        logger.info("No output_tables specified in settings file. Nothing to write.")
        return

    action = output_tables_settings.action
    tables = output_tables_settings.tables
    prefix = output_tables_settings.prefix
    h5_store = output_tables_settings.h5_store
    file_type = output_tables_settings.file_type
    sort = output_tables_settings.sort

    registered_tables = state.registered_tables()
    if action == "include":
        # interpret empty or missing tables setting to mean include all registered tables
        output_tables_list = tables if tables is not None else registered_tables
    elif action == "skip":
        output_tables_list = [t for t in registered_tables if t not in tables]
    else:
        raise f"expected action '{action}' to be either 'include' or 'skip'"

    for table_name in output_tables_list:
        if isinstance(table_name, configuration.OutputTable):
            table_decode_cols = table_name.decode_columns or {}
            table_name = table_name.tablename
        elif not isinstance(table_name, str):
            table_decode_cols = table_name.get("decode_columns", {})
            table_name = table_name["tablename"]
        else:
            table_decode_cols = {}

        if table_name == "checkpoints":
            dt = pa.Table.from_pandas(
                state.checkpoint.get_inventory(), preserve_index=True
            )
        else:
            if table_name not in registered_tables:
                logger.warning("Skipping '%s': Table not found." % table_name)
                continue

            # the write tables method now uses pyarrow to avoid making edits to
            # the internal pipeline dataframes, which need to remain un-decoded
            # for any subsequent summarize step[s].
            dt = state.get_pyarrow(table_name)
            dt_index_name = state.get_dataframe_index_name(table_name)

            if sort:
                traceable_table_indexes = state.tracing.traceable_table_indexes

                if dt_index_name in traceable_table_indexes:
                    dt = dt.sort_by(dt_index_name)
                    logger.debug(
                        f"write_tables sorting {table_name} on index {dt_index_name}"
                    )
                else:
                    # find all registered columns we can use to sort this table
                    # (they are ordered appropriately in traceable_table_indexes)
                    sort_columns = [
                        (c, "ascending")
                        for c in traceable_table_indexes
                        if c in dt.columns
                    ]
                    if len(sort_columns) > 0:
                        dt = dt.sort_by(sort_columns)
                        logger.debug(
                            f"write_tables sorting {table_name} on columns {sort_columns}"
                        )
                    elif dt_index_name is not None:
                        logger.debug(
                            f"write_tables sorting {table_name} on unrecognized index {dt_index_name}"
                        )
                        dt = dt.sort_by(dt_index_name)
                    else:
                        logger.debug(
                            f"write_tables sorting {table_name} on unrecognized index {dt_index_name}"
                        )
                        dt = dt.sort_by(dt_index_name)

        if state.settings.recode_pipeline_columns:
            for colname, decode_instruction in table_decode_cols.items():
                if "|" in decode_instruction:
                    decode_filter, decode_instruction = decode_instruction.split("|")
                    decode_filter = decode_filter.strip()
                    decode_instruction = decode_instruction.strip()
                else:
                    decode_filter = None

                if decode_instruction == "time_period":
                    map_col = list(state.network_settings.skim_time_periods.labels)
                    map_func = map_col.__getitem__
                    revised_col = (
                        pd.Series(dt.column(colname)).astype(int).map(map_func)
                    )
                    dt = dt.drop([colname]).append_column(
                        colname, pa.array(revised_col)
                    )
                    continue

                if "." not in decode_instruction:
                    lookup_col = decode_instruction
                    source_table = table_name
                    parent_table = dt
                else:
                    source_table, lookup_col = decode_instruction.split(".")
                    parent_table = state.get_pyarrow(source_table)
                try:
                    map_col = parent_table.column(f"_original_{lookup_col}")
                except KeyError:
                    map_col = parent_table.column(lookup_col)
                map_col = np.asarray(map_col)
                map_func = map_col.__getitem__
                if decode_filter:
                    if decode_filter == "nonnegative":

                        def map_func(x):
                            return x if x < 0 else map_col[x]

                    else:
                        raise ValueError(f"unknown decode_filter {decode_filter}")
                if colname in dt.column_names:
                    revised_col = (
                        pd.Series(dt.column(colname)).astype(int).map(map_func)
                    )
                    dt = dt.drop([colname]).append_column(
                        colname, pa.array(revised_col)
                    )
                # drop _original_x from table if it is duplicative
                if (
                    source_table == table_name
                    and f"_original_{lookup_col}" in dt.column_names
                ):
                    dt = dt.drop([f"_original_{lookup_col}"])

        if h5_store or file_type == "h5":
            file_path = state.get_output_file_path("%soutput_tables.h5" % prefix)
            dt.to_pandas().to_hdf(
                str(file_path), key=table_name, mode="a", format="fixed"
            )

        else:
            file_name = f"{prefix}{table_name}.{file_type}"
            file_path = state.get_output_file_path(file_name)

            # include the index if it has a name or is a MultiIndex
            if file_type == "csv":
                csv.write_csv(dt, file_path)
            elif file_type == "parquet":
                parquet.write_table(dt, file_path)
            else:
                raise ValueError(f"unknown file_type {file_type}")


@workflow.step
def coalesce_estimation_data_bundles(state: workflow.State) -> None:
    """
    In estimation mode, estimation data bundles are written to separate subdirectories for each subprocess.
    This model will go through each subdirectory and concat / copy the files to the parent directory.
    This will only occur if the lowest level directory contains the multiprocess step names.
    Only multiprocess step names are used because that's how EDBs are written in estimation mode.

    """
    is_estimation = estimation_enabled(state)
    if state.settings.multiprocess and is_estimation:
        _coalesce_estimation_data_bundles(state)
    else:
        logger.info(
            "Not in estimation mode or not using multiprocess. Nothing to coalesce."
        )
    return
