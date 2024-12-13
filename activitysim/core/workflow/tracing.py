from __future__ import annotations

import ast
import csv
import logging
import logging.config
import os
import struct
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from activitysim.core import tracing
from activitysim.core.test import assert_equal, assert_frame_substantively_equal
from activitysim.core.workflow.accessor import FromState, StateAccessor

logger = logging.getLogger(__name__)

CSV_FILE_TYPE = "csv"

DEFAULT_TRACEABLE_TABLES = [
    "households",
    "persons",
    "tours",
    "joint_tour_participants",
    "trips",
    "vehicles",
]


class RunId(str):
    def __new__(cls, x=None):
        if x is None:
            return cls(
                hex(struct.unpack("<Q", struct.pack("<d", time.time()))[0])[-6:].lower()
            )
        return super().__new__(cls, x)


class Tracing(StateAccessor):
    """
    Methods to provide the tracing capabilities of ActivitySim.
    """

    traceable_tables: list[str] = FromState(default_value=DEFAULT_TRACEABLE_TABLES)
    traceable_table_ids: dict[str, Sequence] = FromState(default_init=True)
    traceable_table_indexes: dict[str, str] = FromState(default_init=True)
    run_id: RunId = FromState(default_init=True)

    @property
    def validation_directory(self) -> Path | None:
        if self._obj is None:
            return None
        result = self._obj._context.get("tracing_validation_directory", None)
        if isinstance(result, tempfile.TemporaryDirectory):
            return Path(result.name)
        return result

    @validation_directory.setter
    def validation_directory(self, directory: Path | None):
        if directory is None:
            self._obj._context.pop("tracing_validation_directory", None)
        else:
            directory = Path(directory)
            # decompress cache file into working directory
            if directory.suffixes[-2:] == [".tar", ".gz"]:
                tempdir = tempfile.TemporaryDirectory()
                with tarfile.open(directory) as tfile:
                    tfile.extractall(tempdir.name)
                self._obj._context["tracing_validation_directory"] = tempdir
            else:
                self._obj._context["tracing_validation_directory"] = directory

    def __get__(self, instance, objtype=None) -> "Tracing":
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)

    def initialize(self):
        self.traceable_table_ids = {}

    def register_traceable_table(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Register traceable table

        Parameters
        ----------
        table_name : str
        df: pandas.DataFrame
            The traced dataframe.
        """

        # add index name to traceable_table_indexes

        logger.debug(f"register_traceable_table {table_name}")

        traceable_tables = self.traceable_tables
        if table_name not in traceable_tables:
            logger.error("table '%s' not in traceable_tables" % table_name)
            return

        idx_name = df.index.name
        if idx_name is None:
            logger.error("Can't register table '%s' without index name" % table_name)
            return

        traceable_table_ids = self.traceable_table_ids
        traceable_table_indexes = self.traceable_table_indexes

        if (
            idx_name in traceable_table_indexes
            and traceable_table_indexes[idx_name] != table_name
        ):
            logger.error(
                "table '%s' index name '%s' already registered for table '%s'"
                % (table_name, idx_name, traceable_table_indexes[idx_name])
            )
            return

        # update traceable_table_indexes with this traceable_table's idx_name
        if idx_name not in traceable_table_indexes:
            traceable_table_indexes[idx_name] = table_name
            logger.debug(
                "adding table %s.%s to traceable_table_indexes" % (table_name, idx_name)
            )
            self.traceable_table_indexes = traceable_table_indexes

        # add any new indexes associated with trace_hh_id to traceable_table_ids

        trace_hh_id = self._obj.settings.trace_hh_id
        if trace_hh_id is None:
            return

        new_traced_ids = []
        # if table_name == "households":
        if table_name in ["households", "proto_households"]:
            if trace_hh_id not in df.index:
                logger.warning("trace_hh_id %s not in dataframe" % trace_hh_id)
                new_traced_ids = []
            else:
                logger.info(
                    "tracing household id %s in %s households"
                    % (trace_hh_id, len(df.index))
                )
                new_traced_ids = [trace_hh_id]
        else:
            # find first already registered ref_col we can use to slice this table
            ref_col = next(
                (c for c in traceable_table_indexes if c in df.columns), None
            )

            if ref_col is None:
                logger.error(
                    "can't find a registered table to slice table '%s' index name '%s'"
                    " in traceable_table_indexes: %s"
                    % (table_name, idx_name, traceable_table_indexes)
                )
                return

            # get traceable_ids for ref_col table
            ref_col_table_name = traceable_table_indexes[ref_col]
            ref_col_traced_ids = traceable_table_ids.get(ref_col_table_name, [])

            # inject list of ids in table we are tracing
            # this allows us to slice by id without requiring presence of a household id column
            traced_df = df[df[ref_col].isin(ref_col_traced_ids)]
            new_traced_ids = traced_df.index.tolist()
            if len(new_traced_ids) == 0:
                logger.warning(
                    "register %s: no rows with %s in %s."
                    % (table_name, ref_col, ref_col_traced_ids)
                )

        # update the list of trace_ids for this table
        prior_traced_ids = traceable_table_ids.get(table_name, [])

        if new_traced_ids:
            assert not set(prior_traced_ids) & set(new_traced_ids)
            traceable_table_ids[table_name] = prior_traced_ids + new_traced_ids
            self.traceable_table_ids = traceable_table_ids

        logger.debug(
            "register %s: added %s new ids to %s existing trace ids"
            % (table_name, len(new_traced_ids), len(prior_traced_ids))
        )
        logger.debug(
            "register %s: tracing new ids %s in %s"
            % (table_name, new_traced_ids, table_name)
        )

    def deregister_traceable_table(self, table_name: str) -> None:
        """
        un-register traceable table

        Parameters
        ----------
        table_name : str
        """
        traceable_table_ids = self.traceable_table_ids
        traceable_table_indexes = self.traceable_table_indexes

        if table_name not in self.traceable_tables:
            logger.error("table '%s' not in traceable_tables" % table_name)

        else:
            self.traceable_table_ids = {
                k: v for k, v in traceable_table_ids.items() if k != table_name
            }
            self.traceable_table_indexes = {
                k: v for k, v in traceable_table_indexes.items() if v != table_name
            }

    def write_csv(
        self,
        df,
        file_name,
        index_label=None,
        columns=None,
        column_labels=None,
        transpose=True,
    ):
        """
        Print write_csv

        Parameters
        ----------
        df: pandas.DataFrame or pandas.Series or dict
            traced dataframe
        file_name: str
            output file name
        index_label: str
            index name
        columns: list
            columns to write
        transpose: bool
            whether to transpose dataframe (ignored for series)
        Returns
        -------
        Nothing
        """

        assert len(file_name) > 0

        if not file_name.endswith(".%s" % CSV_FILE_TYPE):
            file_name = "%s.%s" % (file_name, CSV_FILE_TYPE)

        file_path = self._obj.filesystem.get_trace_file_path(
            file_name, tail=self.run_id
        )

        if os.name == "nt":
            abs_path = os.path.abspath(file_path)
            if len(abs_path) > 255:
                msg = f"path length ({len(abs_path)}) may exceed Windows maximum length unless LongPathsEnabled: {abs_path}"
                logger.warning(msg)

        if os.path.isfile(file_path):
            logger.debug("write_csv file exists %s %s" % (type(df).__name__, file_name))

        if isinstance(df, pd.DataFrame):
            # logger.debug("dumping %s dataframe to %s" % (df.shape, file_name))
            tracing.write_df_csv(
                df, file_path, index_label, columns, column_labels, transpose=transpose
            )
        elif isinstance(df, pd.Series):
            # logger.debug("dumping %s element series to %s" % (df.shape[0], file_name))
            tracing.write_series_csv(df, file_path, index_label, columns, column_labels)
        elif isinstance(df, dict):
            df = pd.Series(data=df)
            # logger.debug("dumping %s element dict to %s" % (df.shape[0], file_name))
            tracing.write_series_csv(df, file_path, index_label, columns, column_labels)
        else:
            logger.error(
                "write_csv object for file_name '%s' of unexpected type: %s"
                % (file_name, type(df))
            )

    def trace_df(
        self,
        df: pd.DataFrame,
        label: str,
        slicer=None,
        columns: Optional[list[str]] = None,
        index_label=None,
        column_labels=None,
        transpose=True,
        warn_if_empty=False,
    ):
        """
        Slice dataframe by traced household or person id dataframe and write to CSV

        Parameters
        ----------
        state: workflow.State
        df: pandas.DataFrame
            traced dataframe
        label: str
            tracer name
        slicer: Object
            slicer for subsetting
        columns: list
            columns to write
        index_label: str
            index name
        column_labels: [str, str]
            labels for columns in csv
        transpose: boolean
            whether to transpose file for legibility
        warn_if_empty: boolean
            write warning if sliced df is empty

        Returns
        -------
        Nothing
        """

        target_ids, column = self.get_trace_target(df, slicer)

        if target_ids is not None:
            df = tracing.slice_ids(df, target_ids, column)

        if warn_if_empty and df.shape[0] == 0 and target_ids != []:
            column_name = column or slicer
            logger.warning(
                "slice_canonically: no rows in %s with %s == %s"
                % (label, column_name, target_ids)
            )

        if df.shape[0] > 0:
            self.write_csv(
                df,
                file_name=label,
                index_label=(index_label or slicer),
                columns=columns,
                column_labels=column_labels,
                transpose=transpose,
            )

        if self.validation_directory:
            skip_validation = False
            if label.endswith("constants"):
                skip_validation = (
                    True  # contants sometimes has skimwrapper objects added
                )
            if not skip_validation:
                try:
                    that_path = self._obj.filesystem.find_trace_file_path(
                        label, trace_dir=self.validation_directory, file_type="csv"
                    )
                except FileNotFoundError as err:
                    logger.warning(
                        f"trace validation file not found: {err}\n"
                        f" in validation_directory: {self.validation_directory}"
                    )
                else:
                    if transpose:
                        # wreaks havoc with pandas dtypes and column names
                        # check as a simple list of lists instead
                        def literal_eval(x):
                            try:
                                return ast.literal_eval(x)
                            except Exception:
                                return x

                        def read_csv_as_list_of_lists(finame):
                            with open(finame, newline="") as csvfile:
                                return [
                                    list(map(literal_eval, row))
                                    for row in csv.reader(csvfile)
                                ]

                        that_blob = read_csv_as_list_of_lists(that_path)
                        this_path = self._obj.filesystem.get_trace_file_path(
                            label, tail=self.run_id, file_type="csv"
                        )
                        this_blob = read_csv_as_list_of_lists(this_path)

                        _this_index = [i[0] for i in this_blob]
                        if len(set(_this_index)) == len(_this_index):
                            # indexes are unique, convert to dict
                            this_dict = dict(
                                zip(
                                    [i[0] for i in this_blob],
                                    [i[1:] for i in this_blob],
                                )
                            )
                            that_dict = dict(
                                zip(
                                    [i[0] for i in that_blob],
                                    [i[1:] for i in that_blob],
                                )
                            )
                            assert_equal(this_dict, that_dict)
                        else:
                            try:
                                assert_equal(this_blob, that_blob)
                            except:
                                logger.error(f"trace validation BAD: {label}")
                                raise
                            else:
                                logger.debug(f"trace validation OK: {label}")
                    else:
                        that_df = pd.read_csv(that_path)
                        # check against the file we just wrote
                        this_path = self._obj.filesystem.get_trace_file_path(
                            label, tail=self.run_id, file_type="csv"
                        )
                        this_df = pd.read_csv(this_path)
                        assert_frame_substantively_equal(this_df, that_df)
                        logger.debug(f"trace validation OK: {label}")

    def trace_interaction_eval_results(self, trace_results, trace_ids, label):
        """
        Trace model design eval results for interaction_simulate

        Parameters
        ----------
        trace_results: pandas.DataFrame
            traced model_design dataframe
        trace_ids : tuple (str,  numpy.ndarray)
            column name and array of trace_ids from interaction_trace_rows()
            used to filter the trace_results dataframe by traced hh or person id
        label: str
            tracer name

        Returns
        -------
        Nothing
        """

        assert type(trace_ids[1]) == np.ndarray

        slicer_column_name = trace_ids[0]

        try:
            trace_results[slicer_column_name] = trace_ids[1]
        except ValueError:
            trace_results[slicer_column_name] = int(trace_ids[1])

        targets = np.unique(trace_ids[1])

        if len(trace_results.index) == 0:
            return

        # write out the raw dataframe

        file_path = self._obj.filesystem.get_trace_file_path(
            "%s.raw.csv" % label, tail=self.run_id
        )
        trace_results.to_csv(file_path, mode="a", index=True, header=True)

        # if there are multiple targets, we want them in separate tables for readability
        for target in targets:
            df_target = trace_results[trace_results[slicer_column_name] == target]

            # we want the transposed columns in predictable order
            df_target.sort_index(inplace=True)

            # # remove the slicer (person_id or hh_id) column?
            # del df_target[slicer_column_name]

            target_label = "%s.%s.%s" % (label, slicer_column_name, target)

            self.trace_df(
                df_target,
                label=target_label,
                slicer="NONE",
                transpose=True,
                column_labels=["expression", None],
                warn_if_empty=False,
            )

    def interaction_trace_rows(self, interaction_df, choosers, sample_size=None):
        """
        Trace model design for interaction_simulate

        Parameters
        ----------
        interaction_df: pandas.DataFrame
            traced model_design dataframe
        choosers: pandas.DataFrame
            interaction_simulate choosers
            (needed to filter the model_design dataframe by traced hh or person id)
        sample_size int or None
            int for constant sample size, or None if choosers have different numbers of alternatives
        Returns
        -------
        trace_rows : numpy.ndarray
            array of booleans to flag which rows in interaction_df to trace

        trace_ids : tuple (str,  numpy.ndarray)
            column name and array of trace_ids mapping trace_rows to their target_id
            for use by trace_interaction_eval_results which needs to know target_id
            so it can create separate tables for each distinct target for readability
        """

        # slicer column name and id targets to use for chooser id added to model_design dataframe
        # currently we only ever slice by person_id, but that could change, so we check here...

        traceable_table_ids = self.traceable_table_ids

        # trace proto tables if they exist, otherwise trace actual tables
        # proto tables are used for disaggregate accessibilities and
        # are removed from the traceable_table_ids after the accessibilities are created
        households_table_name = (
            "proto_households"
            if "proto_households" in traceable_table_ids.keys()
            else "households"
        )

        persons_table_name = (
            "proto_persons"
            if "proto_persons" in traceable_table_ids.keys()
            else "persons"
        )

        if (
            choosers.index.name in ["person_id", "proto_person_id"]
        ) and persons_table_name in traceable_table_ids:
            slicer_column_name = choosers.index.name
            targets = traceable_table_ids[persons_table_name]
        elif (
            choosers.index.name in ["household_id", "proto_household_id"]
        ) and households_table_name in traceable_table_ids:
            slicer_column_name = choosers.index.name
            targets = traceable_table_ids[households_table_name]
        elif "household_id" in choosers.columns and "households" in traceable_table_ids:
            slicer_column_name = "household_id"
            targets = traceable_table_ids[households_table_name]
        elif (
            "person_id" in choosers.columns
            and persons_table_name in traceable_table_ids
        ):
            slicer_column_name = "person_id"
            targets = traceable_table_ids[persons_table_name]
        elif (
            choosers.index.name == "proto_tour_id"
            and "proto_tours" in traceable_table_ids
        ):
            slicer_column_name = choosers.index.name
            targets = traceable_table_ids["proto_tours"]
        else:
            print(choosers.columns)
            raise RuntimeError(
                "interaction_trace_rows don't know how to slice index '%s'"
                % choosers.index.name
            )

        if sample_size is None:
            # if sample size not constant, we count on either
            # slicer column being in itneraction_df
            # or index of interaction_df being same as choosers
            if slicer_column_name in interaction_df.columns:
                trace_rows = np.in1d(interaction_df[slicer_column_name], targets)
                trace_ids = interaction_df.loc[trace_rows, slicer_column_name].values
            else:
                assert interaction_df.index.name == choosers.index.name
                trace_rows = np.in1d(interaction_df.index, targets)
                trace_ids = interaction_df[trace_rows].index.values

        else:
            if slicer_column_name == choosers.index.name:
                trace_rows = np.in1d(choosers.index, targets)
                trace_ids = np.asanyarray(choosers[trace_rows].index)
            elif slicer_column_name == "person_id":
                trace_rows = np.in1d(choosers["person_id"], targets)
                trace_ids = np.asanyarray(choosers[trace_rows].person_id)
            elif slicer_column_name == "household_id":
                trace_rows = np.in1d(choosers["household_id"], targets)
                trace_ids = np.asanyarray(choosers[trace_rows].household_id)
            else:
                assert False

            # simply repeat if sample size is constant across choosers
            assert sample_size == len(interaction_df.index) / len(choosers.index)
            trace_rows = np.repeat(trace_rows, sample_size)
            trace_ids = np.repeat(trace_ids, sample_size)

        assert type(trace_rows) == np.ndarray
        assert type(trace_ids) == np.ndarray

        trace_ids = (slicer_column_name, trace_ids)

        return trace_rows, trace_ids

    def get_trace_target(self, df: pd.DataFrame, slicer: str, column: Any = None):
        """
        get target ids and column or index to identify target trace rows in df

        Parameters
        ----------
        df: pandas.DataFrame
            This dataframe is to be sliced
        slicer: str
            name of column or index to use for slicing
        column : Any

        Returns
        -------
        target : int or list of ints
            id or ids that identify tracer target rows
        column : str
            name of column to search for targets or None to search index
        """

        target_ids = (
            None  # id or ids to slice by (e.g. hh_id or person_ids or tour_ids)
        )

        # special do-not-slice code for dumping entire df
        if slicer == "NONE":
            return target_ids, column

        if slicer is None:
            slicer = df.index.name

        if isinstance(df, pd.DataFrame):
            # always slice by household id if we can
            if "household_id" in df.columns:
                slicer = "household_id"
            if slicer in df.columns:
                column = slicer

        if column is None and df.index.name != slicer:
            raise RuntimeError(
                "bad slicer '%s' for df with index '%s'" % (slicer, df.index.name)
            )

        traceable_table_indexes = self.traceable_table_indexes
        traceable_table_ids = self.traceable_table_ids

        if df.empty:
            target_ids = None
        elif slicer in traceable_table_indexes:
            # maps 'person_id' to 'persons', etc
            table_name = traceable_table_indexes[slicer]
            target_ids = traceable_table_ids.get(table_name, [])
        elif slicer == "zone_id":
            target_ids = self._obj.settings.trace_od

        return target_ids, column

    def trace_targets(self, df, slicer=None, column=None):
        target_ids, column = self.get_trace_target(df, slicer, column)

        if target_ids is None:
            targets = None
        else:
            if column is None:
                targets = df.index.isin(target_ids)
            else:
                # convert to numpy array for consistency since that is what index.isin returns
                targets = df[column].isin(target_ids).to_numpy()

        return targets

    def has_trace_targets(self, df, slicer=None, column=None):
        target_ids, column = self.get_trace_target(df, slicer, column)

        if target_ids is None:
            found = False
        else:
            if column is None:
                found = df.index.isin(target_ids).any()
            else:
                found = df[column].isin(target_ids).any()

        return found

    def dump_df(self, dump_switch, df, trace_label, fname):
        if dump_switch:
            trace_label = tracing.extend_trace_label(trace_label, "DUMP.%s" % fname)
            self.trace_df(
                df,
                trace_label,
                index_label=df.index.name,
                slicer="NONE",
                transpose=False,
            )

    def delete_output_files(self, file_type, ignore=None, subdir=None):
        """
        Delete files in output directory of specified type.

        Parameters
        ----------
        file_type : str
            File extension to delete.
        ignore : list[Path-like]
            Specific files to leave alone.
        subdir : list[Path-like], optional
            Subdirectories to scrub.  If not given, the top level output directory
            plus the 'log' and 'trace' directories will be scrubbed.
        """

        output_dir = self._obj.filesystem.get_output_dir()

        subdir = [subdir] if subdir else None
        directories = subdir or ["", "log", "trace"]

        for subdir in directories:
            dir = output_dir.joinpath(output_dir, subdir) if subdir else output_dir

            if not dir.exists():
                continue

            if ignore:
                ignore = [os.path.realpath(p) for p in ignore]

            # logger.debug("Deleting %s files in output dir %s" % (file_type, dir))

            for the_file in os.listdir(dir):
                if the_file.endswith(file_type):
                    file_path = os.path.join(dir, the_file)

                    if ignore and os.path.realpath(file_path) in ignore:
                        continue

                    try:
                        if os.path.isfile(file_path):
                            logger.debug("delete_output_files deleting %s" % file_path)
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)

    def delete_trace_files(self):
        """
        Delete CSV files in output_dir
        """
        self.delete_output_files(CSV_FILE_TYPE, subdir="trace")
        self.delete_output_files(CSV_FILE_TYPE, subdir="log")

        active_log_files = [
            h.baseFilename
            for h in logger.root.handlers
            if isinstance(h, logging.FileHandler)
        ]

        self.delete_output_files("log", ignore=active_log_files)
