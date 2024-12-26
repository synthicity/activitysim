from __future__ import annotations

import glob
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from larch import Dataset, Model
from larch.util import Dict

from .general import (
    apply_coefficients,
    cv_to_ca,
    linear_utility_from_spec,
    remove_apostrophes,
)

_logger = logging.getLogger("larch")


def interaction_simulate_data(
    name="non_mandatory_tour_frequency",
    edb_directory="output/estimation_data_bundle/{name}/",
    settings_file="{name}_model_settings.yaml",
    spec_file="{name}_SPEC.csv",
    alt_def_file="{name}_alternatives.csv",
    coefficients_files="{segment_name}/{name}_coefficients_{segment_name}.csv",
    chooser_data_files="{segment_name}/{name}_choosers_combined.csv",
    alt_values_files="{segment_name}/{name}_alternatives_combined.csv",
    segment_subset=(),
):
    edb_directory = edb_directory.format(name=name)

    def _read_csv(filename, **kwargs):
        filename = Path(edb_directory).joinpath(filename.format(name=name))
        if filename.with_suffix(".parquet").exists():
            print("loading from", filename.with_suffix(".parquet"))
            if "comment" in kwargs:
                kwargs.pop("comment")
            return pd.read_parquet(filename.with_suffix(".parquet"), **kwargs)
        if filename.exists():
            print("loading from", filename)
            return pd.read_csv(filename, **kwargs)
        # if the file does not exist, try to load parquet from a subdirectory
        search_glob = os.path.join(
            edb_directory, "*", filename.with_suffix(".parquet").name
        )
        files = glob.glob(search_glob)
        if files:
            print("loading from", files[0])
            return pd.read_parquet(files[0], **kwargs)
        # otherwise try to load csv from a subdirectory
        search_glob = os.path.join(edb_directory, "*", filename.name)
        files = glob.glob(search_glob)
        if files:
            print("loading from", files[0])
            return pd.read_csv(files[0], **kwargs)
        raise FileNotFoundError(f"File {filename} not found")

    settings_file = settings_file.format(name=name)
    settings_file_resolved = os.path.join(edb_directory, settings_file)
    if not os.path.exists(settings_file_resolved):
        search_glob = os.path.join(edb_directory, "*", settings_file)
        settings_files = glob.glob(search_glob)
        if not settings_files:
            raise FileNotFoundError(f"Settings file {settings_file} not found")
        else:
            settings_file_resolved = settings_files[0]

    with open(settings_file_resolved) as yf:
        settings = yaml.load(
            yf,
            Loader=yaml.SafeLoader,
        )

    coefficients = {}
    chooser_data = {}
    alt_values = {}

    segment_names = [s["NAME"] for s in settings["SPEC_SEGMENTS"]]
    if len(segment_subset) > 0:
        assert set(segment_subset).issubset(
            set(segment_names)
        ), f"{segment_subset} is not a subset of {segment_names}"
        segment_names = segment_subset

    for segment_name in segment_names:
        print(f"Loading EDB for {segment_name} segment")
        coefficients[segment_name] = _read_csv(
            coefficients_files.format(name=name, segment_name=segment_name),
            index_col="coefficient_name",
            comment="#",
        )
        chooser_data[segment_name] = _read_csv(
            chooser_data_files.format(name=name, segment_name=segment_name),
        )
        alt_values[segment_name] = _read_csv(
            alt_values_files.format(name=name, segment_name=segment_name),
            comment="#",
        )

    spec = _read_csv(
        spec_file,
        comment="#",
    )
    spec = remove_apostrophes(spec, ["Label"])
    # alt_names = list(spec.columns[3:])
    # alt_codes = np.arange(1, len(alt_names) + 1)
    # alt_names_to_codes = dict(zip(alt_names, alt_codes))
    # alt_codes_to_names = dict(zip(alt_codes, alt_names))

    alt_def = _read_csv(alt_def_file.format(name=name), index_col=0)

    return Dict(
        edb_directory=Path(edb_directory),
        settings=settings,
        chooser_data=chooser_data,
        coefficients=coefficients,
        alt_values=alt_values,
        spec=spec,
        alt_def=alt_def,
    )


def link_same_value_coefficients(segment_names, coefficients, spec):
    # Assume all coefficients with exactly equal current values are
    # actually the same estimated coefficient value and should be
    # treated as such by Larch.  Comment out this function where called to relax
    # this assumption, although be careful about the number of unique
    # parameters to estimate in these models.

    relabel_coef = {}
    for segment_name in segment_names:
        coef_backwards_map = dict(
            [(j, i) for i, j in coefficients[segment_name]["value"].items()]
        )
        relabel_coef[segment_name] = r = coefficients[segment_name]["value"].map(
            coef_backwards_map
        )
        spec[segment_name] = spec[segment_name].map(r)
    return relabel_coef


def unavail_parameters(model):
    return model.pnames[(model.pvals < -900) & (model.pholdfast != 0)]


def unavail_data_cols(model):
    locks = unavail_parameters(model)
    return [i.data for i in model.utility_ca if i.param in locks]


def unavail(model, x_ca):
    lock_data = unavail_data_cols(model)
    if len(lock_data):
        unav = x_ca[lock_data[0]] > 0
        for j in lock_data[1:]:
            unav |= x_ca[j] > 0
    return unav


# FIXME move all this to larch/general.py? see ActititySim issue #686
def _read_feather(filename, name, edb_directory, **kwargs):
    filename = filename.format(name=name)
    return pd.read_feather(os.path.join(edb_directory, filename), **kwargs)


def _to_feather(df, filename, name, edb_directory, **kwargs):
    filename = filename.format(name=name)
    return df.to_feather(os.path.join(edb_directory, filename), **kwargs)


def _read_pickle(filename, name, edb_directory, **kwargs):
    filename = filename.format(name=name)
    return pd.read_pickle(os.path.join(edb_directory, filename), **kwargs)


def _to_pickle(df, filename, name, edb_directory, **kwargs):
    filename = filename.format(name=name)
    return df.to_pickle(os.path.join(edb_directory, filename), **kwargs)


def _file_exists(filename, name, edb_directory):
    filename = filename.format(name=name)
    return os.path.exists(os.path.join(edb_directory, filename))


def get_x_ca_df(alt_values, name, edb_directory, num_chunks):
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    # process x_ca with cv_to_ca with or without chunking
    x_ca_pickle_file = "{name}_x_ca.pkl"
    if num_chunks == 1:
        x_ca = cv_to_ca(alt_values)
    elif _file_exists(x_ca_pickle_file, name, edb_directory):
        # if pickle file from previous x_ca processing exist, load it to save time
        time_start = datetime.now()
        x_ca = _read_pickle(x_ca_pickle_file, name, edb_directory)
        print(
            f"x_ca data loaded from {name}_x_ca.fea - time elapsed {(datetime.now() - time_start).total_seconds()}"
        )
    else:
        time_start = datetime.now()
        # calculate num_chunks based on chunking_size (or max number of rows per chunk)
        chunking_size = round(len(alt_values) / num_chunks, 3)
        print(
            f"Using {num_chunks} chunks results in chunk size of {chunking_size} (of {len(alt_values)} total rows)"
        )
        all_chunk_ids = list(alt_values.index.get_level_values(0).unique())
        split_ids = list(split(all_chunk_ids, num_chunks))
        x_ca_list = []
        for i, chunk_ids in enumerate(split_ids):
            alt_values_i = alt_values.loc[chunk_ids]
            x_ca_i = cv_to_ca(alt_values_i)
            x_ca_list.append(x_ca_i)
            print(
                f"\rx_ca_i compute done for chunk {i+1}/{num_chunks} - time elapsed {(datetime.now() - time_start).total_seconds()}"
            )
        x_ca = pd.concat(x_ca_list, axis=0)
        # save final x_ca result as pickle file to save time for future data loading
        _to_pickle(x_ca, x_ca_pickle_file, name, edb_directory)
        print(
            f"x_ca compute done - time elapsed {(datetime.now() - time_start).total_seconds()}"
        )
    return x_ca


def nonmand_tour_freq_model(
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
    condense_parameters=False,
    segment_subset=(),
    num_chunks=1,
    *,
    alts_in_cv_format=False,
):
    """
    Prepare nonmandatory tour frequency models for estimation.

    Parameters
    ----------
    edb_directory : str
        Location of estimation data bundle for these models.
    return_data : bool, default False
        Whether to return the data used in preparing this function.
        If returned, data is a dict in the second return value.
    condense_parameters : bool, default False
        Apply a transformation whereby all parameters in each model that
        have the same initial value are converted to have the same name
        (and thus to be the same parameter, used in various places).
    """
    data = interaction_simulate_data(
        name="non_mandatory_tour_frequency",
        edb_directory=edb_directory,
        segment_subset=segment_subset,
    )

    settings = data.settings
    segment_names = [s["NAME"] for s in settings["SPEC_SEGMENTS"]]
    if len(segment_subset) > 0:
        assert set(segment_subset).issubset(
            set(segment_names)
        ), f"{segment_subset} is not a subset of {segment_names}"
        segment_names = segment_subset
    if condense_parameters:
        data.relabel_coef = link_same_value_coefficients(
            segment_names, data.coefficients, data.spec
        )
    spec = data.spec
    coefficients = data.coefficients
    chooser_data = data.chooser_data
    alt_values = data.alt_values
    alt_def = data.alt_def

    # deduplicate rows in alt_def
    alt_def = alt_def[~alt_def.index.duplicated(keep="first")]

    m = {}
    for segment_name in segment_names:
        print(f"Creating larch model for {segment_name}")
        segment_model = m[segment_name] = Model(compute_engine="numba")
        # One of the alternatives is coded as 0, so
        # we need to explicitly initialize the MNL nesting graph
        # and set to root_id to a value other than zero.
        segment_model.initialize_graph(alternative_codes=alt_def.index, root_id=9999)

        # Utility specifications
        segment_model.utility_ca = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=segment_name,
        )
        apply_coefficients(coefficients[segment_name], segment_model)
        segment_model.choice_co_code = "override_choice"

        # Attach Data
        x_co = (
            chooser_data[segment_name]
            .set_index("person_id")
            .rename(columns={"TAZ": "HOMETAZ"})
        )
        if alts_in_cv_format:
            print("\t performing cv to ca step")
            x_ca = get_x_ca_df(
                alt_values=alt_values[segment_name].set_index(
                    ["person_id", "variable"]
                ),
                name=segment_name,
                edb_directory=edb_directory.format(name="non_mandatory_tour_frequency"),
                num_chunks=num_chunks,
            )
        else:
            x_ca = alt_values[segment_name].set_index(["person_id", "alt_id"])

        d_co = Dataset.construct.from_idco(
            x_co,
            alts=alt_def.index.rename("alt_id"),
        )
        x_ca["_avail_"] = ~unavail(segment_model, x_ca)
        # we set crack to False here so that we do not dissolve zero variance IDCAs
        d_ca = Dataset.construct.from_idca(x_ca, crack=False)
        d = d_ca.merge(d_co)
        m[segment_name].datatree = d
        m[segment_name].availability_ca_var = "_avail_"

    if return_data:
        return m, data
    return m
