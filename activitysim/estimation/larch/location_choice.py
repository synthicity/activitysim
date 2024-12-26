from __future__ import annotations

import collections
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Collection

import numpy as np
import pandas as pd
import yaml
from larch import Dataset, Model, P, X
from larch.util import Dict

from .general import (
    apply_coefficients,
    construct_nesting_tree,
    cv_to_ca,
    explicit_value_parameters,
    linear_utility_from_spec,
    remove_apostrophes,
    str_repr,
)


def size_coefficients_from_spec(size_spec):
    size_coef = size_spec.stack().reset_index()
    size_coef.index = size_coef.iloc[:, 0] + "_" + size_coef.iloc[:, 1]
    size_coef = size_coef.loc[size_coef.iloc[:, 2] > 0]
    size_coef["constrain"] = "F"
    one_each = size_coef.groupby("segment").first().reset_index()
    size_coef.loc[one_each.iloc[:, 0] + "_" + one_each.iloc[:, 1], "constrain"] = "T"
    size_coef = size_coef.iloc[:, 2:]
    size_coef.columns = ["value", "constrain"]
    size_coef.index.name = "coefficient_name"
    size_coef["value"] = np.log(size_coef["value"])
    return size_coef


LocationChoiceData = collections.namedtuple(
    "LocationChoiceData",
    field_names=[
        "edb_directory",
        "alt_values",
        "chooser_data",
        "coefficients",
        "landuse",
        "spec",
        "size_spec",
        "master_size_spec",
        "model_selector",
        "settings",
    ],
)


def location_choice_model(
    name="workplace_location",
    edb_directory="output/estimation_data_bundle/{name}/",
    coefficients_file="{name}_coefficients.csv",
    spec_file="{name}_SPEC.csv",
    size_spec_file="{name}_size_terms.csv",
    alt_values_file="{name}_alternatives_combined.csv",
    chooser_file="{name}_choosers_combined.csv",
    settings_file="{name}_model_settings.yaml",
    landuse_file="{name}_landuse.csv",
    return_data=False,
    alt_values_to_feather=False,
    chunking_size=None,
    *,
    alts_in_cv_format=False,
) -> Model | tuple[Model, LocationChoiceData]:
    model_selector = name.replace("_location", "")
    model_selector = model_selector.replace("_destination", "")
    model_selector = model_selector.replace("_subtour", "")
    model_selector = model_selector.replace("_tour", "")
    if model_selector == "joint":
        model_selector = "non_mandatory"
    edb_directory = edb_directory.format(name=name)

    def _read_csv(filename, **kwargs):
        filename = Path(edb_directory).joinpath(filename.format(name=name))
        if filename.with_suffix(".parquet").exists():
            print("loading from", filename.with_suffix(".parquet"))
            return pd.read_parquet(filename.with_suffix(".parquet"), **kwargs)
        print("loading from", filename)
        return pd.read_csv(filename, **kwargs)

    def _read_feather(filename, **kwargs):
        filename = filename.format(name=name)
        return pd.read_feather(os.path.join(edb_directory, filename), **kwargs)

    def _to_feather(df, filename, **kwargs):
        filename = filename.format(name=name)
        return df.to_feather(os.path.join(edb_directory, filename), **kwargs)

    def _read_pickle(filename, **kwargs):
        filename = filename.format(name=name)
        return pd.read_pickle(os.path.join(edb_directory, filename))

    def _to_pickle(df, filename, **kwargs):
        filename = filename.format(name=name)
        return df.to_pickle(os.path.join(edb_directory, filename))

    def _file_exists(filename):
        filename = filename.format(name=name)
        return os.path.exists(os.path.join(edb_directory, filename))

    coefficients = _read_csv(
        coefficients_file,
        index_col="coefficient_name",
    )
    spec = _read_csv(spec_file, comment="#")

    # read alternative values either as csv or feather file
    alt_values_fea_file = alt_values_file.replace(".csv", ".fea")
    if os.path.exists(
        os.path.join(edb_directory, alt_values_fea_file.format(name=name))
    ):
        alt_values = _read_feather(alt_values_fea_file)
    else:
        alt_values = _read_csv(alt_values_file)
        if alt_values_to_feather:
            _to_feather(df=alt_values, filename=alt_values_fea_file)
    chooser_data = _read_csv(chooser_file)
    landuse = _read_csv(landuse_file, index_col="zone_id")
    master_size_spec = _read_csv(size_spec_file)

    # remove temp rows from spec, ASim uses them to calculate the other values written
    # to the EDB, but they are not actually part of the utility function themselves.
    spec = spec.loc[~spec.Expression.isna()]
    spec = spec.loc[~spec.Expression.str.startswith("_")].copy()

    settings_file = settings_file.format(name=name)
    with open(os.path.join(edb_directory, settings_file), "r") as yf:
        settings = yaml.load(
            yf,
            Loader=yaml.SafeLoader,
        )

    include_settings = settings.get("include_settings")
    if include_settings:
        include_settings = os.path.join(edb_directory, include_settings)
    if include_settings and os.path.exists(include_settings):
        with open(include_settings, "r") as yf:
            more_settings = yaml.load(
                yf,
                Loader=yaml.SafeLoader,
            )
        settings.update(more_settings)

    CHOOSER_SEGMENT_COLUMN_NAME = settings.get("CHOOSER_SEGMENT_COLUMN_NAME")
    SEGMENT_IDS = settings.get("SEGMENT_IDS")
    if SEGMENT_IDS is None:
        SEGMENTS = settings.get("SEGMENTS")
        if SEGMENTS is not None:
            SEGMENT_IDS = {i: i for i in SEGMENTS}

    SIZE_TERM_SELECTOR = (
        settings.get("SIZE_TERM_SELECTOR", model_selector) or model_selector
    )

    # filter size spec for this location choice only
    size_spec = (
        master_size_spec.query(f"model_selector == '{SIZE_TERM_SELECTOR}'")
        .drop(columns="model_selector")
        .set_index("segment")
    )
    size_spec = size_spec.loc[:, size_spec.max() > 0]
    assert (
        len(size_spec) > 0
    ), f"Empty size_spec, is model_selector {SIZE_TERM_SELECTOR} in your size term file?"

    size_coef = size_coefficients_from_spec(size_spec)

    indexes_to_drop = [
        "util_size_variable",  # pre-computed size (will be re-estimated)
        "util_size_variable_atwork",  # pre-computed size (will be re-estimated)
        "util_utility_adjustment",  # shadow pricing (ignored in estimation)
        "@df['size_term'].apply(np.log1p)",  # pre-computed size (will be re-estimated)
    ]
    if "Label" in spec.columns:
        indexes_to_drop = [i for i in indexes_to_drop if i in spec.Label.to_numpy()]
        label_column_name = "Label"
    elif "Expression" in spec.columns:
        indexes_to_drop = [
            i for i in indexes_to_drop if i in spec.Expression.to_numpy()
        ]
        label_column_name = "Expression"
    else:
        raise ValueError("cannot find Label or Expression in spec file")

    expression_labels = None
    if label_column_name == "Expression":
        expression_labels = {
            expr: f"variable_label{n:04d}"
            for n, expr in enumerate(spec.Expression.to_numpy())
        }

    # Remove shadow pricing and pre-existing size expression for re-estimation
    spec = spec.set_index(label_column_name).drop(index=indexes_to_drop).reset_index()

    if label_column_name == "Expression":
        spec.insert(0, "Label", spec["Expression"].map(expression_labels))
        if alts_in_cv_format:
            alt_values["variable"] = alt_values["variable"].map(expression_labels)
        else:
            alt_values = alt_values.rename(columns=expression_labels)
        label_column_name = "Label"

    if name == "trip_destination":
        CHOOSER_SEGMENT_COLUMN_NAME = "primary_purpose"
        primary_purposes = spec.columns[3:]
        SEGMENT_IDS = {pp: pp for pp in primary_purposes}

    chooser_index_name = chooser_data.columns[0]
    x_co = chooser_data.set_index(chooser_index_name)

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    if alts_in_cv_format:
        # if alternatives are in CV format, convert them to CA format.
        # The CV format has the chooser index as the first column and the variable name
        # as the second column, with values for each alternative in the remaining columns.
        # This format is inefficient and deprecated as of ActivitySim version 1.4.

        # process x_ca with cv_to_ca with or without chunking
        x_ca_pickle_file = "{name}_x_ca.pkl"
        if chunking_size == None:
            x_ca = cv_to_ca(
                alt_values.set_index([chooser_index_name, alt_values.columns[1]])
            )
        elif _file_exists(x_ca_pickle_file):
            # if pickle file from previous x_ca processing exist, load it to save time
            time_start = datetime.now()
            x_ca = _read_pickle(x_ca_pickle_file)
            print(
                f"x_ca data loaded from {name}_x_ca.fea - time elapsed {(datetime.now() - time_start).total_seconds()}"
            )
        else:
            time_start = datetime.now()
            # calculate num_chunks based on chunking_size (or max number of rows per chunk)
            num_chunks = int(len(alt_values) / chunking_size)
            id_col_name = alt_values.columns[0]
            all_ids = list(alt_values[id_col_name].unique())
            split_ids = list(split(all_ids, num_chunks))
            x_ca_list = []
            i = 0
            for chunk_ids in split_ids:
                alt_values_i = alt_values[alt_values[id_col_name].isin(chunk_ids)]
                x_ca_i = cv_to_ca(
                    alt_values_i.set_index(
                        [chooser_index_name, alt_values_i.columns[1]]
                    )
                )
                x_ca_list.append(x_ca_i)
                print(
                    f"\rx_ca_i compute done for chunk {i}/{num_chunks} - time elapsed {(datetime.now() - time_start).total_seconds()}"
                )
                i = i + 1
            x_ca = pd.concat(x_ca_list, axis=0)
            # save final x_ca result as pickle file to save time for future data loading
            _to_pickle(df=x_ca, filename=x_ca_pickle_file)
            print(
                f"x_ca compute done - time elapsed {(datetime.now() - time_start).total_seconds()}"
            )
    else:
        # otherwise, we assume that the alternatives are already in the correct IDCA format with
        # the cases and alternatives as the first two columns, and the variables as the
        # remaining columns.  This is a much more efficient format for the data.
        assert alt_values.columns[0] == chooser_index_name
        x_ca = alt_values.set_index([chooser_index_name, alt_values.columns[1]])

    if CHOOSER_SEGMENT_COLUMN_NAME is not None:
        # label segments with names
        SEGMENT_IDS_REVERSE = {v: k for k, v in SEGMENT_IDS.items()}
        x_co["_segment_label"] = x_co[CHOOSER_SEGMENT_COLUMN_NAME].apply(
            lambda x: SEGMENT_IDS_REVERSE[x]
        )
    else:
        x_co["_segment_label"] = size_spec.index[0]

    # compute total size values by segment
    for segment in size_spec.index:
        total_size_segment = pd.Series(0, index=landuse.index)
        x_co["total_size_" + segment] = 0
        for land_use_field in size_spec.loc[segment].index:
            total_size_segment += (
                landuse[land_use_field] * size_spec.loc[segment, land_use_field]
            )
        x_co["total_size_" + segment] = total_size_segment.loc[
            x_co["override_choice"]
        ].to_numpy()

    # for each chooser, collate the appropriate total size value
    x_co["total_size_segment"] = 0
    for segment in size_spec.index:
        labels = "total_size_" + segment
        rows = x_co["_segment_label"] == segment
        x_co.loc[rows, "total_size_segment"] = x_co[labels][rows]

    # Remove choosers with invalid observed choice (appropriate total size value = 0)
    valid_observed_zone = x_co["total_size_segment"] > 0
    prior_n_cases = len(x_co)
    x_co = x_co[valid_observed_zone]
    x_ca = x_ca[x_ca.index.get_level_values(chooser_index_name).isin(x_co.index)]
    after_n_cases = len(x_co)
    if prior_n_cases != after_n_cases:
        warnings.warn(
            f"Removed {prior_n_cases - after_n_cases} choosers with invalid (zero-sized) observed choice",
            stacklevel=2,
        )

    # Merge land use characteristics into CA data
    x_ca_1 = pd.merge(
        x_ca, landuse, left_on=x_ca.index.get_level_values(1), right_index=True
    )
    x_ca_1 = x_ca_1.sort_index()

    # relabel zones to reduce memory usage.
    # We will core the original zone ids in a new column _original_zone_id,
    # and create a new index with a dummy zone id.  This way, if we have sampled
    # only a subset of 30 zones, then we only need 30 unique alternatives in the
    # data structure.
    original_zone_ids = x_ca_1.index.get_level_values(1)

    dummy_zone_ids_index = pd.MultiIndex.from_arrays(
        [
            x_ca_1.index.get_level_values(0),
            x_ca_1.groupby(level=0).cumcount() + 1,
        ],
        names=[x_ca_1.index.names[0], "dummy_zone_id"],
    )
    x_ca_1.index = dummy_zone_ids_index
    x_ca_1["_original_zone_id"] = original_zone_ids
    choice_def = {"choice_ca_var": "override_choice == _original_zone_id"}

    # Availability of choice zones
    if "util_no_attractions" in x_ca_1:
        av = (
            x_ca_1["util_no_attractions"]
            .apply(lambda x: False if x == 1 else True)
            .astype(np.int8)
            .to_xarray()
        )
    elif "@df['size_term']==0" in x_ca_1:
        av = (
            x_ca_1["@df['size_term']==0"]
            .apply(lambda x: False if x == 1 else True)
            .astype(np.int8)
            .to_xarray()
        )
    elif expression_labels is not None and "@df['size_term']==0" in expression_labels:
        av = (
            x_ca_1[expression_labels["@df['size_term']==0"]]
            .apply(lambda x: False if x == 1 else True)
            .astype(np.int8)
            .to_xarray()
        )
    else:
        av = None

    assert len(x_co) > 0, "Empty chooser dataframe"
    assert len(x_ca_1) > 0, "Empty alternatives dataframe"

    d_ca = Dataset.construct.from_idca(x_ca_1)
    d_co = Dataset.construct.from_idco(x_co)
    d = d_ca.merge(d_co)
    if av is not None:
        d["_avail_"] = av

    m = Model(datatree=d, compute_engine="numba")

    # One of the alternatives might be coded as 0, so
    # we need to explicitly initialize the MNL nesting graph
    # and set to root_id to a value other than zero.
    root_id = 0
    if root_id in d.dc.altids():
        root_id = -1
    m.initialize_graph(alternative_codes=d.dc.altids(), root_id=root_id)

    if len(spec.columns) == 4 and all(
        spec.columns == ["Label", "Description", "Expression", "coefficient"]
    ):
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=spec.columns[-1],
            ignore_x=("local_dist",),
            x_validator=d,
            expr_col="Expression",
        )
    elif (
        len(spec.columns) == 4
        and all(spec.columns[:3] == ["Label", "Description", "Expression"])
        and len(SEGMENT_IDS) == 1
        and spec.columns[3] == list(SEGMENT_IDS.values())[0]
    ):
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=spec.columns[-1],
            ignore_x=("local_dist",),
            x_validator=d,
            expr_col="Expression",
        )
    else:
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col=label_column_name,
            p_col=SEGMENT_IDS,
            ignore_x=("local_dist",),
            segment_id=CHOOSER_SEGMENT_COLUMN_NAME,
            x_validator=d,
            expr_col="Expression",
        )

    if CHOOSER_SEGMENT_COLUMN_NAME is None:
        assert len(size_spec) == 1
        m.quantity_ca = sum(
            P(f"{i}_{q}") * X(q)
            for i in size_spec.index
            for q in size_spec.columns
            if size_spec.loc[i, q] != 0
        )
    else:
        m.quantity_ca = sum(
            P(f"{i}_{q}")
            * X(q)
            * X(f"{CHOOSER_SEGMENT_COLUMN_NAME}=={str_repr(SEGMENT_IDS[i])}")
            for i in size_spec.index
            for q in size_spec.columns
            if size_spec.loc[i, q] != 0
        )

    apply_coefficients(coefficients, m, minimum=-25, maximum=25)
    apply_coefficients(size_coef, m, minimum=-6, maximum=6)

    m.choice_def(choice_def)
    if av is not None:
        m.availability_ca_var = "_avail_"
    else:
        m.availability_any = True

    if return_data:
        return (
            m,
            LocationChoiceData(
                edb_directory=Path(edb_directory),
                alt_values=alt_values,
                chooser_data=chooser_data,
                coefficients=coefficients,
                landuse=landuse,
                spec=spec,
                size_spec=size_spec,
                master_size_spec=master_size_spec,
                model_selector=model_selector,
                settings=settings,
            ),
        )

    return m


def update_size_spec(model, data, result_dir=Path("."), output_file=None):
    master_size_spec = data.master_size_spec
    size_spec = data.size_spec
    model_selector = data.model_selector

    # Write size coefficients into size_spec
    for c in size_spec.columns:
        for i in size_spec.index:
            param_name = f"{i}_{c}"
            j = (master_size_spec["segment"] == i) & (
                master_size_spec["model_selector"] == model_selector
            )
            try:
                master_size_spec.loc[j, c] = np.exp(model.get_value(param_name))
            except KeyError:
                pass

    # Rescale each row to total 1, not mathematically needed
    # but to maintain a consistent approach from existing ASim
    master_size_spec.iloc[:, 2:] = master_size_spec.iloc[:, 2:].div(
        master_size_spec.iloc[:, 2:].sum(1), axis=0
    )

    if output_file is not None:
        os.makedirs(result_dir, exist_ok=True)
        master_size_spec.reset_index().to_csv(
            result_dir / output_file,
            index=False,
        )

    return master_size_spec


def workplace_location_model(**kwargs):
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="workplace_location",
        **kwargs,
    )


def school_location_model(**kwargs):
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="school_location",
        **kwargs,
    )


def atwork_subtour_destination_model(**kwargs):
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="atwork_subtour_destination",
        **kwargs,
    )


def joint_tour_destination_model(**kwargs):
    # goes with non_mandatory_tour_destination
    unused = kwargs.pop("name", None)
    if "coefficients_file" not in kwargs:
        kwargs["coefficients_file"] = "non_mandatory_tour_destination_coefficients.csv"
    return location_choice_model(
        name="joint_tour_destination",
        **kwargs,
    )


def non_mandatory_tour_destination_model(**kwargs):
    # goes with joint_tour_destination
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="non_mandatory_tour_destination",
        **kwargs,
    )


def trip_destination_model(**kwargs):
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="trip_destination",
        **kwargs,
    )
