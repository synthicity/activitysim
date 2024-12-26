from __future__ import annotations

import os
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


def schedule_choice_model(
    name,
    edb_directory="output/estimation_data_bundle/{name}/",
    coefficients_file="{name}_coefficients.csv",
    spec_file="{name}_SPEC.csv",
    alt_values_file="{name}_alternatives_combined.csv",
    chooser_file="{name}_choosers_combined.csv",
    settings_file="{name}_model_settings.yaml",
    return_data=False,
    *,
    alts_in_cv_format=False,
):
    model_selector = name.replace("_location", "")
    model_selector = model_selector.replace("_destination", "")
    model_selector = model_selector.replace("_subtour", "")
    model_selector = model_selector.replace("_tour", "")
    edb_directory = edb_directory.format(name=name)

    def _read_csv(filename, optional=False, **kwargs):
        filename = Path(edb_directory).joinpath(filename.format(name=name))
        if filename.with_suffix(".parquet").exists():
            print("loading from", filename.with_suffix(".parquet"))
            return pd.read_parquet(filename.with_suffix(".parquet"), **kwargs)
        if filename.exists():
            print("loading from", filename)
            return pd.read_csv(filename, **kwargs)
        if optional:
            return None
        raise FileNotFoundError(filename)

    settings_file = settings_file.format(name=name)
    with open(os.path.join(edb_directory, settings_file), "r") as yf:
        settings = yaml.load(
            yf,
            Loader=yaml.SafeLoader,
        )

    try:
        coefficients = _read_csv(
            coefficients_file,
            index_col="coefficient_name",
        )
    except FileNotFoundError:
        # possibly mis-named file is shown in settings
        coefficients_file = settings.get("COEFFICIENTS", coefficients_file)
        coefficients = _read_csv(
            coefficients_file,
            index_col="coefficient_name",
        )

    spec = _read_csv(spec_file, comment="#")
    alt_values = _read_csv(alt_values_file)
    chooser_data = _read_csv(chooser_file)

    # remove temp rows from spec, ASim uses them to calculate the other values written
    # to the EDB, but they are not actually part of the utility function themselves.
    spec = spec.loc[~spec.Expression.str.startswith("_")].copy()

    include_settings = settings.get("include_settings")
    if include_settings:
        with open(os.path.join(edb_directory, include_settings), "r") as yf:
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

    if "Label" in spec.columns:
        label_column_name = "Label"
    elif "Expression" in spec.columns:
        label_column_name = "Expression"
    else:
        raise ValueError("cannot find Label or Expression in spec file")

    m = Model(compute_engine="numba")
    if len(spec.columns) == 4 and (
        [c.lower() for c in spec.columns]
        == ["label", "description", "expression", "coefficient"]
    ):
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=spec.columns[-1],
            ignore_x=("local_dist",),
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
        )
    else:
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col=label_column_name,
            p_col=SEGMENT_IDS,
            ignore_x=("local_dist",),
            segment_id=CHOOSER_SEGMENT_COLUMN_NAME,
        )

    apply_coefficients(coefficients, m, minimum=-25, maximum=25)

    chooser_index_name = chooser_data.columns[0]
    x_co = chooser_data.set_index(chooser_index_name).dropna(axis=1, how="all")
    alt_values.fillna(0, inplace=True)
    if alts_in_cv_format:
        x_ca = cv_to_ca(
            alt_values.set_index([chooser_index_name, alt_values.columns[1]]),
            required_labels=spec[label_column_name],
        )
    else:
        # the alternative code is "tdd"
        x_ca = alt_values.set_index([chooser_index_name, "tdd"])

    # if CHOOSER_SEGMENT_COLUMN_NAME is not None:
    #     # label segments with names
    #     SEGMENT_IDS_REVERSE = {v: k for k, v in SEGMENT_IDS.items()}
    #     x_co["_segment_label"] = x_co[CHOOSER_SEGMENT_COLUMN_NAME].apply(
    #         lambda x: SEGMENT_IDS_REVERSE[x]
    #     )
    # else:
    #     x_co["_segment_label"] = size_spec.index[0]

    alt_codes = np.arange(len(x_ca.index.levels[1])) + 1
    x_ca.index = x_ca.index.set_levels(alt_codes, level=1)
    x_co["override_choice_plus1"] = x_co["override_choice"] + 1
    x_co["model_choice_plus1"] = x_co["model_choice"] + 1

    unavail_coefs = coefficients.query("(constrain == 'T') & (value < -900)").index
    unavail_data = [i.data for i in m.utility_ca if i.param in unavail_coefs]

    if "mode_choice_logsum" in x_ca and not len(unavail_data):
        joint_avail = "~(mode_choice_logsum_missing)"
    elif len(unavail_data):
        joint_unavail = "|".join(f"({i}>0)" for i in unavail_data)
        joint_avail = f"~({joint_unavail})"
    else:
        joint_avail = None

    # d = DataFrames(co=x_co, ca=x_ca, av=joint_avail)  # larch 5.7
    d_ca = Dataset.construct.from_idca(x_ca)
    if joint_avail == "~(mode_choice_logsum_missing)":
        tmp = np.isnan(d_ca["mode_choice_logsum"])
        tmp = tmp.drop_vars(tmp.coords)
        d_ca = d_ca.assign(mode_choice_logsum_missing=tmp)
    d_co = Dataset.construct.from_idco(x_co)
    d = d_ca.merge(d_co)
    # if joint_avail is not None:
    #     d["_avail_"] = joint_avail

    m.datatree = d
    m.choice_co_code = "override_choice_plus1"
    if joint_avail is not None:
        m.availability_ca_var = joint_avail

    if return_data:
        return (
            m,
            Dict(
                edb_directory=Path(edb_directory),
                alt_values=alt_values,
                chooser_data=chooser_data,
                coefficients=coefficients,
                spec=spec,
                model_selector=model_selector,
                joint_avail=joint_avail,
            ),
        )

    return m


def construct_availability_ca(model, chooser_data, alt_codes_to_names):
    """
    Construct an availability dataframe based on -999 parameters.

    Parameters
    ----------
    model : larch.Model
    chooser_data : pandas.DataFrame
    alt_codes_to_names : Mapping[int,str]

    Returns
    -------
    pandas.DataFrame
    """
    avail = {}
    for acode, aname in alt_codes_to_names.items():
        unavail_cols = list(
            (
                chooser_data[i.data]
                if i.data in chooser_data
                else chooser_data.eval(i.data)
            )
            for i in model.utility_co[acode]
            if (i.param == "-999" or i.param == "-999.0")
        )
        if len(unavail_cols):
            avail[acode] = sum(unavail_cols) == 0
        else:
            avail[acode] = 1
    avail = pd.DataFrame(avail).astype(np.int8)
    avail.index = chooser_data.index
    return avail


def mandatory_tour_scheduling_work_model(
    edb_directory="output/estimation_data_bundle/{name}/", return_data=False
):
    return schedule_choice_model(
        name="mandatory_tour_scheduling_work",
        edb_directory=edb_directory,
        return_data=return_data,
        coefficients_file="tour_scheduling_work_coefficients.csv",
    )


def mandatory_tour_scheduling_school_model(
    edb_directory="output/estimation_data_bundle/{name}/", return_data=False
):
    return schedule_choice_model(
        name="mandatory_tour_scheduling_school",
        edb_directory=edb_directory,
        return_data=return_data,
        coefficients_file="tour_scheduling_school_coefficients.csv",
    )


def non_mandatory_tour_scheduling_model(
    edb_directory="output/estimation_data_bundle/{name}/", return_data=False
):
    return schedule_choice_model(
        name="non_mandatory_tour_scheduling",
        edb_directory=edb_directory,
        return_data=return_data,
    )


def joint_tour_scheduling_model(
    edb_directory="output/estimation_data_bundle/{name}/", return_data=False
):
    return schedule_choice_model(
        name="joint_tour_scheduling",
        edb_directory=edb_directory,
        return_data=return_data,
    )


def atwork_subtour_scheduling_model(
    edb_directory="output/estimation_data_bundle/{name}/", return_data=False
):
    return schedule_choice_model(
        name="atwork_subtour_scheduling",
        edb_directory=edb_directory,
        return_data=return_data,
    )
