from __future__ import annotations

import os
from pathlib import Path
from typing import Collection

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from larch import Dataset, Model, P, X
from larch.util import Dict

from .general import (
    apply_coefficients,
    clean_values,
    construct_nesting_tree,
    explicit_value_parameters,
    linear_utility_from_spec,
    remove_apostrophes,
)
from .simple_simulate import construct_availability, simple_simulate_data


def mode_choice_model(
    name,
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
    override_filenames=None,
):
    if override_filenames is None:
        override_filenames = {}
    edb_directory = edb_directory.format(name=name)
    data = simple_simulate_data(
        name=name,
        edb_directory=edb_directory,
        **override_filenames,
    )
    coefficients = data.coefficients
    coef_template = data.coef_template
    spec = data.spec
    chooser_data = data.chooser_data
    settings = data.settings

    chooser_data = clean_values(
        chooser_data,
        alt_names_to_codes=data.alt_names_to_codes,
        choice_code="override_choice_code",
    )

    tree = construct_nesting_tree(data.alt_names, settings["NESTS"])

    purposes = list(coef_template.columns)
    if "atwork" in name:
        purposes = ["atwork"]
    elif "atwork" in purposes:
        purposes.remove("atwork")

    # Setup purpose specific models
    m = {
        purpose: Model(graph=tree, title=purpose, compute_engine="numba")
        for purpose in purposes
    }
    for alt_code, alt_name in tree.elemental_names().items():
        # Read in base utility function for this alt_name
        u = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=alt_name,
            ignore_x=("#",),
        )
        for purpose in purposes:
            # Modify utility function based on template for purpose
            u_purp = sum(
                (P(coef_template[purpose].get(i.param, i.param)) * i.data * i.scale)
                for i in u
            )
            m[purpose].utility_co[alt_code] = u_purp

    for model in m.values():
        explicit_value_parameters(model)
        model.availability_ca_var = "_avail_"
    apply_coefficients(coefficients, m)

    avail = construct_availability(
        m[purposes[0]], chooser_data, data.alt_codes_to_names
    )

    d = Dataset.construct.from_idco(
        chooser_data, alts=dict(zip(data.alt_codes, data.alt_names))
    )
    d["_avail_"] = xr.DataArray(avail, dims=(d.dc.CASEID, d.dc.ALTID))

    if "atwork" not in name:
        for purpose, model in m.items():
            model.datatree = d.dc.query_cases(f"tour_type=='{purpose}'")
            model.choice_co_code = "override_choice_code"
    else:
        for purpose, model in m.items():
            model.datatree = d
            model.choice_co_code = "override_choice_code"

    from larch.model.model_group import ModelGroup

    mg = ModelGroup(m.values())
    explicit_value_parameters(mg)
    apply_coefficients(coefficients, mg)

    if return_data:
        return (
            mg,
            Dict(
                edb_directory=Path(edb_directory),
                chooser_data=chooser_data,
                avail=avail,
                coefficients=coefficients,
                coef_template=coef_template,
                spec=spec,
                settings=settings,
            ),
        )

    return mg


def tour_mode_choice_model(
    name="tour_mode_choice",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return mode_choice_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
    )


def trip_mode_choice_model(
    name="trip_mode_choice",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return mode_choice_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
    )


def atwork_subtour_mode_choice_model(
    name="atwork_subtour_mode_choice",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return mode_choice_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
        override_filenames=dict(
            coefficients_file="tour_mode_choice_coefficients.csv",
        ),
    )
