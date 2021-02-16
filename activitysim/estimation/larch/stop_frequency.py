import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from larch.util import Dict
from larch import Model, DataFrames

from .general import (
    remove_apostrophes,
    dict_of_linear_utility_from_spec,
    apply_coefficients,
    construct_nesting_tree,
)


def stop_frequency_data(
        edb_directory="output/estimation_data_bundle/{name}/",
        settings_file="{name}_model_settings.yaml",
        chooser_data_file="{name}_values_combined.csv",
        values_index_col="tour_id",
):
    name = 'stop_frequency'
    edb_directory = Path(edb_directory.format(name=name))

    settings_file = settings_file.format(name=name)
    with open(os.path.join(edb_directory, settings_file), "r") as yf:
        settings = yaml.load(yf, Loader=yaml.SafeLoader, )

    seg_coefficients = []
    seg_spec = []
    seg_alt_names = []
    seg_alt_codes = []
    seg_alt_names_to_codes = []
    seg_alt_codes_to_names = []
    seg_chooser_data = []

    for seg in settings["SPEC_SEGMENTS"]:
        seg_purpose = seg['primary_purpose']
        seg_subdir = edb_directory / seg_purpose
        coefs = pd.read_csv(
            seg_subdir / seg['COEFFICIENTS'],
            index_col="coefficient_name",
        )
        if "constrain" not in coefs.columns:
            coefs["constrain"] = "F"
        seg_coefficients.append(coefs[["value", "constrain"]])
        spec = pd.read_csv(seg_subdir / "stop_frequency_SPEC.csv")
        spec = remove_apostrophes(spec, ["Label"])
        seg_spec.append(spec)

        alt_names = list(spec.columns[3:])
        alt_codes = np.arange(1, len(alt_names) + 1)
        alt_names_to_codes = dict(zip(alt_names, alt_codes))
        alt_codes_to_names = dict(zip(alt_codes, alt_names))

        seg_alt_names.append(alt_names)
        seg_alt_codes.append(alt_codes)
        seg_alt_names_to_codes.append(alt_names_to_codes)
        seg_alt_codes_to_names.append(alt_codes_to_names)

        chooser_data = pd.read_csv(
            seg_subdir / chooser_data_file.format(name=name),
            index_col=values_index_col,
        )
        seg_chooser_data.append(chooser_data)

    return Dict(
        edb_directory=edb_directory,
        settings=settings,
        chooser_data=seg_chooser_data,
        coefficients=seg_coefficients,
        spec=seg_spec,
        alt_names=seg_alt_names,
        alt_codes=seg_alt_codes,
        alt_names_to_codes=seg_alt_names_to_codes,
        alt_codes_to_names=seg_alt_codes_to_names,
    )


def stop_frequency_model(
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    data = stop_frequency_data(
        edb_directory=edb_directory, values_index_col="tour_id",
    )

    models = []

    for n in range(len(data.coefficients)):

        coefficients = data.coefficients[n]
        # coef_template = data.coef_template # not used
        spec = data.spec[n]
        chooser_data = data.chooser_data[n]
        settings = data.settings

        alt_names = data.alt_names[n]
        alt_codes = data.alt_codes[n]

        from .general import clean_values
        chooser_data = clean_values(
            chooser_data,
            alt_names_to_codes=data.alt_names_to_codes[n],
            choice_code="override_choice_code",
        )

        if settings.get('LOGIT_TYPE') == 'NL':
            tree = construct_nesting_tree(data.alt_names[n], settings["NESTS"])
            m = Model(graph=tree)
        else:
            m = Model()

        m.utility_co = dict_of_linear_utility_from_spec(
            spec, "Label", dict(zip(alt_names, alt_codes)),
        )

        apply_coefficients(coefficients, m)

        avail = True

        d = DataFrames(co=chooser_data, av=avail, alt_codes=alt_codes, alt_names=alt_names, )

        m.dataservice = d
        m.choice_co_code = "override_choice_code"
        models.append(m)

    from larch.model.model_group import ModelGroup
    models = ModelGroup(models)

    if return_data:
        return (
            models,
            data,
        )

    return models
