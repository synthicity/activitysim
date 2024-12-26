from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from larch import Dataset, Model
from larch.util import Dict

from .general import (
    apply_coefficients,
    construct_nesting_tree,
    dict_of_linear_utility_from_spec,
    remove_apostrophes,
)


def stop_frequency_data(
    edb_directory="output/estimation_data_bundle/{name}/",
    settings_file="{name}_model_settings.yaml",
    chooser_data_file="{name}_values_combined.csv",
    values_index_col="tour_id",
):
    name = "stop_frequency"
    edb_directory = edb_directory.format(name=name)

    settings_file = settings_file.format(name=name)
    try:
        with open(os.path.join(edb_directory, settings_file), "r") as yf:
            settings = yaml.load(
                yf,
                Loader=yaml.SafeLoader,
            )
    except FileNotFoundError:
        # search in all first level subdirectories for the settings file
        for subdir in os.listdir(edb_directory):
            if os.path.isdir(os.path.join(edb_directory, subdir)):
                try:
                    with open(
                        os.path.join(edb_directory, subdir, settings_file), "r"
                    ) as yf:
                        settings = yaml.load(
                            yf,
                            Loader=yaml.SafeLoader,
                        )
                        break
                except FileNotFoundError:
                    pass
        else:
            raise

    segments = [i["primary_purpose"] for i in settings["SPEC_SEGMENTS"]]

    master_coef = {}
    prior_segs = []
    coef_map = {seg: {} for seg in segments}

    segment_coef = {}
    for seg_ in settings["SPEC_SEGMENTS"]:
        seg_purpose = seg_["primary_purpose"]
        seg_subdir = Path(os.path.join(edb_directory, seg_purpose))
        segment_coef[seg_["primary_purpose"]] = pd.read_csv(
            seg_subdir / seg_["COEFFICIENTS"], index_col="coefficient_name", comment="#"
        )

    for seg in segments:
        for cname, value in segment_coef[seg].value.items():
            if cname in master_coef:
                if master_coef[cname] == value:
                    coef_map[seg][cname] = cname
                else:
                    for pseg in prior_segs:
                        if master_coef.get(f"{cname}_{pseg}", None) == value:
                            coef_map[seg][cname] = f"{cname}_{pseg}"
                            break
                    else:  # no break
                        master_coef[f"{cname}_{seg}"] = value
                        coef_map[seg][cname] = f"{cname}_{seg}"
            else:
                master_coef[cname] = value
                coef_map[seg][cname] = cname
        prior_segs.append(seg)

    # rewrite revised spec files with common segment_coef names
    for seg in segments:
        seg_subdir = Path(os.path.join(edb_directory, seg))
        with open(seg_subdir / f"stop_frequency_SPEC.csv", "rt") as f:
            spec = f.read()
        for kcoef, v in coef_map[seg].items():
            spec = spec.replace(kcoef, v)
        with open(seg_subdir / f"stop_frequency_SPEC_.csv", "wt") as f:
            f.write(spec)

    master_coef_df = pd.DataFrame(data=master_coef, index=["value"]).T
    master_coef_df.index.name = "coefficient_name"

    seg_coefficients = []
    seg_spec = []
    seg_alt_names = []
    seg_alt_codes = []
    seg_alt_names_to_codes = []
    seg_alt_codes_to_names = []
    seg_chooser_data = []

    for seg in settings["SPEC_SEGMENTS"]:
        seg_purpose = seg["primary_purpose"]
        seg_subdir = Path(os.path.join(edb_directory, seg_purpose))
        coeffs_ = pd.read_csv(
            seg_subdir / seg["COEFFICIENTS"], index_col="coefficient_name", comment="#"
        )
        coeffs_.index = pd.Index(
            [f"{i}_{seg_purpose}" for i in coeffs_.index], name="coefficient_name"
        )
        seg_coefficients.append(coeffs_)
        spec = pd.read_csv(seg_subdir / "stop_frequency_SPEC_.csv", comment="#")
        spec = remove_apostrophes(spec, ["Label"])
        # spec.iloc[:, 3:] = spec.iloc[:, 3:].applymap(lambda x: f"{x}_{seg_purpose}" if not pd.isna(x) else x)
        seg_spec.append(spec)

        alt_names = list(spec.columns[3:])
        alt_codes = np.arange(1, len(alt_names) + 1)
        alt_names_to_codes = dict(zip(alt_names, alt_codes))
        alt_codes_to_names = dict(zip(alt_codes, alt_names))

        seg_alt_names.append(alt_names)
        seg_alt_codes.append(alt_codes)
        seg_alt_names_to_codes.append(alt_names_to_codes)
        seg_alt_codes_to_names.append(alt_codes_to_names)

        # load parquet if available, otherwise pickle, or if all else fails csv
        if (
            (seg_subdir / chooser_data_file.format(name=name))
            .with_suffix(".parquet")
            .exists()
        ):
            chooser_data = pd.read_parquet(
                (seg_subdir / chooser_data_file.format(name=name)).with_suffix(
                    ".parquet"
                ),
            ).set_index(values_index_col)
        elif (
            (seg_subdir / chooser_data_file.format(name=name))
            .with_suffix(".pkl")
            .exists()
        ):
            chooser_data = pd.read_pickle(
                (seg_subdir / chooser_data_file.format(name=name)).with_suffix(".pkl"),
            ).set_index(values_index_col)
        else:
            chooser_data = pd.read_csv(
                seg_subdir / chooser_data_file.format(name=name),
                index_col=values_index_col,
            )
        seg_chooser_data.append(chooser_data)

    return Dict(
        edb_directory=Path(edb_directory),
        settings=settings,
        chooser_data=seg_chooser_data,
        coefficients=master_coef_df,
        spec=seg_spec,
        alt_names=seg_alt_names,
        alt_codes=seg_alt_codes,
        alt_names_to_codes=seg_alt_names_to_codes,
        alt_codes_to_names=seg_alt_codes_to_names,
        segments=segments,
        coefficient_map=coef_map,
        segment_coefficients=segment_coef,
    )


def stop_frequency_model(
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    data = stop_frequency_data(
        edb_directory=edb_directory,
        values_index_col="tour_id",
    )

    models = []

    for n in range(len(data.spec)):

        coefficients = data.coefficients
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

        if settings.get("LOGIT_TYPE") == "NL":
            tree = construct_nesting_tree(data.alt_names[n], settings["NESTS"])
            m = Model(graph=tree, compute_engine="numba")
        else:
            m = Model(compute_engine="numba")

        m.utility_co = dict_of_linear_utility_from_spec(
            spec,
            "Label",
            dict(zip(alt_names, alt_codes)),
        )

        apply_coefficients(coefficients, m)

        avail = True

        d = Dataset.construct.from_idco(
            chooser_data, alts=dict(zip(alt_codes, alt_names))
        )
        # d = DataFrames(
        #     co=chooser_data,
        #     av=avail,
        #     alt_codes=alt_codes,
        #     alt_names=alt_names,
        # )

        m.datatree = d
        m.choice_co_code = "override_choice_code"
        m.availability_any = True
        models.append(m)

    from larch.model.model_group import ModelGroup

    models = ModelGroup(models)

    if return_data:
        return (
            models,
            data,
        )

    return models


def update_segment_coefficients(model, data, result_dir=Path("."), output_file=None):
    for m, segment_name in zip(model, data.segments):
        coefficient_map = data.coefficient_map[segment_name]
        segment_c = []
        master_c = []
        for c_local, c_master in coefficient_map.items():
            if c_master in model.pf.index:
                segment_c.append(c_local)
                master_c.append(c_master)
        coefficients = data.segment_coefficients[segment_name].copy()
        coefficients.loc[segment_c, "value"] = model.pf.loc[
            master_c, "value"
        ].to_numpy()
        if output_file is not None:
            os.makedirs(result_dir, exist_ok=True)
            coefficients.reset_index().to_csv(
                result_dir / output_file.format(segment_name=segment_name),
                index=False,
            )
