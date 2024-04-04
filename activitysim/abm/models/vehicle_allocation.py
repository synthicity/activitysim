# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.core import (
    config,
    estimation,
    expressions,
    los,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.configuration.logit import LogitComponentSettings

logger = logging.getLogger(__name__)


def annotate_vehicle_allocation(
    state: workflow.State, model_settings: VehicleAllocationSettings, trace_label: str
):
    """
    Add columns to the tours table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : VehicleAllocationSettings
    trace_label : str
    """
    tours = state.get_dataframe("tours")
    expressions.assign_columns(
        state,
        df=tours,
        model_settings=model_settings.annotate_tours,
        trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
    )
    state.add_table("tours", tours)


def get_skim_dict(network_los: los.Network_LOS, choosers: pd.DataFrame):
    """
    Returns a dictionary of skim wrappers to use in expression writing.

    Skims have origin as home_zone_id and destination as the tour destination.

    Parameters
    ----------
    network_los : activitysim.core.los.Network_LOS object
    choosers : pd.DataFrame

    Returns
    -------
    skims : dict
        index is skim wrapper name, value is the skim wrapper
    """
    skim_dict = network_los.get_default_skim_dict()
    orig_col_name = "home_zone_id"
    dest_col_name = "destination"

    out_time_col_name = "start"
    in_time_col_name = "end"
    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="out_period"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="in_period"
    )

    choosers["in_period"] = network_los.skim_time_period_label(
        choosers[in_time_col_name]
    )
    choosers["out_period"] = network_los.skim_time_period_label(
        choosers[out_time_col_name]
    )

    skims = {
        "odt_skims": odt_skim_stack_wrapper.set_df(choosers),
        "dot_skims": dot_skim_stack_wrapper.set_df(choosers),
    }
    return skims


class VehicleAllocationSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `joint_tour_scheduling` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    OCCUPANCY_LEVELS: list = [1]  # TODO Check this
    """Occupancy level

    It will create columns in the tour table selecting a vehicle for each of the
    occupancy levels. They are named vehicle_occup_1, vehicle_occup_2,... etc.
    if not supplied, will default to only one occupancy level of 1
    """

    annotate_tours: PreprocessorSettings | None = None
    """Preprocessor settings to annotate tours"""


@workflow.step
def vehicle_allocation(
    state: workflow.State,
    persons: pd.DataFrame,
    households: pd.DataFrame,
    vehicles: pd.DataFrame,
    tours: pd.DataFrame,
    tours_merged: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: VehicleAllocationSettings | None = None,
    model_settings_file_name: str = "vehicle_allocation.yaml",
    trace_label: str = "vehicle_allocation",
) -> None:
    """Selects a vehicle for each occupancy level for each tour.

    Alternatives consist of the up to the number of household vehicles plus one
    option for non-household vehicles.

    The model will be run once for each tour occupancy defined in the model yaml.
    Output tour table will columns added for each occupancy level.

    The user may also augment the `tours` tables with new vehicle
    type-based fields specified via the annotate_tours option.

    Parameters
    ----------
    state : workflow.State
    persons : pd.DataFrame
    households : pd.DataFrame
    vehicles : pd.DataFrame
    tours : pd.DataFrame
    tours_merged : pd.DataFrame
    network_los : los.Network_LOS
    """

    if model_settings is None:
        model_settings = VehicleAllocationSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    # logsum_column_name = model_settings.MODE_CHOICE_LOGSUM_COLUMN_NAME

    estimator = estimation.manager.begin_estimation(state, "vehicle_allocation")

    model_spec_raw = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec_raw, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(coefficients_df)

    # ------ constructing alternatives from model spec and joining to choosers
    vehicles_wide = vehicles.pivot_table(
        index="household_id",
        columns="vehicle_num",
        values="vehicle_type",
        aggfunc=lambda x: "".join(x),
    )

    alts_from_spec = model_spec.columns
    # renaming vehicle numbers to alternative names in spec
    vehicle_alt_columns_dict = {}
    for veh_num in range(1, len(alts_from_spec)):
        vehicle_alt_columns_dict[veh_num] = alts_from_spec[veh_num - 1]
    vehicles_wide.rename(columns=vehicle_alt_columns_dict, inplace=True)

    # if the number of vehicles is less than the alternatives, fill with NA
    # e.g. all households only have 1 or 2 vehicles because of small sample size,
    #   still need columns for alternatives 3 and 4
    for veh_num, col_name in vehicle_alt_columns_dict.items():
        if col_name not in vehicles_wide.columns:
            vehicles_wide[col_name] = ""

    # last entry in spec is the non-hh-veh option
    assert (
        alts_from_spec[-1] == "non_hh_veh"
    ), "Last option in spec needs to be non_hh_veh"
    vehicles_wide[alts_from_spec[-1]] = ""

    # merging vehicle alternatives to choosers
    choosers = tours_merged.reset_index()
    choosers = pd.merge(choosers, vehicles_wide, how="left", on="household_id")
    choosers.set_index("tour_id", inplace=True)

    ## get categorical dtype for vehicle_type and use it to create new dtype for
    ## vehicle_occup_* and selected_vehicle columns
    veh_type_dtype = vehicles["vehicle_type"].dtype
    if isinstance(veh_type_dtype, pd.CategoricalDtype):
        veh_categories = list(veh_type_dtype.categories)
        if "non_hh_veh" not in veh_categories:
            veh_categories.append("non_hh_veh")
        veh_choice_dtype = pd.CategoricalDtype(veh_categories, ordered=False)
    else:
        veh_choice_dtype = "category"

    # ----- setup skim keys
    skims = get_skim_dict(network_los, choosers)
    locals_dict.update(skims)

    # ------ preprocessor
    preprocessor_settings = model_settings.preprocessor
    if preprocessor_settings:
        expressions.assign_columns(
            state,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    logger.info("Running %s with %d tours", trace_label, len(choosers))

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    # ------ running for each occupancy level selected
    tours_veh_occup_cols = []
    for occup in model_settings.OCCUPANCY_LEVELS:
        logger.info("Running for occupancy = %d", occup)
        # setting occup for access in spec expressions
        locals_dict.update({"occup": occup})

        choices = simulate.simple_simulate(
            state,
            choosers=choosers,
            spec=model_spec,
            nest_spec=nest_spec,
            skims=skims,
            locals_d=locals_dict,
            trace_label=trace_label,
            trace_choice_name="vehicle_allocation",
            estimator=estimator,
            overflow_protection=model_settings.overflow_protection,
        )

        # matching alt names to choices
        choices = choices.map(dict(enumerate(alts_from_spec))).to_frame()
        choices.columns = ["alt_choice"]

        # last alternative is the non-household vehicle option
        for alt in alts_from_spec[:-1]:
            choices.loc[choices["alt_choice"] == alt, "choice"] = choosers.loc[
                choices["alt_choice"] == alt, alt
            ]
        choices.loc[
            choices["alt_choice"] == alts_from_spec[-1], "choice"
        ] = alts_from_spec[-1]

        # creating a column for choice of each occupancy level
        tours_veh_occup_col = f"vehicle_occup_{occup}"
        tours[tours_veh_occup_col] = choices["choice"]
        tours[tours_veh_occup_col] = tours[tours_veh_occup_col].astype(veh_choice_dtype)
        tours_veh_occup_cols.append(tours_veh_occup_col)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "households", "vehicle_allocation"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    state.add_table("tours", tours)

    tracing.print_summary(
        "vehicle_allocation", tours[tours_veh_occup_cols], value_counts=True
    )

    annotate_settings = model_settings.annotate_tours
    if annotate_settings:
        annotate_vehicle_allocation(state, model_settings, trace_label)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(tours, label="vehicle_allocation", warn_if_empty=True)
