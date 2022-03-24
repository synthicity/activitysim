# ActivitySim
# See full license in LICENSE.txt.

import logging

import pandas as pd
import numpy as np
import itertools
import os

from activitysim.core.interaction_simulate import interaction_simulate
from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import expressions
from activitysim.core import logit
from activitysim.core import assign
from activitysim.core import los

from activitysim.abm.models.vehicle_type_choice import read_vehicle_type_data

from activitysim.core.util import assign_in_place

from .util.mode import mode_choice_simulate
from .util import estimation


logger = logging.getLogger(__name__)


def annotate_vehicle_allocation(model_settings, trace_label):
    # - annotate tours table
    tours = inject.get_table('tours').to_frame()
    expressions.assign_columns(
        df=tours,
        model_settings=model_settings.get('annotate_tours'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_tours'))
    pipeline.replace_table("tours", tours)


@inject.step()
def vehicle_allocation(
        persons,
        households,
        vehicles,
        tours,
        tours_merged,
        network_los,
        chunk_size,
        trace_hh_id):
    """Selects a vehicle for each occupancy level for each tour.

    Alternatives consist of the up to the number of household vehicles plus one
    option for non-household vehicles.

    The model will be run 3 times, once for each tour occupancy defined in the model yaml.
    Output tour table will have 3 columns added, one for each occupancy level.

    The user may also augment the `tours` tables with new vehicle
    type-based fields specified via expressions in "annotate_tours_vehicle_allocation.csv".

    Parameters
    ----------
    persons : orca.DataFrameWrapper
    households : orca.DataFrameWrapper
    vehicles : orca.DataFrameWrapper
    vehicles_merged : orca.DataFrameWrapper
    tours : orca.DataFrameWrapper
    tours_merged : orca.DataFrameWrapper
    chunk_size : orca.injectable
    trace_hh_id : orca.injectable

    Returns
    -------

    """
    trace_label = 'vehicle_allocation'
    model_settings_file_name = 'vehicle_allocation.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get('MODE_CHOICE_LOGSUM_COLUMN_NAME')

    estimator = estimation.manager.begin_estimation('vehicle_allocation')

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(coefficients_df)

    # ------ constructing alternatives from model spec and joining to choosers
    vehicles_wide = vehicles.to_frame().pivot_table(
        index='household_id', columns='vehicle_num',
        values='vehicle_type', aggfunc=lambda x: ''.join(x))

    alts_from_spec = model_spec.columns
    # renaming vehicle numbers to alternative names in spec
    vehicle_alt_columns_dict = {}
    for veh_num in range(1, len(alts_from_spec)):
        vehicle_alt_columns_dict[veh_num] = alts_from_spec[veh_num-1]
    vehicles_wide.rename(columns=vehicle_alt_columns_dict, inplace=True)

    # if the number of vehicles is less than the alternatives, fill with NA
    # e.g. all households only have 1 or 2 vehicles because of small sample size,
    #   still need columns for alternatives 3 and 4
    for veh_num, col_name in vehicle_alt_columns_dict.items():
        if col_name not in vehicles_wide.columns:
            vehicles_wide[col_name] = ''

    # last entry in spec is the non-hh-veh option
    vehicles_wide[alts_from_spec[-1]] = ''

    # merging vehicle alternatives to choosers
    choosers = tours_merged.to_frame().reset_index()
    choosers = pd.merge(choosers, vehicles_wide, how='left', on='household_id')
    choosers.set_index('tour_id', inplace=True)

    # merging vehicle data into choosers if supplied
    if model_settings.get('VEHICLE_TYPE_DATA_FILE'):
        vehicle_type_data = pd.read_csv(config.config_file_path(
            model_settings.get('VEHICLE_TYPE_DATA_FILE')), comment='#')
        fleet_year = model_settings.get('FLEET_YEAR')
        vehicle_type_data['age'] = (1 + fleet_year - vehicle_type_data['vehicle_year']).astype(int)
        vehicle_type_data['vehicle_type'] = vehicle_type_data[
            ['body_type', 'age', 'fuel_type']].astype(str).agg('_'.join, axis = 1)

        vehicle_data_cols = model_settings.get('VEHICLE_DATA_TO_INCLUDE')
        vehicle_type_data.set_index('vehicle_type', inplace=True)
        vehicle_type_data = vehicle_type_data[vehicle_data_cols]
        cols = vehicle_type_data.columns

        # joining data for each alternative where the column name is suffixed by
        #  the alternative number.  i.e. Range -> Range_1, Range_2, etc.
        for veh_num, col_name in vehicle_alt_columns_dict.items():
            vehicle_type_data.columns = cols + '_' + str(veh_num)
            # need to ensure type incase column is all NA (i.e. no hh has 4 veh in sample)
            # choosers[col_name] = choosers[col_name].astype(str)
            choosers = pd.merge(choosers, vehicle_type_data, how='left', left_on=col_name, right_index=True)

    # ----- setup skim keys
    skim_dict = network_los.get_default_skim_dict()
    orig_col_name = 'home_zone_id'
    dest_col_name = 'destination'

    out_time_col_name = 'start'
    in_time_col_name = 'end'
    odt_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=orig_col_name, dest_key=dest_col_name,
                                               dim3_key='out_period')
    dot_skim_stack_wrapper = skim_dict.wrap_3d(orig_key=dest_col_name, dest_key=orig_col_name,
                                               dim3_key='in_period')

    choosers['in_period'] = network_los.skim_time_period_label(choosers[in_time_col_name])
    choosers['out_period'] = network_los.skim_time_period_label(choosers[out_time_col_name])

    skims = {
        "odt_skims": odt_skim_stack_wrapper.set_df(choosers),
        "dot_skims": dot_skim_stack_wrapper.set_df(choosers),
    }
    locals_dict.update(skims)

    # ------ preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:
        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    logger.info("Running %s with %d tours", trace_label, len(choosers))

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    tours = tours.to_frame()
    choosers.to_csv('allocation_choosers.csv')

    # ------ running for each occupancy level selected
    tours_veh_occup_cols = []
    for occup in model_settings.get('OCCUPANCY_LEVELS', [1]):
        logger.info("Running for occupancy = %d", occup)
        # setting occup for access in spec expressions
        locals_dict.update({'occup': occup})

        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=model_spec,
            nest_spec=nest_spec,
            skims=skims,
            locals_d=locals_dict,
            chunk_size=chunk_size,
            trace_label=trace_label,
            trace_choice_name='vehicle_allocation',
            estimator=estimator)

        # matching alt names to choices
        choices = choices.map(dict(list(zip(list(range(len(alts_from_spec))), alts_from_spec)))).to_frame()
        choices.columns = ['alt_choice']
        choices.to_csv(f'choices_{occup}.csv')
        # last alternative is the non-household vehicle option
        for alt in alts_from_spec[:-1]:
            choices.loc[choices['alt_choice'] == alt, 'choice'] = \
                choosers.loc[choices['alt_choice'] == alt, alt]
        choices.loc[choices['alt_choice'] == alts_from_spec[-1], 'choice'] = alts_from_spec[-1]

        # creating a column for choice of each occupancy level
        tours_veh_occup_col = f'vehicle_occup_{occup}'
        tours[tours_veh_occup_col] = choices['choice']
        tours_veh_occup_cols.append(tours_veh_occup_col)


    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, 'households', 'vehicle_type_choice')
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    pipeline.replace_table("tours", tours)

    tracing.print_summary('vehicle_allocation', tours[tours_veh_occup_cols], value_counts=True)

    annotate_settings = model_settings.get('annotate_tours', None)
    if annotate_settings:
        annotate_vehicle_allocation(model_settings, trace_label)

    if trace_hh_id:
        tracing.trace_df(tours,
                         label='vehicle_allocation',
                         warn_if_empty=True)
