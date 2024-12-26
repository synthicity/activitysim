from __future__ import annotations

import datetime
import os
from pathlib import Path

import pandas as pd
import platformdirs
import pytest

from activitysim.core import workflow


@pytest.fixture(scope="module")
def est_data():
    cwd = os.getcwd()
    working_dir = Path(platformdirs.user_cache_dir(appname="ActivitySim")).joinpath(
        f"estimation-test-base"
    )
    working_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(working_dir)
    if not working_dir.joinpath("success.txt").exists():
        import activitysim.abm

        state = workflow.create_example("example_estimation_sf", directory=working_dir)
        state.run.all()
        working_dir.joinpath("success.txt").write_text(
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
        )

    os.chdir(working_dir.joinpath("example_estimation_sf"))
    yield str(working_dir.joinpath("example_estimation_sf"))

    os.chdir(cwd)


def _regression_check(dataframe_regression, df, basename=None, rtol=None):
    if rtol is None:
        rtol = 0.1
    dataframe_regression.check(
        df.select_dtypes("number")
        .drop(columns=["holdfast", "minimum", "maximum"], errors="ignore")
        .clip(-9e9, 9e9),
        # pandas 1.3 handles int8 dtypes as actual numbers, so holdfast needs to be dropped manually
        # we're dropping it not adding to the regression check so older pandas will also work.
        basename=basename,
        default_tolerance=dict(atol=1e-6, rtol=rtol)
        # can set a little loose, as there is sometimes a little variance in these
        # results when switching backend implementations. We're checking all
        # the parameters and the log likelihood, so modest variance in individual
        # parameters, especially those with high std errors, is not problematic.
    )


@pytest.mark.parametrize(
    "name,method",
    [
        ("free_parking", "BHHH"),
        ("mandatory_tour_frequency", "SLSQP"),
        ("joint_tour_frequency", "SLSQP"),
        ("joint_tour_composition", "SLSQP"),
        ("joint_tour_participation", "SLSQP"),
        ("mandatory_tour_frequency", "BHHH"),
        ("atwork_subtour_frequency", "SLSQP"),
        ("auto_ownership", "BHHH"),
        ("trip_mode_choice", "SLSQP"),
    ],
)
def test_simple_simulate(est_data, num_regression, dataframe_regression, name, method):
    from activitysim.estimation.larch import component_model

    m = component_model(name)
    m.load_data()
    m.doctor(repair_ch_av="-")
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method=method, options={"maxiter": 1000, "ftol": 1e-9})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename=f"test_simple_simulate_{name}_{method}_loglike",
        default_tolerance=dict(atol=1e-6, rtol=1e-3),
    )
    _regression_check(dataframe_regression, m.pf)


@pytest.mark.parametrize(
    "name,method,rtol",
    [
        ("workplace_location", "SLSQP", None),
        ("school_location", "SLSQP", None),
        ("school_location", "BHHH", None),
        ("non_mandatory_tour_destination", "SLSQP", None),
        ("atwork_subtour_destination", "BHHH", None),
        ("trip_destination", "BHHH", None),
        # ("trip_destination", "SLSQP", 0.12),
        # trip_destination model has unusual parameter variance on a couple
        # parameters when switching platforms, possibly related to default data
        # types and high standard errors.  Most parameters and the overall
        # log likelihoods behave fine, suggesting it is just a numerical
        # precision issue.
    ],
)
def test_location_model(
    est_data, num_regression, dataframe_regression, name, method, rtol
):
    from activitysim.estimation.larch import component_model, update_size_spec

    m, data = component_model(name, return_data=True)
    m.doctor(repair_av_zq="-", repair_nan_utility=True)
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method=method, options={"maxiter": 1000, "ftol": 1.0e-8})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename=f"test_loc_{name}_{method}_loglike",
    )
    _regression_check(dataframe_regression, m.pf, rtol=rtol)
    size_spec = update_size_spec(
        m,
        data,
        result_dir=None,
        output_file=None,
    )
    dataframe_regression.check(
        size_spec,
        basename=f"test_loc_{name}_{method}_size_spec",
        default_tolerance=dict(atol=1e-6, rtol=5e-2)
        # set a little loose, as there is sometimes a little variance in these
        # results when switching backend implementations.
    )


@pytest.mark.parametrize(
    "name,method",
    [
        ("non_mandatory_tour_scheduling", "SLSQP"),
        ("joint_tour_scheduling", "SLSQP"),
        # ("atwork_subtour_scheduling", "SLSQP"),  # TODO this test is unstable, needs to be updated with better data
        ("mandatory_tour_scheduling_work", "SLSQP"),
        ("mandatory_tour_scheduling_school", "SLSQP"),
    ],
)
def test_scheduling_model(est_data, num_regression, dataframe_regression, name, method):
    from activitysim.estimation.larch import component_model

    m, data = component_model(name, return_data=True)
    m.load_data()
    m.doctor(repair_ch_av="-")
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method=method, options={"maxiter": 1000, "ftol": 1.0e-9})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename=f"test_{name}_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_stop_freq_model(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch import component_model

    name = "stop_frequency"
    m, data = component_model(name, return_data=True)
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000, "ftol": 1.0e-9})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename=f"test_{name}_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_workplace_location(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch import component_model, update_size_spec

    m, data = component_model("workplace_location", return_data=True)
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP")
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_workplace_location_loglike",
    )
    _regression_check(dataframe_regression, m.pf)
    size_spec = update_size_spec(
        m,
        data,
        result_dir=None,
        output_file=None,
    )
    dataframe_regression.check(
        size_spec,
        basename="test_workplace_location_size_spec",
        default_tolerance=dict(atol=1e-6, rtol=1e-2),
    )


def test_school_location(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch import component_model, update_size_spec

    m, data = component_model("school_location", return_data=True)
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="BHHH")
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_school_location_loglike",
    )
    _regression_check(dataframe_regression, m.pf)
    size_spec = update_size_spec(
        m,
        data,
        result_dir=None,
        output_file=None,
    )
    dataframe_regression.check(
        size_spec,
        basename="test_school_location_size_spec",
        default_tolerance=dict(atol=1e-6, rtol=1e-2),
    )


def test_cdap_model(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch.cdap import cdap_model

    m = cdap_model()
    m.load_data()
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000, "ftol": 1.0e-7})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_cdap_model_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_nonmand_and_joint_tour_dest_choice(
    est_data, num_regression, dataframe_regression
):
    from activitysim.estimation.larch import component_model

    modelname = ("non_mandatory_tour_destination", "joint_tour_destination")
    m, d = component_model(modelname, return_data=True)
    m.load_data()
    m.doctor(repair_ch_av="-")
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_nonmand_and_joint_tour_dest_choice_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_tour_and_subtour_mode_choice(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch.mode_choice import (
        atwork_subtour_mode_choice_model,
        tour_mode_choice_model,
    )

    m = tour_mode_choice_model()
    s = atwork_subtour_mode_choice_model()
    m.extend(s)  # join the atwork subtour model to the main group
    m.load_data()
    m.doctor(repair_ch_av="-")
    loglike_prior = m.loglike()
    r = m.maximize_loglike(method="SLSQP", options={"maxiter": 1000, "ftol": 1.0e-9})
    num_regression.check(
        {"loglike_prior": loglike_prior, "loglike_converge": r.loglike},
        basename="test_tour_mode_choice_loglike",
    )
    _regression_check(dataframe_regression, m.pf)


def test_nonmand_tour_freq(est_data, num_regression, dataframe_regression):
    from activitysim.estimation.larch.nonmand_tour_freq import nonmand_tour_freq_model

    m = nonmand_tour_freq_model(condense_parameters=True)
    loglike_prior = {}
    expected_n_params = {
        "PTYPE_FULL": 72,
        "PTYPE_PART": 51,
        "PTYPE_UNIVERSITY": 70,
        "PTYPE_NONWORK": 77,
        "PTYPE_RETIRED": 53,
        "PTYPE_DRIVING": 43,
        "PTYPE_SCHOOL": 34,
        "PTYPE_PRESCHOOL": 25,
    }
    for segment_name in m:
        m[segment_name].load_data()
        m[segment_name].doctor(repair_ch_av="-")
        loglike_prior[segment_name] = m[segment_name].loglike()
        assert len(m[segment_name].pf) == expected_n_params[segment_name]
        assert len(m[segment_name].utility_ca) == 210
    r = {}
    for segment_name in m:
        r[segment_name] = m[segment_name].maximize_loglike(
            method="SLSQP", options={"maxiter": 1000}
        )
    loglike_priors = [value for key, value in sorted(loglike_prior.items())]
    loglike_converge = [value.loglike for key, value in sorted(r.items())]
    num_regression.check(
        {"loglike_prior": loglike_priors, "loglike_converge": loglike_converge},
        basename="test_nonmand_tour_freq_loglike",
    )
    _regression_check(dataframe_regression, pd.concat([x.pf for x in m.values()]))


def test_nonmand_tour_freq_not_condensed(
    est_data, num_regression, dataframe_regression
):
    from activitysim.estimation.larch.nonmand_tour_freq import nonmand_tour_freq_model

    m = nonmand_tour_freq_model(condense_parameters=False)
    for segment_name in m:
        assert len(m[segment_name].pf) == 210
        assert len(m[segment_name].utility_ca) == 210
