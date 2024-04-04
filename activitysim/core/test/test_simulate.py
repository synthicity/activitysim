# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os.path
import warnings

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import logit, simulate, workflow
from activitysim.core.chunk import ChunkSizer
from activitysim.core.config import get_logit_model_settings
from activitysim.core.configuration.logit import LogitComponentSettings


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def spec_name(data_dir):
    return "sample_spec.csv"


@pytest.fixture
def state(data_dir) -> workflow.State:
    state = workflow.State()
    state.initialize_filesystem(
        working_dir=os.path.dirname(__file__), data_dir=(data_dir,)
    ).default_settings()
    return state


@pytest.fixture
def spec(state, spec_name):
    return state.filesystem.read_model_spec(file_name=spec_name)


@pytest.fixture
def data(data_dir):
    return pd.read_csv(os.path.join(data_dir, "data.csv"))


def test_read_model_spec(state, spec_name):
    spec = state.filesystem.read_model_spec(file_name=spec_name)

    assert len(spec) == 4
    assert spec.index.name == "Expression"
    assert list(spec.columns) == ["alt0", "alt1"]
    npt.assert_array_equal(spec.values, [[1.1, 11], [2.2, 22], [3.3, 33], [4.4, 44]])


def test_eval_variables(state, spec, data):
    result = simulate.eval_variables(state, spec.index, data)

    expected = pd.DataFrame(
        [[1, 0, 4, 1], [0, 1, 4, 1], [0, 1, 5, 1]], index=data.index, columns=spec.index
    )

    expected[expected.columns[0]] = expected[expected.columns[0]].astype(np.int8)
    expected[expected.columns[1]] = expected[expected.columns[1]].astype(np.int8)
    expected[expected.columns[2]] = expected[expected.columns[2]].astype(np.int64)
    expected[expected.columns[3]] = expected[expected.columns[3]].astype(int)

    print("\nexpected\n%s" % expected.dtypes)
    print("\nresult\n%s" % result.dtypes)

    pdt.assert_frame_equal(result, expected, check_names=False)


def test_simple_simulate(state, data, spec):
    state.settings.check_for_variability = False

    choices = simulate.simple_simulate(state, choosers=data, spec=spec, nest_spec=None)
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected, check_dtype=False)


def test_simple_simulate_chunked(state, data, spec):
    state.settings.check_for_variability = False
    state.settings.chunk_size = 2
    choices = simulate.simple_simulate(
        state,
        choosers=data,
        spec=spec,
        nest_spec=None,
    )
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected, check_dtype=False)


def test_eval_utilities(state, data, spec):
    locals_d = {}
    log_alt_losers = False
    have_trace_targets = False
    trace_label = "test_eval_utilities"
    estimator = None
    trace_column_names = []
    spec_sh = None

    chunk_sizer = ChunkSizer(
        state, "chunkless", trace_label, 0, 0, state.settings.chunk_training_mode
    )
    raw_utilities = simulate.eval_utilities(
        state,
        spec,
        data,
        locals_d,
        log_alt_losers=log_alt_losers,
        trace_label=trace_label,
        have_trace_targets=have_trace_targets,
        estimator=estimator,
        trace_column_names=trace_column_names,
        spec_sh=spec_sh,
        chunk_sizer=chunk_sizer,
    )
    assert raw_utilities.to_numpy() == pytest.approx(
        np.asarray(
            [
                (18.7, 187.0),
                (19.8, 198.0),
                (23.1, 231.0),
            ]
        )
    )


@pytest.fixture
def eval_nl_setup(state):
    data = pd.DataFrame(
        {
            "var0": [0, 0, 0, 0, 0, 0, 5, 0],
            "var1": [1, 0, 0, 0, 0, 0, 5, 0],
            "var2": [0, 1, 1, 0, 2, 0, 5, 0],
            "var3": [4, -4, 5, 0, -1, 0, 5, 0],
            "var4": [1, 1, 1, 0, 0, 0, 5, 0],
            "var5": [0, 0, 0, 0, 1, 1, 1, 0],
            "var6": [0, 0, 0, 0, 0, 0, 0, 1.1],
            "var7": [0, 0, 0, 0, 0, 0, 0, -1.1],
        },
        index=pd.Index(
            [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008], name="chooser_id"
        ),
    )

    spec = pd.DataFrame(
        {
            "DRIVE": [1, 0, 0, 0, 0, -999.9, 2.0, 0],
            "WALK_TO_TRANSIT": [0, 1, 0, 0, 0, -999.9, 2.0, 0],
            "DRIVE_TO_TRANSIT": [0, 0, 1, 0, 0, -999.9, 2.0, 0],
            "WALK": [0, 0, 0, 1, 0, -999.9, 0, 2.0],
            "BIKE": [0, 0, 0, 0, 1, -999.9, 0, 2.0],
        },
        index=pd.Index(
            ["var0", "var1", "var2", "var3", "var4", "var5", "var6", "var7"],
            name="Expression",
        ),
    )

    component_settings = LogitComponentSettings.model_validate(
        {
            "LOGIT_TYPE": "NL",
            "SPEC": "not-needed.csv",
            "NESTS": {
                "name": "root",
                "coefficient": 1.0,
                "alternatives": [
                    {
                        "name": "MOTORIZED",
                        "coefficient": 0.5,
                        "alternatives": [
                            "DRIVE",
                            {
                                "name": "TRANSIT",
                                "coefficient": 0.25,
                                "alternatives": [
                                    "WALK_TO_TRANSIT",
                                    "DRIVE_TO_TRANSIT",
                                ],
                            },
                        ],
                    },
                    {
                        "name": "NONMOTORIZED",
                        "coefficient": 0.5,
                        "alternatives": [
                            "WALK",
                            "BIKE",
                        ],
                    },
                ],
            },
        }
    )

    return data, spec, component_settings


@pytest.mark.parametrize("overflow_protection", [True, False])
def test_eval_nl(state, overflow_protection, eval_nl_setup):

    locals_d = {}
    log_alt_losers = False
    have_trace_targets = False
    trace_label = "test_eval_utilities"
    estimator = None
    trace_column_names = []
    spec_sh = None
    data, spec, component_settings = eval_nl_setup
    nest_spec = get_logit_model_settings(component_settings)
    chunk_sizer = ChunkSizer(
        state, "chunkless", trace_label, 0, 0, state.settings.chunk_training_mode
    )

    if overflow_protection:
        with pytest.warns(
            RuntimeWarning, match="overflow protection is making 3 choices"
        ):
            simulate.eval_nl(
                state,
                data,
                spec,
                nest_spec,
                locals_d,
                custom_chooser=None,
                estimator=None,
                log_alt_losers=False,
                want_logsums=False,
                trace_label="None",
                trace_choice_name=None,
                trace_column_names=None,
                chunk_sizer=chunk_sizer,
                overflow_protection=overflow_protection,
            )
    else:
        with pytest.raises(RuntimeError):
            simulate.eval_nl(
                state,
                data,
                spec,
                nest_spec,
                locals_d,
                custom_chooser=None,
                estimator=None,
                log_alt_losers=False,
                want_logsums=False,
                trace_label="None",
                trace_choice_name=None,
                trace_column_names=None,
                chunk_sizer=chunk_sizer,
                overflow_protection=overflow_protection,
            )


@pytest.mark.parametrize("overflow_protection", [True, False])
def test_simple_simulate_with_nest_spec(state, overflow_protection, eval_nl_setup):
    state.settings.check_for_variability = False

    data, spec, component_settings = eval_nl_setup

    nest_spec = get_logit_model_settings(component_settings)
    trace_label = "test_simple_simulate_with_nest_spec"
    logit.validate_nest_spec(nest_spec, trace_label)
    locals_d = {}
    chunk_sizer = ChunkSizer(
        state, "chunkless", trace_label, 0, 0, state.settings.chunk_training_mode
    )

    raw_utilities = simulate.eval_utilities(
        state,
        spec,
        data,
        locals_d,
        log_alt_losers=False,
        trace_label=trace_label,
        have_trace_targets=False,
        estimator=None,
        trace_column_names=[],
        spec_sh=None,
        chunk_sizer=chunk_sizer,
    )
    expected_utilities = {
        "DRIVE": {
            1001: 0.0,
            1002: 0.0,
            1003: 0.0,
            1004: 0.0,
            1005: -999.9,
            1006: -999.9,
            1007: -994.9,
            1008: 2.2,
        },
        "WALK_TO_TRANSIT": {
            1001: 1.0,
            1002: 0.0,
            1003: 0.0,
            1004: 0.0,
            1005: -999.9,
            1006: -999.9,
            1007: -994.9,
            1008: 2.2,
        },
        "DRIVE_TO_TRANSIT": {
            1001: 0.0,
            1002: 1.0,
            1003: 1.0,
            1004: 0.0,
            1005: -997.9,
            1006: -999.9,
            1007: -994.9,
            1008: 2.2,
        },
        "WALK": {
            1001: 4.0,
            1002: -4.0,
            1003: 5.0,
            1004: 0.0,
            1005: -1000.9,
            1006: -999.9,
            1007: -994.9,
            1008: -2.2,
        },
        "BIKE": {
            1001: 1.0,
            1002: 1.0,
            1003: 1.0,
            1004: 0.0,
            1005: -999.9,
            1006: -999.9,
            1007: -994.9,
            1008: -2.2,
        },
    }
    pd.testing.assert_frame_equal(
        raw_utilities,
        pd.DataFrame(expected_utilities),
        check_names=False,
    )

    if overflow_protection:
        # exponentiated utils may overflow, downshift them
        shifts = raw_utilities.to_numpy().max(1, keepdims=True)
        if shifts.min() < -85:
            warnings.warn(
                "overflow protection is making choices for choosers "
                "who appear to have no valid alternatives"
            )
        raw_utilities -= shifts
    else:
        shifts = None

    nested_exp_utilities = simulate.compute_nested_exp_utilities(
        raw_utilities, nest_spec
    )
    expected_nested_exp_utilities = {
        "DRIVE": {
            1001: 1.0,
            1002: 1.0,
            1003: 1.0,
            1004: 1.0,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 81.45086866496814,
        },
        "WALK_TO_TRANSIT": {
            1001: 2980.9579870417283,
            1002: 1.0,
            1003: 1.0,
            1004: 1.0,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 44013193.53483411,
        },
        "DRIVE_TO_TRANSIT": {
            1001: 1.0,
            1002: 2980.9579870417283,
            1003: 2980.9579870417283,
            1004: 1.0,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 44013193.53483411,
        },
        "TRANSIT": {
            1001: 7.389675709034251,
            1002: 7.389675709034251,
            1003: 7.389675709034251,
            1004: 1.189207115002721,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 96.86195253953227,
        },
        "MOTORIZED": {
            1001: 2.896493692213786,
            1002: 2.896493692213786,
            1003: 2.896493692213786,
            1004: 1.4795969434284193,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 13.353382388162945,
        },
        "WALK": {
            1001: 2980.9579870417283,
            1002: 0.00033546262790251185,
            1003: 22026.465794806718,
            1004: 1.0,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 0.012277339903068436,
        },
        "BIKE": {
            1001: 7.38905609893065,
            1002: 7.38905609893065,
            1003: 7.38905609893065,
            1004: 1.0,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 0.012277339903068436,
        },
        "NONMOTORIZED": {
            1001: 54.66577579382422,
            1002: 2.718343532660755,
            1003: 148.43805054939804,
            1004: 1.414213562373095,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 0.1566993293097864,
        },
        "root": {
            1001: 57.562269486038,
            1002: 5.614837224874541,
            1003: 151.33454424161187,
            1004: 2.893810505801514,
            1005: 0.0,
            1006: 0,
            1007: 0,
            1008: 13.510081717472728,
        },
    }
    if not overflow_protection:
        pd.testing.assert_frame_equal(
            nested_exp_utilities,
            pd.DataFrame(expected_nested_exp_utilities),
            check_names=False,
        )

    nested_probabilities = simulate.compute_nested_probabilities(
        state, nested_exp_utilities, nest_spec, trace_label=trace_label
    )
    print("\n\n==== nested_probabilities\n")
    print(nested_probabilities.to_dict())
    expected_nested_probs = {
        "MOTORIZED": {
            1001: 0.05031931016751075,
            1002: 0.515864231892939,
            1003: 0.019139673012061372,
            1004: 0.5112971082460728,
            1005: 0.8749672738657182 if overflow_protection else 0.0,
            1006: 0.5112971082460728 if overflow_protection else 0.0,
            1007: 0.5112971082460728 if overflow_protection else 0.0,
            1008: 0.9884013041085367,
        },
        "NONMOTORIZED": {
            1001: 0.9496806898324892,
            1002: 0.48413576810706105,
            1003: 0.9808603269879386,
            1004: 0.4887028917539273,
            1005: 0.12503272613428196 if overflow_protection else 0.0,
            1006: 0.4887028917539273 if overflow_protection else 0.0,
            1007: 0.4887028917539273 if overflow_protection else 0.0,
            1008: 0.011598695891463442,
        },
        "DRIVE": {
            1001: 0.11919411842381113,
            1002: 0.11919411842381113,
            1003: 0.11919411842381113,
            1004: 0.45678638313705516,
            1005: 0.01798620946517266 if overflow_protection else 0.0,
            1006: 0.45678638313705516 if overflow_protection else 0.0,
            1007: 0.45678638313705516 if overflow_protection else 0.0,
            1008: 0.45678638313705516,
        },
        "TRANSIT": {
            1001: 0.880805881576189,
            1002: 0.880805881576189,
            1003: 0.880805881576189,
            1004: 0.5432136168629449,
            1005: 0.9820137905348273 if overflow_protection else 0.0,
            1006: 0.5432136168629449 if overflow_protection else 0.0,
            1007: 0.5432136168629449 if overflow_protection else 0.0,
            1008: 0.5432136168629449,
        },
        "WALK_TO_TRANSIT": {
            1001: 0.9996646498695335,
            1002: 0.0003353501304664781,
            1003: 0.0003353501304664781,
            1004: 0.5,
            1005: 1.12535162055095e-07 if overflow_protection else 0.0,
            1006: 0.5 if overflow_protection else 0.0,
            1007: 0.5 if overflow_protection else 0.0,
            1008: 0.5,
        },
        "DRIVE_TO_TRANSIT": {
            1001: 0.0003353501304664781,
            1002: 0.9996646498695335,
            1003: 0.9996646498695335,
            1004: 0.5,
            1005: 0.9999998874648379 if overflow_protection else 0.0,
            1006: 0.5 if overflow_protection else 0.0,
            1007: 0.5 if overflow_protection else 0.0,
            1008: 0.5,
        },
        "WALK": {
            1001: 0.9975273768433652,
            1002: 4.5397868702434395e-05,
            1003: 0.9996646498695335,
            1004: 0.5,
            1005: 0.11920292202211756 if overflow_protection else 0.0,
            1006: 0.5 if overflow_protection else 0.0,
            1007: 0.5 if overflow_protection else 0.0,
            1008: 0.5,
        },
        "BIKE": {
            1001: 0.0024726231566347743,
            1002: 0.9999546021312975,
            1003: 0.0003353501304664781,
            1004: 0.5,
            1005: 0.8807970779778824 if overflow_protection else 0.0,
            1006: 0.5 if overflow_protection else 0.0,
            1007: 0.5 if overflow_protection else 0.0,
            1008: 0.5,
        },
    }
    pd.testing.assert_frame_equal(
        nested_probabilities,
        pd.DataFrame(expected_nested_probs),
        check_names=False,
    )

    # logsum of nest root
    logsums = np.log(nested_exp_utilities.root)
    if shifts is not None:
        logsums += np.squeeze(shifts, 1)
    logsums = pd.Series(logsums, index=data.index)
    assert logsums.to_dict() == pytest.approx(
        {
            1001: 4.052867309421848,
            1002: 1.7254125984335273,
            1003: 5.01949291093778,
            1004: 1.062574147766087,
            1005: -997.7573562276069 if overflow_protection else -np.inf,
            1006: -998.8374258522339 if overflow_protection else -np.inf,
            1007: -993.8374258522339 if overflow_protection else -np.inf,
            1008: 2.60343620061945,
        }
    )

    base_probabilities = simulate.compute_base_probabilities(
        nested_probabilities, nest_spec, spec
    )
    expected_base_probs = {
        "DRIVE": {
            1001: 0.00599776581511076,
            1002: 0.06148798234685533,
            1003: 0.002281336451592665,
            1004: 0.23355355678415896,
            1005: 0.0157373446629199 if overflow_protection else 0.0,
            1006: 0.23355355678415896 if overflow_protection else 0.0,
            1007: 0.23355355678415896 if overflow_protection else 0.0,
            1008: 0.451488256791687,
        },
        "WALK_TO_TRANSIT": {
            1001: 0.04430668111671894,
            1002: 0.00015237513456614819,
            1003: 5.65344536500098e-06,
            1004: 0.1388717757309569,
            1005: 9.669357932542469e-08 if overflow_protection else 0.0,
            1006: 0.1388717757309569 if overflow_protection else 0.0,
            1007: 0.1388717757309569 if overflow_protection else 0.0,
            1008: 0.2684565236584249,
        },
        "DRIVE_TO_TRANSIT": {
            1001: 1.4863235681053134e-05,
            1002: 0.4542238744115175,
            1003: 0.01685268311510371,
            1004: 0.1388717757309569,
            1005: 0.8592298325092188 if overflow_protection else 0.0,
            1006: 0.1388717757309569 if overflow_protection else 0.0,
            1007: 0.1388717757309569 if overflow_protection else 0.0,
            1008: 0.2684565236584249,
        },
        "WALK": {
            1001: 0.9473324873674005,
            1002: 2.197873203467658e-05,
            1003: 0.9805313953493138,
            1004: 0.24435144587696364,
            1005: 0.014904266303597593 if overflow_protection else 0.0,
            1006: 0.24435144587696364 if overflow_protection else 0.0,
            1007: 0.24435144587696364 if overflow_protection else 0.0,
            1008: 0.005799347945731721,
        },
        "BIKE": {
            1001: 0.0023482024650886995,
            1002: 0.48411378937502636,
            1003: 0.0003289316386247976,
            1004: 0.24435144587696364,
            1005: 0.11012845983068437 if overflow_protection else 0.0,
            1006: 0.24435144587696364 if overflow_protection else 0.0,
            1007: 0.24435144587696364 if overflow_protection else 0.0,
            1008: 0.005799347945731721,
        },
    }
    pd.testing.assert_frame_equal(
        base_probabilities,
        pd.DataFrame(expected_base_probs),
        check_names=False,
    )

    BAD_PROB_THRESHOLD = 0.001
    no_choices = (base_probabilities.sum(axis=1) - 1).abs() > BAD_PROB_THRESHOLD

    if not overflow_protection:
        assert no_choices.sum() == 3
    else:
        assert no_choices.sum() == 0
