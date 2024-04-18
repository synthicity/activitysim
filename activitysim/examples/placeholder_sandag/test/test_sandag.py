from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pkg_resources
import pytest

from activitysim.core import configuration, test, workflow


def example_path(dirname):
    resource = os.path.join("examples", "placeholder_sandag", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def mtc_example_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def psrc_example_path(dirname):
    resource = os.path.join("examples", "placeholder_psrc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def build_data():
    shutil.copy(
        example_path(os.path.join("data_3", "maz_to_maz_bike.csv")),
        example_path(os.path.join("data_2", "maz_to_maz_bike.csv")),
    )


@pytest.fixture(scope="module")
def data():
    build_data()


def run_test(zone, multiprocess=False, sharrow=False, recode=True):
    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress(zone):

        # ## regress tours
        if sharrow and os.path.isfile(
            test_path(f"regress/final_{zone}_zone_tours_sh.csv")
        ):
            regress_tours_df = pd.read_csv(
                test_path(f"regress/final_{zone}_zone_tours_sh.csv")
            )
        else:
            regress_tours_df = pd.read_csv(
                test_path(f"regress/final_{zone}_zone_tours.csv")
            )
        tours_df = pd.read_csv(test_path(f"output_{zone}/final_{zone}_zone_tours.csv"))
        tours_df.to_csv(
            test_path(f"regress/final_{zone}_zone_tours_last_run.csv"), index=False
        )
        print("regress tours")
        test.assert_frame_substantively_equal(
            tours_df, regress_tours_df, rtol=1e-03, check_dtype=False
        )

        # ## regress trips
        if sharrow and os.path.isfile(
            test_path(f"regress/final_{zone}_zone_trips_sh.csv")
        ):
            regress_trips_df = pd.read_csv(
                test_path(f"regress/final_{zone}_zone_trips_sh.csv")
            )
        else:
            regress_trips_df = pd.read_csv(
                test_path(f"regress/final_{zone}_zone_trips.csv")
            )
        trips_df = pd.read_csv(test_path(f"output_{zone}/final_{zone}_zone_trips.csv"))
        trips_df.to_csv(
            test_path(f"regress/final_{zone}_zone_trips_last_run.csv"), index=False
        )
        print("regress trips")
        test.assert_frame_substantively_equal(
            trips_df, regress_trips_df, rtol=1e-03, check_dtype=False
        )

        if zone == "2":
            # also test accessibility for the 2-zone system
            regress_accessibility_df = pd.read_csv(
                test_path(
                    f"regress/final_{zone}_zone_proto_disaggregate_accessibility.csv"
                )
            )
            final_accessibility_df = pd.read_csv(
                test_path(
                    f"output_{zone}/final_{zone}_zone_proto_disaggregate_accessibility.csv"
                )
            )
            final_accessibility_df = final_accessibility_df[
                [
                    c
                    for c in final_accessibility_df.columns
                    if not c.startswith("_original_")
                ]
            ]
            test.assert_frame_substantively_equal(
                final_accessibility_df,
                regress_accessibility_df,
                check_dtype=False,
            )

    # run test
    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    if zone == "2":
        base_configs = psrc_example_path("configs")
    else:
        base_configs = mtc_example_path("configs")

    run_args = [
        "-c",
        test_path(f"configs_{zone}_zone"),
        "-c",
        example_path(f"configs_{zone}_zone"),
        "-c",
        base_configs,
        "-d",
        example_path(f"data_{zone}"),
        "-o",
        test_path(f"output_{zone}"),
    ]

    if multiprocess:
        run_args = run_args + ["-s", "settings_mp.yaml"]
    elif not recode:
        run_args = run_args + ["-s", "settings_no_recode.yaml"]

    if sharrow:
        run_args = ["-c", test_path(f"configs_{zone}_sharrow")] + run_args

    try:
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    except FileNotFoundError:
        subprocess.run([sys.executable, file_path] + run_args, check=True)
        from tempfile import TemporaryFile
        from time import sleep

        with TemporaryFile() as outputstream:
            env = os.environ.copy()
            pythonpath = env.pop("PYTHONPATH", None)
            process = subprocess.Popen(
                args=[sys.executable, file_path] + run_args,
                shell=True,
                stdout=outputstream,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env=env,
            )
            while process.poll() is None:
                where = outputstream.tell()
                lines = outputstream.read()
                if not lines:
                    # Adjust the sleep interval to your needs
                    sleep(0.25)
                    # make sure pointing to the last place we read
                    outputstream.seek(where)
                else:
                    # Windows adds an extra carriage return and then chokes on
                    # it when displaying (or, as it were, not displaying) the
                    # output.  So we give Windows a little helping hand.
                    print(lines.decode().replace("\r\n", "\n"), end="")

    regress(zone)


def test_1_zone(data):
    run_test(zone="1", multiprocess=False)


def test_1_zone_mp(data):
    run_test(zone="1", multiprocess=True)


def test_1_zone_sharrow(data):
    # Run both single and MP in one test function
    # guarantees that compile happens in single
    run_test(zone="1", multiprocess=False, sharrow=True)
    run_test(zone="1", multiprocess=True, sharrow=True)


def test_2_zone(data):
    run_test(zone="2", multiprocess=False)


def test_2_zone_local_compute(data):
    def _test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    import activitysim.abm  # register components # noqa: F401

    state = workflow.State.make_default(
        data_dir=example_path("data_2"),
        configs_dir=(
            _test_path("configs_2_zone"),
            example_path("configs_2_zone"),
            psrc_example_path("configs"),
        ),
        output_dir=_test_path("output_2"),
    )
    state.run.all(resume_after=None)
    # ## regress tours
    regress_tours_df = pd.read_csv(_test_path(f"regress/final_2_zone_tours.csv"))
    tours_df = pd.read_csv(_test_path(f"output_2/final_2_zone_tours.csv"))
    tours_df.to_csv(
        _test_path(f"regress/final_2_zone_tours_last_run_localcompute.csv"), index=False
    )
    test.assert_frame_substantively_equal(
        tours_df, regress_tours_df, rtol=1e-03, check_dtype=False
    )

    # ## regress trips
    regress_trips_df = pd.read_csv(_test_path(f"regress/final_2_zone_trips.csv"))
    trips_df = pd.read_csv(_test_path(f"output_2/final_2_zone_trips.csv"))
    trips_df.to_csv(_test_path(f"regress/final_2_zone_trips_last_run.csv"), index=False)
    test.assert_frame_substantively_equal(
        trips_df, regress_trips_df, rtol=1e-03, check_dtype=False
    )

    # also test accessibility for the 2-zone system
    regress_accessibility_df = pd.read_csv(
        _test_path(f"regress/final_2_zone_proto_disaggregate_accessibility.csv")
    )
    final_accessibility_df = pd.read_csv(
        _test_path(f"output_2/final_2_zone_proto_disaggregate_accessibility.csv")
    )
    final_accessibility_df = final_accessibility_df[
        [c for c in final_accessibility_df.columns if not c.startswith("_original_")]
    ]
    test.assert_frame_substantively_equal(
        final_accessibility_df,
        regress_accessibility_df,
        check_dtype=False,
    )


def test_2_zone_norecode(data):
    run_test(zone="2", multiprocess=False, recode=False)


def test_2_zone_mp(data):
    run_test(zone="2", multiprocess=True)


def test_2_zone_sharrow(data):
    # Run both single and MP in one test function
    # guarantees that compile happens in single
    run_test(zone="2", multiprocess=False, sharrow=True)
    run_test(zone="2", multiprocess=True, sharrow=True)


def test_3_zone(data):
    run_test(zone="3", multiprocess=False)


def test_3_zone_mp(data):
    run_test(zone="3", multiprocess=True)


def test_3_zone_sharrow(data):
    # Run both single and MP in one test function
    # guarantees that compile happens in single
    run_test(zone="3", multiprocess=False, sharrow=True)
    run_test(zone="3", multiprocess=True, sharrow=True)


EXPECTED_MODELS_3_ZONE = [
    "initialize_landuse",
    "initialize_households",
    "compute_accessibility",
    "initialize_los",
    "initialize_tvpb",
    "school_location",
    "workplace_location",
    "auto_ownership_simulate",
    "free_parking",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
    "joint_tour_frequency",
    "joint_tour_composition",
    "joint_tour_participation",
    "joint_tour_destination",
    "joint_tour_scheduling",
    "non_mandatory_tour_frequency",
    "non_mandatory_tour_destination",
    "non_mandatory_tour_scheduling",
    "tour_mode_choice_simulate",
    "atwork_subtour_frequency",
    "atwork_subtour_destination",
    "atwork_subtour_scheduling",
    "atwork_subtour_mode_choice",
    "stop_frequency",
    "trip_purpose",
    "trip_destination",
    "trip_purpose_and_destination",
    "trip_scheduling",
    "trip_mode_choice",
    "write_data_dictionary",
    "track_skim_usage",
    "write_trip_matrices",
    "write_tables",
]


@test.run_if_exists("placeholder_sandag_3_zone_reference_pipeline.zip")
def test_3_zone_progressive():
    import activitysim.abm  # register components

    state = workflow.create_example(
        "placeholder_sandag_3_zone", directory="/tmp/placeholder_sandag_3_zone"
    )

    assert state.settings.models == EXPECTED_MODELS_3_ZONE
    assert state.settings.chunk_size == 0
    assert state.settings.sharrow == False

    state.settings.recode_pipeline_columns = True
    state.settings.treat_warnings_as_errors = False
    state.settings.households_sample_size = 30
    state.settings.use_shadow_pricing = False
    state.settings.want_dest_choice_sample_tables = False
    state.settings.want_dest_choice_presampling = True
    state.settings.cleanup_pipeline_after_run = True
    state.settings.output_tables = configuration.OutputTables(
        h5_store=False,
        action="include",
        prefix="final_3_zone_",
        sort=True,
        tables=["trips", "tours"],
    )
    from activitysim.abm.tables.skims import network_los_preload

    state.get(network_los_preload)
    state.network_settings.read_skim_cache = False
    state.network_settings.write_skim_cache = False
    state.network_settings.rebuild_tvpb_cache = False

    for step_name in EXPECTED_MODELS_3_ZONE:
        state.run.by_name(step_name)
        try:
            state.checkpoint.check_against(
                Path(__file__).parent.joinpath(
                    "placeholder_sandag_3_zone_reference_pipeline.zip"
                ),
                checkpoint_name=step_name,
            )
        except Exception:
            print(f"> placeholder_sandag_3_zone {step_name}: ERROR")
            raise
        else:
            print(f"> placeholder_sandag_3_zone {step_name}: ok")


if __name__ == "__main__":

    # call each test explicitly so we get a pass/fail for each
    build_data()
    run_test(zone="1", multiprocess=False)
    run_test(zone="1", multiprocess=True)
    run_test(zone="1", multiprocess=False, sharrow=True)
    run_test(zone="1", multiprocess=True, sharrow=True)

    run_test(zone="2", multiprocess=False)
    run_test(zone="2", multiprocess=True)
    run_test(zone="2", multiprocess=False, sharrow=True)
    run_test(zone="2", multiprocess=True, sharrow=True)

    run_test(zone="3", multiprocess=False)
    run_test(zone="3", multiprocess=True)
    run_test(zone="3", multiprocess=False, sharrow=True)
    run_test(zone="3", multiprocess=True, sharrow=True)
