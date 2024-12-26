from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import argparse
import importlib
import logging
import os
import sys
import warnings

import numpy as np

from activitysim.core import chunk, config, mem, tracing, workflow
from activitysim.core.configuration import FileSystem, Settings

logger = logging.getLogger(__name__)


INJECTABLES = [
    "data_dir",
    "configs_dir",
    "data_model_dir",
    "output_dir",
    "cache_dir",
    "settings_file_name",
    "imported_extensions",
]


def add_run_args(parser, multiprocess=True):
    """Run command args"""
    parser.add_argument(
        "-w",
        "--working_dir",
        type=str,
        metavar="PATH",
        help="path to example/project directory (default: %s)" % os.getcwd(),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        metavar="PATH",
        help="path to config dir",
    )
    parser.add_argument(
        "-o", "--output", type=str, metavar="PATH", help="path to output dir"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        action="append",
        metavar="PATH",
        help="path to data dir",
    )
    parser.add_argument(
        "--data_model",
        type=str,
        action="append",
        metavar="PATH",
        help="path to data model dir",
    )
    parser.add_argument(
        "-r", "--resume", type=str, metavar="STEPNAME", help="resume after step"
    )
    parser.add_argument(
        "-p", "--pipeline", type=str, metavar="FILE", help="pipeline file name"
    )
    parser.add_argument(
        "-s", "--settings_file", type=str, metavar="FILE", help="settings file name"
    )
    parser.add_argument(
        "-g", "--chunk_size", type=int, metavar="BYTES", help="chunk size"
    )
    parser.add_argument(
        "--chunk_training_mode",
        type=str,
        help="chunk training mode, one of [training, adaptive, production, disabled]",
    )
    parser.add_argument(
        "--households_sample_size", type=int, metavar="N", help="households sample size"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Do not limit process to one thread. "
        "Can make single process runs faster, "
        "but will cause thrashing on MP runs.",
    )
    parser.add_argument(
        "--persist-sharrow-cache",
        action="store_true",
        help="Store the sharrow cache in a persistent user cache directory.",
    )

    parser.add_argument(
        "-e",
        "--ext",
        type=str,
        action="append",
        metavar="PATH",
        help="Package of extension modules to load. Use of this option is not "
        "generally secure.",
    )

    if multiprocess:
        parser.add_argument(
            "-m",
            "--multiprocess",
            default=False,
            const=-1,
            metavar="(N)",
            nargs="?",
            type=int,
            help="run multiprocess. Adds configs_mp settings "
            "by default as the first config directory, but only if it is found"
            "and is not already explicitly included elsewhere in the list of "
            "configs. Optionally give a number of processes greater than 1, "
            "which will override the number of processes written in settings file.",
        )


def validate_injectable(state: workflow.State, name, make_if_missing=False):
    try:
        dir_paths = state.get(name)
    except RuntimeError:
        # injectable is missing, meaning is hasn't been explicitly set
        # and defaults cannot be found.
        sys.exit(
            f"Error({name}): please specify either a --working_dir "
            "containing 'configs', 'data', and 'output' folders "
            "or all three of --config, --data, and --output"
        )

    dir_paths = [dir_paths] if isinstance(dir_paths, str) else dir_paths

    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            if make_if_missing:
                os.makedirs(dir_path)
            else:
                sys.exit("Could not find %s '%s'" % (name, os.path.abspath(dir_path)))

    return dir_paths


def handle_standard_args(state: workflow.State, args, multiprocess=True):
    def inject_arg(name, value):
        assert name in INJECTABLES
        state.set(name, value)

    if args.working_dir:
        # activitysim will look in the current working directory for
        # 'configs', 'data', and 'output' folders by default
        os.chdir(args.working_dir)

    if args.ext:
        for e in args.ext:
            basepath, extpath = os.path.split(e)
            if not basepath:
                basepath = "."
            sys.path.insert(0, os.path.abspath(basepath))
            try:
                importlib.import_module(extpath)
            except ImportError as err:
                logger.exception("ImportError")
                raise
            except Exception as err:
                logger.exception(f"Error {err}")
                raise
            finally:
                del sys.path[0]
        inject_arg("imported_extensions", args.ext)
    else:
        inject_arg("imported_extensions", ())

    state.filesystem = FileSystem.parse_args(args)
    for config_dir in state.filesystem.get_configs_dir():
        if not config_dir.is_dir():
            print(f"missing config directory: {config_dir}", file=sys.stderr)
            raise NotADirectoryError(f"missing config directory: {config_dir}")
    for data_dir in state.filesystem.get_data_dir():
        if not data_dir.is_dir():
            print(f"missing data directory: {data_dir}", file=sys.stderr)
            raise NotADirectoryError(f"missing data directory: {data_dir}")

    try:
        state.load_settings()
    except Exception as err:
        logger.exception(f"Error {err} in loading settings")
        raise

    if args.multiprocess:
        if "configs_mp" not in state.filesystem.configs_dir:
            # when triggering multiprocessing from command arguments,
            # add 'configs_mp' as the first config directory, but only
            # if it exists, and it is not already explicitly included
            # in the set of config directories.
            if not state.filesystem.get_working_subdir("configs_mp").exists():
                logger.warning("could not find 'configs_mp'. skipping...")
            else:
                logger.info("adding 'configs_mp' to config_dir list...")
                state.filesystem.configs_dir = (
                    "configs_mp",
                ) + state.filesystem.configs_dir

        state.settings.multiprocess = True
        if args.multiprocess > 1:
            # setting --multiprocess to just 1 implies using the number of
            # processes discovered in the configs file, while setting to more
            # than 1 explicitly overrides that setting
            state.settings.num_processes = args.multiprocess

    if args.chunk_size:
        state.settings.chunk_size = int(args.chunk_size)
    if args.chunk_training_mode is not None:
        state.settings.chunk_training_mode = args.chunk_training_mode
    if args.households_sample_size is not None:
        state.settings.households_sample_size = args.households_sample_size

    if args.pipeline:
        state.filesystem.pipeline_file_name = args.pipeline

    if args.resume:
        state.settings.resume_after = args.resume

    if args.persist_sharrow_cache:
        state.filesystem.persist_sharrow_cache()

    return state


def cleanup_output_files(state: workflow.State):
    tracing.delete_trace_files(state)

    csv_ignore = []
    if state.settings.memory_profile:
        # memory profiling is opened potentially before `cleanup_output_files`
        # is called, but we want to leave any (newly created) memory profiling
        # log files that may have just been created.
        mem_prof_log = state.get_log_file_path("memory_profile.csv")
        csv_ignore.append(mem_prof_log)

    state.tracing.delete_output_files("h5")
    state.tracing.delete_output_files("csv", ignore=csv_ignore)
    state.tracing.delete_output_files("txt")
    state.tracing.delete_output_files("yaml")
    state.tracing.delete_output_files("prof")
    state.tracing.delete_output_files("omx")


def run(args):
    """
    Run the models. Specify a project folder using the '--working_dir' option,
    or point to the config, data, and output folders directly with
    '--config', '--data', and '--output'. Both '--config' and '--data' can be
    specified multiple times. Directories listed first take precedence.

    returns:
        int: sys.exit exit code
    """

    state = workflow.State()

    # register abm steps and other abm-specific injectables
    # by default, assume we are running activitysim.abm
    # other callers (e.g. populationsim) will have to arrange to register their own steps and injectables
    # (presumably) in a custom run_simulation.py instead of using the 'activitysim run' command
    if not "preload_injectables" in state:
        # register abm steps and other abm-specific injectables
        from activitysim import abm  # noqa: F401

    state.logging.config_logger(basic=True)
    state = handle_standard_args(state, args)  # possibly update injectables

    if state.settings.rotate_logs:
        state.logging.rotate_log_directory()

    if state.settings.memory_profile and not state.settings.multiprocess:
        # Memory sidecar is only useful for single process runs
        # multiprocess runs log memory usage without blocking in the controlling process.
        mem_prof_log = state.get_log_file_path("memory_profile.csv")
        from ..core.memory_sidecar import MemorySidecar

        memory_sidecar_process = MemorySidecar(mem_prof_log)
    else:
        memory_sidecar_process = None

    # legacy support for run_list setting nested 'models' and 'resume_after' settings
    # if state.settings.run_list:
    #     warnings.warn(
    #         "Support for 'run_list' settings group will be removed.\n"
    #         "The run_list.steps setting is renamed 'models'.\n"
    #         "The run_list.resume_after setting is renamed 'resume_after'.\n"
    #         "Specify both 'models' and 'resume_after' directly in settings config file.",
    #         FutureWarning,
    #     )
    #     run_list = state.settings.run_list
    #     if "steps" in run_list:
    #         assert not config.setting(
    #             "models"
    #         ), f"Don't expect 'steps' in run_list and 'models' as stand-alone setting!"
    #         config.override_setting("models", run_list["steps"])
    #
    #     if "resume_after" in run_list:
    #         assert not config.setting(
    #             "resume_after"
    #         ), f"Don't expect 'resume_after' both in run_list and as stand-alone setting!"
    #         config.override_setting("resume_after", run_list["resume_after"])

    # If you provide a resume_after argument to pipeline.run
    # the pipeline manager will attempt to load checkpointed tables from the checkpoint store
    # and resume pipeline processing on the next submodel step after the specified checkpoint
    resume_after = state.settings.resume_after

    # cleanup if not resuming
    if not resume_after:
        cleanup_output_files(state)
    elif state.settings.cleanup_trace_files_on_resume:
        tracing.delete_trace_files(state)

    state.logging.config_logger(
        basic=False
    )  # update using possibly new logging configs
    config.filter_warnings(state)
    logging.captureWarnings(capture=True)

    # directories
    for k in ["configs_dir", "settings_file_name", "data_dir", "output_dir"]:
        logger.info("SETTING %s: %s" % (k, getattr(state.filesystem, k, None)))

    log_settings = state.settings.log_settings
    for k in log_settings:
        logger.info("SETTING %s: %s" % (k, getattr(state.settings, k, None)))

    # OMP_NUM_THREADS: openmp
    # OPENBLAS_NUM_THREADS: openblas
    # MKL_NUM_THREADS: mkl
    for env in [
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ]:
        logger.info(f"ENV {env}: {os.getenv(env)}")

    np_info_keys = [
        "atlas_blas_info",
        "atlas_blas_threads_info",
        "atlas_info",
        "atlas_threads_info",
        "blas_info",
        "blas_mkl_info",
        "blas_opt_info",
        "lapack_info",
        "lapack_mkl_info",
        "lapack_opt_info",
        "mkl_info",
    ]

    for cfg_key in np_info_keys:
        try:
            info = np.__config__.get_info(cfg_key)
        except AttributeError:
            info = np.show_config("dicts").get(cfg_key, "MISSING")
        if info:
            for info_key in ["libraries"]:
                if info_key in info:
                    logger.info(f"NUMPY {cfg_key} {info_key}: {info[info_key]}")

    t0 = tracing.print_elapsed_time()

    try:
        if state.settings.multiprocess:
            logger.info("run multiprocess simulation")

            from activitysim.core import mp_tasks

            injectables = {}
            for k in INJECTABLES:
                try:
                    injectables[k] = state.get_injectable(k)
                except KeyError:
                    # if injectable is not set, just ignore it
                    pass
            injectables["settings"] = state.settings
            # injectables["settings_package"] = state.settings.dict()
            mp_tasks.run_multiprocess(state, injectables)

            if state.settings.cleanup_pipeline_after_run:
                state.checkpoint.cleanup()

        else:
            logger.info("run single process simulation")

            state.run(
                models=state.settings.models,
                resume_after=resume_after,
                memory_sidecar_process=memory_sidecar_process,
            )

            if state.settings.cleanup_pipeline_after_run:
                state.checkpoint.cleanup()  # has side effect of closing open pipeline
            else:
                state.checkpoint.close_store()

            mem.log_global_hwm()  # main process
    except Exception:
        # log time until error and the error traceback
        tracing.print_elapsed_time("all models until this error", t0)
        logger.exception("activitysim run encountered an unrecoverable error")
        raise

    chunk.consolidate_logs(state)
    mem.consolidate_logs(state)

    from activitysim.core.flow import TimeLogger

    # TimeLogger.aggregate_summary(logger)

    tracing.print_elapsed_time("all models", t0)

    if memory_sidecar_process:
        memory_sidecar_process.stop()

    return 0


if __name__ == "__main__":
    from activitysim import abm  # register injectables  # noqa: F401

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))
