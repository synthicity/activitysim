from __future__ import annotations

import glob
import logging
import os
import struct
import time
from pathlib import Path
from typing import Any

import numba
import pandas as pd
import platformdirs
import yaml
from pydantic import DirectoryPath, validator

from activitysim.core.configuration.base import PydanticBase
from activitysim.core.configuration.logit import LogitComponentSettings
from activitysim.core.exceptions import SettingsFileNotFoundError
from activitysim.core.util import parse_suffix_args, suffix_tables_in_settings

logger = logging.getLogger(__name__)


class FileSystem(PydanticBase, validate_assignment=True):
    """
    Manage finding and loading files for ActivitySim's command line interface.
    """

    working_dir: DirectoryPath = None
    """
    Name of the working directory.

    All other directories (configs, data, output, cache), when given as relative
    paths, are assumed to be relative to this working directory. If it is not
    provided, the usual Python current working directory is used.
    """

    configs_dir: tuple[Path, ...] = ("configs",)
    """
    Name[s] of the config directory.
    """

    @validator("configs_dir")
    def configs_dirs_must_exist(cls, configs_dir, values):
        working_dir = values.get("working_dir", None) or Path.cwd()
        for c in configs_dir:
            c_full = working_dir.joinpath(c)
            if not c_full.exists():
                raise ValueError(f"config directory {c_full} does not exist")
        return configs_dir

    data_dir: tuple[Path, ...] = ("data",)
    """
    Name of the data directory.
    """

    @validator("data_dir")
    def data_dirs_must_exist(cls, data_dir, values):
        working_dir = values.get("working_dir", None) or Path.cwd()
        for d in data_dir:
            d_full = working_dir.joinpath(d)
            if not d_full.exists():
                raise ValueError(f"data directory {d_full} does not exist")
        return data_dir

    data_model_dir: tuple[Path, ...] = ("data_model",)
    """
    Name of the data model directory.
    """

    @validator("data_model_dir")
    def data_model_dirs_must_exist(cls, data_model_dir, values):
        working_dir = values.get("working_dir", None) or Path.cwd()
        for d in data_model_dir:
            d_full = working_dir.joinpath(d)
            if not d_full.exists():
                raise ValueError(f"data model directory {d_full} does not exist")
        return data_model_dir

    output_dir: Path = "output"
    """
    Name of the output directory.

    This directory will be created on access if it does not exist.
    """

    profile_dir: Path = None
    """
    Name of the output directory for pyinstrument profiling files.

    If not given, a unique time-stamped directory will be created inside
    the usual output directory.
    """

    cache_dir: Path = None
    """
    Name of the output directory for general cache files.

    If not given, a directory named "cache" will be created inside
    the usual output directory.
    """

    sharrow_cache_dir: Path = None
    """
    Name of the output directory for sharrow cache files.

    If not given, the sharrow cache is stored in a run-independent persistent
    location, according to `platformdirs.user_cache_dir`.  See `persist_sharrow_cache`.
    """

    settings_file_name: str = "settings.yaml"

    pipeline_file_name: str = "pipeline"
    """
    The name for the base pipeline file or directory.
    """

    @classmethod
    def parse_args(cls, args):
        self = cls()

        def _parse_arg(name, x):
            v = getattr(args, x, None)
            if v is not None:
                setattr(self, name, v)

        _parse_arg("working_dir", "working_dir")
        _parse_arg("settings_file_name", "settings_file")
        _parse_arg("configs_dir", "config")
        _parse_arg("data_dir", "data")
        _parse_arg("data_model_dir", "data_model")
        _parse_arg("output_dir", "output")

        return self

    def parse_settings(self, settings):
        def _parse_setting(name, x):
            v = getattr(settings, x, None)
            if v is not None:
                setattr(self, name, v)

        _parse_setting("cache_dir", "cache_dir")
        _parse_setting("sharrow_cache_dir", "sharrow_cache_dir")
        _parse_setting("profile_dir", "profile_dir")
        _parse_setting("pipeline_file_name", "pipeline_file_name")
        return

    def get_working_subdir(self, subdir) -> Path:
        if self.working_dir:
            return self.working_dir.joinpath(subdir)
        else:
            return Path(subdir)

    def get_output_dir(self, subdir=None) -> Path:
        """
        Get an output directory, creating it if needed.

        Parameters
        ----------
        subdir : Path-like, optional
            If given, get this subdirectory of the output_dir.

        Returns
        -------
        Path
        """
        out = self.get_working_subdir(self.output_dir)
        if subdir is not None:
            out = out.joinpath(subdir)
        if not out.exists():
            out.mkdir(parents=True, exist_ok=True)
            # we set exist_ok=True so we avoid multiprocess race conditions
        return out

    def get_output_file_path(self, file_name) -> Path:
        return self.get_output_dir().joinpath(file_name)

    def get_pipeline_filepath(self) -> Path:
        """
        Get the complete path to the pipeline file or directory.

        Returns
        -------
        Path
        """
        return self.get_output_dir().joinpath(self.pipeline_file_name)

    def get_profiling_file_path(self, file_name) -> Path:
        """
        Get the complete path to a profile output file.

        Parameters
        ----------
        file_name : str
            Base name of the profiling output file.

        Returns
        -------
        Path
        """
        if self.profile_dir is None:
            profile_dir = self.get_output_dir(
                time.strftime("profiling--%Y-%m-%d--%H-%M-%S")
            )
            profile_dir.mkdir(parents=True, exist_ok=True)
            self.profile_dir = profile_dir
        return self.profile_dir.joinpath(file_name)

    def get_log_file_path(self, file_name) -> Path:
        """
        Get the complete path to a log file.

        Parameters
        ----------
        file_name : str
            Base name of the log file.

        Returns
        -------
        Path
        """

        output_dir = self.get_output_dir()

        # - check if running asv and if so, log to commit-specific subfolder
        asv_commit = os.environ.get("ASV_COMMIT", None)
        if asv_commit:
            output_dir = os.path.join(output_dir, f"log-{asv_commit}")
            os.makedirs(output_dir, exist_ok=True)

        # - check for optional log subfolder
        if os.path.exists(os.path.join(output_dir, "log")):
            output_dir = os.path.join(output_dir, "log")

        file_path = os.path.join(output_dir, file_name)

        return Path(file_path)

    def get_trace_file_path(
        self, file_name, tail=None, trace_dir=None, create_dirs=True, file_type=None
    ):
        """
        Get the complete path to a trace file.

        Parameters
        ----------
        file_name : str
            Base name of the trace file.
        tail : str or False, optional
            Add this suffix to filenames.  If not given, a quasi-random short
            string is derived from the current time.  Set to `False` to omit
            the suffix entirely.  Having a unique suffix makes it easier to
            open multiple comparable trace files side-by-side in Excel, which
            doesn't allow identically named files to be open simultaneously.
            Omitting the suffix can be valuable for using automated tools to
            find file differences across many files simultaneously.
        trace_dir : path-like, optional
            Construct the trace file path within this directory.  If not
            provided (typically for normal operation) the "trace" sub-directory
            of the normal output directory given by `get_output_dir` is used.
            The option to give a different location is primarily used to
            conduct trace file validation testing.
        create_dirs : bool, default True
            If the path to the containing directory of the trace file does not
            yet exist, create it.
        file_type : str, optional
            If provided, ensure that the generated file path has this extension.

        Returns
        -------
        Path
        """
        if trace_dir is None:
            output_dir = self.get_output_dir()

            # - check for trace subfolder, create it if missing
            trace_dir = output_dir.joinpath("trace")
            if not trace_dir.exists():
                trace_dir.mkdir(parents=True)

        if tail is None:
            # construct a unique tail string from the time
            # this is a convenience for opening multiple similarly named trace files
            tail = (
                "-"
                + hex(struct.unpack("<Q", struct.pack("<d", time.time()))[0])[
                    -6:
                ].lower()
            )
        elif not tail:
            tail = ""
        elif not tail.startswith("-"):
            tail = f"-{tail}"

        if file_type is not None:
            if not file_name.endswith(f".{file_type}"):
                file_name = f"{file_name}.{file_type}"

        file_parts = str(file_name).split(".")

        file_path = (
            os.path.join(trace_dir, *file_parts[:-1]) + f"{tail}.{file_parts[-1]}"
        )
        if create_dirs:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return Path(file_path)

    def find_trace_file_path(
        self, file_name, trace_dir=None, return_all=False, file_type=None
    ):
        """
        Find the complete path to one or more existing trace file(s).

        Parameters
        ----------
        file_name : str
            Base name of the trace file.
        trace_dir : path-like, optional
            Construct the trace file path within this directory.  If not
            provided (typically for normal operation) the "trace" sub-directory
            of the normal output directory given by `get_output_dir` is used.
            The option to give a different location is primarily used to
            conduct trace file validation testing.
        return_all : bool, default False
            By default, only a single matching filename is returned, otherwise
            an exception is raised.  Alternatively, set this to true to return
            all matches.
        file_type : str, optional
            If provided, ensure that the located file path(s) have this extension.

        Returns
        -------
        Path or list[Path]
            A single Path if return_all is False, otherwise a list

        Raises
        ------
        FileNotFoundError
            If there are zero OR multiple matches.
        """
        target = self.get_trace_file_path(
            file_name,
            trace_dir=trace_dir,
            tail="-*",
            create_dirs=False,
            file_type=file_type,
        )
        target1 = str(target).replace(
            "-*", "-" + "[0123456789abcdef]" * 6
        )  # targets with hex tails
        target2 = str(target).replace("-*", "")  # targets without any tail
        result = list(glob.glob(target1)) + list(glob.glob(target2))
        if return_all:
            return result
        elif len(result) == 0:
            raise FileNotFoundError(file_name)
        elif len(result) == 1:
            return result[0]
        else:
            raise FileNotFoundError(f"multiple matches for {file_name}")

    def get_cache_dir(self, subdir=None) -> Path:
        """
        Get the cache directory, creating it if needed.

        The cache directory is used to store:
            - skim memmaps created by skim+dict_factories
            - tvpb tap_tap table cache
            - pre-compiled sharrow modules


        Parameters
        ----------
        subdir : Path-like, optional
            If given, get this subdirectory of the output_dir.

        Returns
        -------
        Path
        """
        if self.cache_dir is None:
            out = self.get_output_dir("cache")
        else:
            out = self.get_working_subdir(self.cache_dir)
            if subdir is not None:
                out = out.joinpath(subdir)
            if not out.exists():
                out.mkdir(parents=True)

        # create a git-ignore in the cache dir if it does not exist.
        # this helps prevent accidentally committing cache contents to git
        gitignore = out.joinpath(".gitignore")
        if not gitignore.exists():
            gitignore.write_text("/**")

        return out

    def get_sharrow_cache_dir(self) -> Path:
        """
        Get the sharrow cache directory, creating it if needed.

        The sharrow cache directory is used to store only sharrow's cache
        of pre-compiled functions.

        Returns
        -------
        Path
        """
        if self.sharrow_cache_dir is None:
            self.persist_sharrow_cache()
            out = self.sharrow_cache_dir
        else:
            out = self.get_working_subdir(self.sharrow_cache_dir)
        if not out.exists():
            out.mkdir(parents=True)

        # create a git-ignore in the sharrow cache dir if it does not exist.
        # this helps prevent accidentally committing cache contents to git
        gitignore = out.joinpath(".gitignore")
        if not gitignore.exists():
            gitignore.write_text("/**")

        return out

    def persist_sharrow_cache(self) -> None:
        """
        Change the sharrow cache directory to a persistent, user-global location.

        The change is made in-place to `sharrow_cache_dir` for this object. The
        location for the cache is selected by `platformdirs.user_cache_dir`.
        An extra directory layer based on the current numba version is also added
        to the cache directory, which allows for different sets of cache files to
        co-exist for different version of numba (i.e. different conda envs).
        This location is not configurable -- to select a different location,
        change the value of `FileSystem.sharrow_cache_dir` itself.

        See Also
        --------
        FileSystem.sharrow_cache_dir
        """
        import sharrow as sh

        sharrow_minor_version = ".".join(sh.__version__.split(".")[:2])
        self.sharrow_cache_dir = Path(
            platformdirs.user_cache_dir(appname="ActivitySim")
        ).joinpath(f"sharrow-{sharrow_minor_version}-numba-{numba.__version__}")
        self.sharrow_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cascading_input_file_path(
        self, file_name, dir_list_injectable_name, mandatory=True, allow_glob=False
    ) -> Path:
        """
        Find the first matching file among a group of directories.

        Parameters
        ----------
        file_name : Path-like
            The name of the file to match.
        dir_list_injectable_name : {'configs_dir', 'data_dir'}
            The group of directories to search.
        mandatory : bool, default True
            Raise a FileNotFoundError if no match is found.  If set to False,
            this method returns None when there is no match.
        allow_glob : bool, default False
            Allow glob-style matches.

        Returns
        -------
        Path or None
        """
        if dir_list_injectable_name == "configs_dir":
            dir_paths = self.get_configs_dir()
        elif dir_list_injectable_name == "data_dir":
            dir_paths = self.get_data_dir()
        else:
            dir_paths = getattr(self, dir_list_injectable_name)
        dir_paths = [dir_paths] if isinstance(dir_paths, str) else dir_paths

        file_path = None
        if file_name is not None:
            for dir in dir_paths:
                p = os.path.join(dir, file_name)
                if os.path.isfile(p):
                    file_path = p
                    break

                if allow_glob and len(glob.glob(p)) > 0:
                    file_path = p
                    break

        if mandatory and not file_path:
            raise FileNotFoundError(
                "file_path %s: file '%s' not in %s"
                % (dir_list_injectable_name, file_name, [str(i) for i in dir_paths])
            )

        return Path(file_path) if file_path else None

    def expand_input_file_list(self, input_files) -> list[Path]:
        """
        expand list by unglobbing globs globs
        """

        # be nice and accept a string as well as a list of strings
        if isinstance(input_files, (str, Path)):
            input_files = [Path(input_files)]
        else:
            input_files = [Path(i) for i in input_files]

        expanded_files = []
        ungroked_files = 0

        for file_name in input_files:
            file_name = self.get_data_file_path(file_name, allow_glob=True)

            if file_name.is_file():
                expanded_files.append(file_name)
                continue

            if file_name.is_dir():
                logger.warning(
                    "WARNING: _expand_input_file_list skipping directory: "
                    f"(use glob instead): {file_name}",
                )
                ungroked_files += 1
                continue

            # - not an exact match, could be a glob pattern
            logger.debug(f"expand_input_file_list trying {file_name} as glob")
            globbed_files = glob.glob(str(file_name))
            for globbed_file in globbed_files:
                if os.path.isfile(globbed_file) or os.path.islink(globbed_file):
                    expanded_files.append(Path(globbed_file))
                else:
                    logger.warning(
                        "WARNING: expand_input_file_list skipping: "
                        f"(does not grok) {file_name}"
                    )
                    ungroked_files += 1

            if len(globbed_files) == 0:
                logger.warning(
                    f"WARNING: expand_input_file_list file/glob not found: {file_name}",
                )

        assert ungroked_files == 0, f"{ungroked_files} ungroked file names"

        return sorted(expanded_files)

    def get_configs_dir(self) -> tuple[Path]:
        """
        Get the configs directories.

        Returns
        -------
        tuple[Path]
        """
        return tuple(self.get_working_subdir(i) for i in self.configs_dir)

    def get_config_file_path(
        self, file_name: Path | str, mandatory: bool = True, allow_glob: bool = False
    ) -> Path:
        """
        Find the first matching file among config directories.

        Parameters
        ----------
        file_name : Path-like
            The name of the file to match.
        mandatory : bool, default True
            Raise a FileNotFoundError if no match is found.  If set to False,
            this method returns None when there is no match.
        allow_glob : bool, default False
            Allow glob-style matches.

        Returns
        -------
        Path or None
        """
        return self._cascading_input_file_path(
            file_name, "configs_dir", mandatory, allow_glob
        )

    def get_data_dir(self) -> tuple[Path]:
        """
        Get the data directories.

        Returns
        -------
        tuple[Path]
        """
        return tuple(self.get_working_subdir(i) for i in self.data_dir)

    def get_data_file_path(
        self, file_name, mandatory=True, allow_glob=False, alternative_suffixes=()
    ) -> Path:
        """
        Find the first matching file among data directories.

        Parameters
        ----------
        file_name : Path-like
            The name of the file to match.
        mandatory : bool, default True
            Raise a FileNotFoundError if no match is found.  If set to False,
            this method returns None when there is no match.
        allow_glob : bool, default False
            Allow glob-style matches.
        alternative_suffixes : Iterable[str], optional
            Other file suffixes to search for, if the expected filename is
            not found. This allows, for example, the data files to be stored
            as compressed csv ("*.csv.gz") without changing the config files.

        Returns
        -------
        Path or None
        """
        try:
            return self._cascading_input_file_path(
                file_name, "data_dir", mandatory, allow_glob
            )
        except FileNotFoundError:
            if not allow_glob:
                file_name = Path(file_name)
                for alt in alternative_suffixes:
                    alt_file = self._cascading_input_file_path(
                        file_name.with_suffix(alt), "data_dir", mandatory=False
                    )
                    if alt_file:
                        return alt_file
            raise

    def open_log_file(self, file_name, mode, header=None, prefix=False):
        if prefix:
            file_name = f"{prefix}-{file_name}"
        file_path = self.get_log_file_path(file_name)

        want_header = header and not os.path.exists(file_path)

        f = open(file_path, mode)

        if want_header:
            assert mode in [
                "a",
                "w",
            ], f"open_log_file: header requested but mode was {mode}"
            print(header, file=f)

        return f

    def read_settings_file(
        self,
        file_name: str,
        mandatory: bool = True,
        include_stack: bool = False,
        configs_dir_list: tuple[Path] | None = None,
        validator_class: type[PydanticBase] | None = None,
    ) -> PydanticBase | dict:
        """
        Load settings from one or more yaml files.

        This method will look for first occurrence of a yaml file named
        <file_name> in the directories in configs_dir list, and
        read settings from that yaml file.

        Settings file may contain directives that affect which file settings
        are returned:

        - inherit_settings (boolean)
            If found and set to true, this method will backfill settings
            in the current file with values from the next settings file
            in configs_dir list (if any)
        - include_settings: string <include_file_name>
            Read settings from specified include_file in place of the current
            file. To avoid confusion, this directive must appear ALONE in the
            target file, without any additional settings or directives.

        Parameters
        ----------
        file_name : str
        mandatory : boolean, default True
            If true, raise SettingsFileNotFoundError if no matching settings file
            is found in any config directory, otherwise this method will return
            an empty dict or an all-default instance of the validator class.
        include_stack : boolean or list
            Only used for recursive calls, provides a list of files included
            so far to detect and prevent cycles.
        validator_class : type[pydantic.BaseModel], optional
            This model is used to validate the loaded settings.

        Returns
        -------
        dict or validator_class
        """
        if isinstance(file_name, Path):
            file_name = str(file_name)

        def backfill_settings(settings, backfill):
            new_settings = backfill.copy()
            new_settings.update(settings)
            return new_settings

        if configs_dir_list is None:
            configs_dir_list = self.get_configs_dir()
            assert len(configs_dir_list) == len(
                set(configs_dir_list)
            ), f"repeating file names not allowed in config_dir list: {configs_dir_list}"

        args = parse_suffix_args(file_name)
        file_name = args.filename

        assert isinstance(args.ROOTS, list)
        assert (args.SUFFIX is not None and args.ROOTS) or (
            args.SUFFIX is None and not args.ROOTS
        ), (
            "Expected to find both 'ROOTS' and 'SUFFIX' in %s, missing one"
            % args.filename
        )

        if not file_name.lower().endswith(".yaml"):
            file_name = "%s.yaml" % (file_name,)

        inheriting = False
        settings = {}
        if isinstance(include_stack, list):
            source_file_paths = include_stack.copy()
        else:
            source_file_paths = []
        for dir in configs_dir_list:
            file_path = os.path.join(dir, file_name)
            if os.path.exists(file_path):
                if inheriting:
                    # we must be inheriting
                    logger.debug(
                        "inheriting additional settings for %s from %s"
                        % (file_name, file_path)
                    )
                    inheriting = True

                assert (
                    file_path not in source_file_paths
                ), f"read_settings_file - recursion in reading 'file_path' after loading: {source_file_paths}"

                with open(file_path) as f:
                    s = yaml.load(f, Loader=yaml.SafeLoader)
                    if s is None:
                        s = {}

                settings = backfill_settings(settings, s)

                # maintain a list of files we read from to improve error message when an expected setting is not found
                source_file_paths += [file_path]

                include_file_name = s.get("include_settings", False)
                if include_file_name:
                    # FIXME - prevent users from creating borgesian garden of branching paths?
                    # There is a lot of opportunity for confusion if this feature were over-used
                    # Maybe we insist that a file with an include directive is the 'end of the road'
                    # essentially the current settings firle is an alias for the included file
                    if len(s) > 1:
                        logger.error(
                            "'include_settings' must appear alone in settings file."
                        )
                        additional_settings = list(
                            set(s.keys()).difference({"include_settings"})
                        )
                        logger.error(
                            f"Unexpected additional settings: {additional_settings}"
                        )
                        raise RuntimeError(
                            "'include_settings' must appear alone in settings file."
                        )

                    logger.debug(
                        "including settings for %s from %s"
                        % (file_name, include_file_name)
                    )

                    # recursive call to read included file INSTEAD of the file  with include_settings sepcified
                    s, source_file_paths = self.read_settings_file(
                        include_file_name,
                        mandatory=True,
                        include_stack=source_file_paths,
                    )

                    # FIXME backfill with the included file
                    settings = backfill_settings(settings, s)

                # we are done as soon as we read one file successfully
                # unless if inherit_settings is set to true in this file

                if not s.get("inherit_settings", False):
                    break

                # if inheriting, continue and backfill settings from the next existing settings file configs_dir_list

                inherit_settings = s.get("inherit_settings")
                if isinstance(inherit_settings, str):
                    inherit_file_name = inherit_settings
                    assert (
                        os.path.join(dir, inherit_file_name) not in source_file_paths
                    ), f"circular inheritance of {inherit_file_name}: {source_file_paths}: "
                    # make a recursive call to switch inheritance chain to specified file

                    logger.debug(
                        "inheriting additional settings for %s from %s"
                        % (file_name, inherit_file_name)
                    )
                    s, source_file_paths = self.read_settings_file(
                        inherit_file_name,
                        mandatory=True,
                        include_stack=source_file_paths,
                        configs_dir_list=configs_dir_list,
                    )

                    # backfill with the inherited file
                    settings = backfill_settings(settings, s)
                    break  # break the current inheritance chain (not as bad luck as breaking a chain-letter chain?...)

        if len(source_file_paths) > 0:
            settings["source_file_paths"] = source_file_paths

        if mandatory and not settings:
            raise SettingsFileNotFoundError(file_name, configs_dir_list)

        # Adds proto_ suffix for disaggregate accessibilities
        if args.SUFFIX is not None and args.ROOTS:
            settings = suffix_tables_in_settings(settings, args.SUFFIX, args.ROOTS)

        # we don't want to actually have inherit_settings or include_settings
        # as they won't validate
        settings.pop("inherit_settings", None)
        settings.pop("include_settings", None)

        if validator_class is not None:
            settings = validator_class.model_validate(settings)

        if include_stack:
            # if we were called recursively, return an updated list of source_file_paths
            return settings, source_file_paths

        else:
            return settings

    def read_model_settings(
        self,
        file_name,
        mandatory=False,
    ):
        # in the legacy implementation, this function has a default mandatory=False
        return self.read_settings_file(file_name, mandatory=mandatory)

    def read_model_spec(self, file_name: Path | str):
        from activitysim.core import simulate

        return simulate.read_model_spec(self, file_name)

    def read_model_coefficients(
        self,
        model_settings: LogitComponentSettings | dict[str, Any] | None = None,
        file_name: str | None = None,
    ) -> pd.DataFrame:
        from activitysim.core import simulate

        return simulate.read_model_coefficients(
            self, model_settings=model_settings, file_name=file_name
        )

    def get_segment_coefficients(
        self, model_settings: PydanticBase | dict, segment_name: str
    ):
        from activitysim.core import simulate

        return simulate.get_segment_coefficients(self, model_settings, segment_name)
