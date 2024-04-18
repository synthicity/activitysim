from __future__ import annotations

import glob
import hashlib
import importlib.resources
import logging
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pkg_resources
import requests
import yaml

PACKAGE = "activitysim"
EXAMPLES_DIR = "examples"
MANIFEST = "example_manifest.yaml"


def _example_path(resource):
    resource = os.path.join(EXAMPLES_DIR, resource)
    return importlib.resources.as_file(
        importlib.resources.files(PACKAGE).joinpath(resource)
    )


def _load_manifest():
    with _example_path(MANIFEST) as f_pth:
        with open(f_pth, "r") as f:
            manifest = yaml.safe_load(f.read())

    assert manifest, f"error: could not load {MANIFEST}"
    return {example["name"]: example for example in manifest}


EXAMPLES = _load_manifest()


def add_create_args(parser):
    """Create command args"""
    create_group = parser.add_mutually_exclusive_group(required=True)
    create_group.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list available example directories and exit",
    )
    create_group.add_argument(
        "-e", "--example", type=str, metavar="PATH", help="example directory to copy"
    )

    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        metavar="PATH",
        default=os.getcwd(),
        help="path to new project directory (default: %(default)s)",
    )

    parser.add_argument(
        "--link",
        action="store_true",
        help="cache and reuse downloaded files via symlinking",
    )


def create(args):
    """
    Create a new ActivitySim configuration from an existing template.

    Use the -l flag to view a list of example configurations, then
    copy to your own working directory. These new project files can
    be run with the 'run' command.
    """

    if args.list:
        list_examples()
        return 0

    if args.example:
        try:
            get_example(args.example, args.destination, link=args.link)
        except Exception:
            logging.getLogger().exception("failure in activitysim create")
            return 101
        return 0


def list_examples():
    print("*** Available examples ***\n")

    ret = []
    for example in list(EXAMPLES.values()):
        del example["include"]
        ret.append(example)
        print(yaml.dump(example))

    return ret


def get_example(
    example_name,
    destination,
    benchmarking=False,
    optimize=True,
    link=True,
    with_subdirs=False,
):
    """
    Copy project data to user-specified directory.

    Examples and their data are described in a manifest
    YAML file. Each example contains at least a `name` and
    `include` field which is a list of files/folders to include
    in the copied example.

    Parameters
    ----------
    example_name: str, name of the example to copy.
        Options can be found via list_examples()
    destination: name of target directory to copy files to.
        If the target directory does not exist, it is created.
        Project files will then be copied into a subdirectory
        with the same name as the example
    benchmarking: bool
    optimize: bool
    link: bool or path-like
        Files downloaded via http pointers will be cached in
        this location.  If a path is not given but just a truthy
        value, then a cache directory is created using in a location
        selected by the platformdirs library (or, if not installed,
        linking is skipped.)
    with_subdirs: bool, default False
        Also return any instructions about sub-directories.

    Returns
    -------
    Path or (Path, dict)
        The path to the location where the example was installed, and
        optionally also a mapping of example subdirectory locations.
    """
    if example_name not in EXAMPLES:
        sys.exit(f"error: could not find example '{example_name}'")

    if os.path.isdir(destination):
        dest_path = os.path.join(destination, example_name)
    elif os.path.isfile(destination):
        raise FileExistsError(destination)
    else:
        os.makedirs(destination)
        dest_path = os.path.join(destination, example_name)

    example = EXAMPLES[example_name]
    itemlist = example.get("include", [])
    if benchmarking:
        itemlist.extend(example.get("benchmarking", []))

    for item in itemlist:
        # split include string into source/destination paths
        items = item.split()
        assets = items[0]
        if len(items) == 3:
            target_path = os.path.join(dest_path, items[1])
            sha256 = items[-1]
        elif len(items) == 2:
            target_path = os.path.join(dest_path, items[-1])
            sha256 = None
        else:
            target_path = dest_path
            sha256 = None

        if assets.startswith("http"):
            download_asset(
                assets, target_path, sha256, link=link, base_path=destination
            )

        else:
            with _example_path(assets) as pth:
                for asset_path in glob.glob(str(pth)):
                    copy_asset(asset_path, target_path, dirs_exist_ok=True)

    print(f"copied! new project files are in {os.path.abspath(dest_path)}")

    if optimize:
        optimize_func_names = example.get("optimize", None)
        if isinstance(optimize_func_names, str):
            optimize_func_names = [optimize_func_names]
        if optimize_func_names:
            from ..examples import optimize_example_data

            for optimize_func_name in optimize_func_names:
                getattr(
                    optimize_example_data,
                    optimize_func_name,
                )(os.path.abspath(dest_path))

    instructions = example.get("instructions")
    if instructions:
        print(instructions)

    if with_subdirs:
        subdirs = example.get("subdirs", {})
        subdirs.setdefault("configs_dir", ("configs",))
        subdirs.setdefault("data_dir", ("data",))
        subdirs.setdefault("output_dir", "output")

        return Path(dest_path), subdirs
    else:
        return Path(dest_path)


def copy_asset(asset_path, target_path, dirs_exist_ok=False):
    print(f"copying {os.path.basename(asset_path)} ...")
    sys.stdout.flush()
    if os.path.isdir(asset_path):
        target_path = os.path.join(target_path, os.path.basename(asset_path))
        shutil.copytree(asset_path, target_path, dirs_exist_ok=dirs_exist_ok)

    else:
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        shutil.copy(asset_path, target_path)


def _decompress_archive(archive_path: Path, target_location: Path):
    # decompress archive file into working directory
    if archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path) as tfile:
            common_prefix = os.path.commonprefix(tfile.getnames())
            if common_prefix in {"", ".", "./", None}:
                working_dir = target_location
                working_dir.mkdir(parents=True, exist_ok=True)
                working_subdir = working_dir
            else:
                working_subdir = target_location.joinpath(common_prefix)
            tfile.extractall(working_dir)
    elif archive_path.suffixes[-2:] == [".tar", ".zst"]:
        working_dir = target_location
        try:
            working_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        working_subdir = working_dir
        from sharrow.utils.tar_zst import extract_zst

        extract_zst(archive_path, working_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            common_prefix = os.path.commonprefix(zf.namelist())
            if common_prefix in {"", ".", "./", None}:
                working_dir = target_location
                working_dir.mkdir(parents=True, exist_ok=True)
                working_subdir = working_dir
            else:
                working_subdir = target_location.joinpath(common_prefix)
            zf.extractall(working_dir)
    else:
        raise ValueError(f"unknown archive file type {''.join(archive_path.suffixes)}")
    return working_subdir


def download_asset(
    url: str,
    target_path: str,
    sha256: str = None,
    link: bool = True,
    base_path: str | None = None,
    unpack: str | None = None,
):
    """
    Download assets (extra files) associated with examples.

    Parameters
    ----------
    url : str
        The URL to download.
    target_path : str
        The location where the asset should be made available. The raw asset
        file is not necessarily stored here, as it may be stored in a cache
        directory and symlinked here instead (see `link`).
    sha256 : str, optional
        Checksum for the file.  If there is already a cached file and the
        checksum matches, it is not re-downloaded and the cached version is
        used.  Otherwise, the file is downloaded, and if the downloaded file's
        checksum does not match, an error is raised.
    link : bool, default True
        Download the raw asset to a cache location, and then symlink to the
        desired `target_path` location.  Note symlinks may not work on Windows
        so the file will still be stored in the cache but it will be *copied*
        instead of linked.
    base_path : str, optional
        Give the base directory for the example.
    unpack : str, optional
        If the asset is an archive file (.zip, .tar.gz, or .tar.zst), it
        will be decompressed into this location.
    """
    if isinstance(target_path, Path):
        target_path = str(target_path)
    original_target_path = target_path
    if link or unpack:
        original_target_path = target_path
        if base_path is not None and os.path.isabs(target_path):
            target_path = os.path.relpath(target_path, base_path)
        if base_path is not None:
            if unpack:
                if os.path.isabs(unpack):
                    unpack = os.path.relpath(unpack, base_path)
                else:
                    unpack = os.path.join(base_path, unpack)
        if not isinstance(link, str | Path):
            try:
                import platformdirs
            except ImportError:
                link = False
            else:
                link = platformdirs.user_data_dir("ActivitySim")
        target_path = os.path.join(link, target_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if url.endswith(".gz") and not target_path.endswith(".gz"):
        target_path_dl = target_path + ".gz"
    else:
        target_path_dl = target_path
    download = True
    if sha256 and os.path.isfile(target_path):
        computed_sha256 = sha256_checksum(target_path)
        if sha256 == computed_sha256:
            print(f"not re-downloading existing {os.path.basename(target_path)} ...")
            download = False
        else:
            print(f"re-downloading existing {os.path.basename(target_path)} ...")
            print(f"   expected checksum {sha256}")
            print(f"   computed checksum {computed_sha256}")
    else:
        print(f"downloading {os.path.basename(target_path)} ...")
    sys.stdout.flush()
    if download:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            print(f"|        as {target_path_dl}")
            with open(target_path_dl, "wb") as f:
                for chunk in r.iter_content(chunk_size=None):
                    f.write(chunk)
        if target_path_dl != target_path:
            import gzip

            with gzip.open(target_path_dl, "rb") as f_in:
                with open(target_path, "wb") as f_out:
                    print(f"|  unzip to {target_path}")
                    shutil.copyfileobj(f_in, f_out)
            os.remove(target_path_dl)
        computed_sha256 = sha256_checksum(target_path)
        if sha256 and sha256 != computed_sha256:
            raise ValueError(
                f"downloaded {os.path.basename(target_path)} has incorrect checksum\n"
                f"   expected checksum {sha256}\n"
                f"   computed checksum {computed_sha256}"
            )
        elif not sha256:
            print(f"|  computed checksum {computed_sha256}")
    if link or unpack:
        target_dir = os.path.dirname(os.path.normpath(original_target_path))
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        # check if the original_target_path exists and if so check if it is the correct file
        if os.path.isfile(os.path.normpath(original_target_path)):
            if sha256 is None:
                sha256 = sha256_checksum(os.path.normpath(target_path))
            existing_sha256 = sha256_checksum(os.path.normpath(original_target_path))
            if existing_sha256 != sha256:
                os.unlink(os.path.normpath(original_target_path))

        if unpack:
            _decompress_archive(
                Path(os.path.normpath(target_path)),
                Path(os.path.normpath(unpack)),
            )
            print(f"|  unpacked to {os.path.normpath(unpack)}")
        elif link:
            # if the original_target_path exists now it is the correct file, keep it
            if not os.path.isfile(os.path.normpath(original_target_path)):
                try:
                    os.symlink(
                        os.path.normpath(target_path),
                        os.path.normpath(original_target_path),
                    )
                except OSError:
                    # permission errors likely foil symlinking on windows
                    shutil.copy(
                        os.path.normpath(target_path),
                        os.path.normpath(original_target_path),
                    )
                    print(f"|    copied to {os.path.normpath(original_target_path)}")
                else:
                    print(f"| symlinked to {os.path.normpath(original_target_path)}")


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def display_sha256_checksums(directory=None):
    print("SHA 256 CHECKSUMS")
    if directory is None:
        if len(sys.argv) > 1:
            directory = sys.argv[1]
        else:
            directory = os.getcwd()
    print(f"  in {directory}")
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"- in {dirpath}")
        for filename in filenames:
            f = os.path.join(dirpath, filename)
            print(f"= {sha256_checksum(f)} = {f}")
