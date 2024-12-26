from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

# suppress RuntimeWarning from xarray
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="xarray",
)


def prepare(
    household_sample_size: int = 20_000, subdir: str = "test-estimation-data"
) -> Path:
    """Prepare the example for estimation.

    This function prepares the example for estimation by downloading the example data and
    setting up the working directory. The current working directory is then set to the
    created example directory.

    Parameters
    ----------
    household_sample_size : int, optional
        The number of households to sample from the synthetic population. The default is 20_000.
    subdir : str, optional
        The subdirectory to store the example data. The default is "test-estimation-data".

    Returns
    -------
    Path
        The path to the created example directory.
    """
    root_dir = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(root_dir))

    try:
        from build_full_mtc_example import as_needed
    except ImportError:
        print(
            "Please run this script from the "
            "activitysim/examples/example_estimation/notebooks directory."
        )
        raise

    as_needed(root_dir / "notebooks" / subdir, household_sample_size)
    relative_path = os.path.relpath(
        root_dir / "notebooks" / subdir / "activitysim-prototype-mtc-extended"
    )
    os.chdir(relative_path)
    return Path(relative_path)


def backup(filename: str | os.PathLike):
    """Create or restore from a backup copy of a file."""
    backup_filename = f"{filename}.bak"
    if Path(backup_filename).exists():
        shutil.copy(backup_filename, filename)
    else:
        shutil.copy(filename, backup_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the example for estimation.")
    parser.add_argument(
        "--household_sample_size",
        type=int,
        default=20_000,
        help="The number of households to sample from the synthetic population.",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="test-estimation-data",
        help="The subdirectory to store the example data.",
    )
    args = parser.parse_args()
    prepare(args.household_sample_size, args.subdir)
