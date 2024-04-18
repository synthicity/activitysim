# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd


def run_if_exists(filename):
    import pytest

    stack = traceback.extract_stack()
    base_dir = Path(stack[-2].filename).parent
    target_file = base_dir.joinpath(filename)

    return pytest.mark.skipif(
        not target_file.exists(), reason=f"required file {filename} is missing"
    )


def assert_frame_substantively_equal(
    left,
    right,
    *args,
    ignore_column_order=True,
    ignore_extra_columns_left=False,
    check_column_type_loosely=False,
    **kwargs,
):
    """
    Check that left and right DataFrame are substantively equal.

    This method generalizes the usual pandas DataFrame test, by allowing
    the ordering of columns to be different, and allowing the left dataframe to
    have extra columns (e.g. as might happen if more reporting or debugging
    data is output into a dataframe, but we want to make sure that the "core"
    expected things are all there and correct.

    Parameters
    ----------
    left, right : pd.DataFrame
    *args
        Forwarded to pandas.testing.assert_frame_equal
    ignore_column_order : bool, default True
        Keyword only argument.
    ignore_extra_columns_left : bool, default False
        This cannot be True unless `ignore_column_order` is also True
    check_column_type_loosely : bool, default False
        Check that the dtype kind matches, not the dtype itself, for example
        if one column is int32 and the other is int64 that is ok.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    **kwargs
        Forwarded to pandas.testing.assert_frame_equal
    """
    __tracebackhide__ = True  # don't show this code in pytest outputs

    if ignore_extra_columns_left:
        assert ignore_column_order
        assert set(right.columns).issubset(left.columns)
        left = left[right.columns]

    elif ignore_column_order:
        # column order may not match, so fix it before checking
        assert sorted(left.columns) == sorted(right.columns)
        # if there are duplicate column names, we disavow this option
        if not left.columns.has_duplicates:
            left = left[right.columns]

    if check_column_type_loosely:
        left_kinds = {k: i.kind for k, i in left.dtypes.items()}
        right_kinds = {k: i.kind for k, i in left.dtypes.items()}
        assert left_kinds == right_kinds
        kwargs["check_column_type"] = False

    try:
        pd.testing.assert_frame_equal(left, right, *args, **kwargs)
    except Exception as err:
        print(err)
        raise


def assert_equal(x, y):
    __tracebackhide__ = True  # don't show this code in pytest outputs
    try:
        import pytest
    except ImportError:
        assert x == y
    else:
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for n_, (x_, y_) in enumerate(zip(x, y)):
                assert x_ == pytest.approx(y_), f"error at index {n_}"
        elif isinstance(x, dict) and isinstance(y, dict) and x.keys() == y.keys():
            for n_ in x.keys():
                assert x[n_] == pytest.approx(y[n_]), f"error at key {n_}"
        else:
            try:
                assert x == pytest.approx(y)
            except (TypeError, AssertionError):
                # pytest.approx() does not support nested data structures
                for x_, y_ in zip(x, y):
                    assert x_ == pytest.approx(y_)


def progressive_checkpoint_test(
    state, ref_target: Path, expected_models: list[str], name: str = "unnamed-example"
) -> None:
    """
    Compare the results of a pipeline to a reference pipeline.

    Parameters
    ----------
    state : workflow.State
    ref_target : Path
        Location of the reference pipeline file.  If this file does not exist,
        it will be created (and the test will fail).
    expected_models : list[str]
        List of model names to run and compare results against the reference
        pipeline.
    name : str, optional
        Name of the test example used in logging, by default "unnamed-example"
    """

    for step_name in expected_models:
        state.run.by_name(step_name)
        if ref_target.exists():
            try:
                state.checkpoint.check_against(ref_target, checkpoint_name=step_name)
            except Exception as e:
                print(f"> {name} {step_name}: ERROR {e}")
                raise
            else:
                print(f"> {name} {step_name}: ok")
        else:
            print(f"> {name} {step_name}: regenerated")

    # generate the reference pipeline if it did not exist
    if not ref_target.exists():
        state.checkpoint.store.make_zip_archive(ref_target)
        raise RuntimeError(
            f"Reference pipeline {ref_target} did not exist, so it was created."
        )
