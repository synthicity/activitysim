import os
import pandas as pd
import numpy as np
import pandas.testing as pdt
import yaml


configs_estimation_dir = (
    "activitysim/estimation/test/test_edb_creation/configs_estimation"
)
configs_dir = "activitysim/examples/prototype_mtc/configs"
survey_data_dir = "activitysim/estimation/test/test_edb_creation/survey_data"
data_dir = "activitysim/examples/prototype_mtc/data"
base_output_dir = "activitysim/estimation/test/test_edb_creation/outputs"


def launch_est_example(output_dir, multiprocess, fileformat):

    # setting multiprocess setting
    settings_template_file = os.path.join(
        configs_estimation_dir, "settings_template.yaml"
    )
    settings_file = os.path.join(configs_estimation_dir, "settings.yaml")
    settings = yaml.safe_load(open(settings_template_file))
    assert multiprocess in [True, False]
    settings["multiprocess"] = multiprocess
    yaml.dump(settings, open(settings_file, "w"))

    # setting fileformat setting
    settings_template_file = os.path.join(
        configs_estimation_dir, "estimation_template.yaml"
    )
    est_settings_file = os.path.join(configs_estimation_dir, "estimation.yaml")
    est_settings = yaml.safe_load(open(settings_template_file))
    assert fileformat in ["csv", "parquet", "pkl"]
    est_settings["EDB_FILETYPE"] = fileformat
    yaml.dump(est_settings, open(est_settings_file, "w"))

    run_cmd = f"activitysim run -c {configs_estimation_dir} -c {configs_dir} -d {survey_data_dir} -d {data_dir} -o {base_output_dir}/{output_dir}"
    print(
        f"launching with options output_dir={output_dir} multiprocess={multiprocess} fileformat={fileformat}"
    )
    print("launching from ", os.getcwd())
    result = os.system(run_cmd)
    assert result == 0, "ActivitySim run failed"


def read_table(file_name):
    if file_name.endswith(".csv"):
        return pd.read_csv(file_name, low_memory=False)
    elif file_name.endswith(".parquet"):
        df = pd.read_parquet(file_name).reset_index(drop=True)
        df.columns.name = None
        return df
    elif file_name.endswith(".pkl"):
        df = pd.read_pickle(file_name).reset_index(drop=True)
        df.columns = df.columns.astype(str)
        df.columns.name = None
        return df
    else:
        raise ValueError(f"Unsupported file format: {file_name}")


def process_table(df):
    if "variable" == df.columns[1]:
        # need to sort variable column
        df = df.sort_values(by=[df.columns[0], "variable"]).reset_index(drop=True)
        df.columns.name = None

    if "chunk_id" in df.columns:
        # remove chunk_id column
        df = df.drop(columns=["chunk_id"])

    return df


def find_lowest_level_directories(starting_directory):
    lowest_dirs = list()

    for root, dirs, files in os.walk(starting_directory):
        if not dirs:
            lowest_dirs.append(root)

    return lowest_dirs


def try_compare_with_same_dtypes(ser1, ser2, rtol, atol):
    try:
        concatenated = pd.concat([ser1, ser2])
        common_type = concatenated.dtype
        pdt.assert_series_equal(
            ser1.astype(common_type),
            ser2.astype(common_type),
            check_dtype=False,
            rtol=rtol,
            atol=atol,
        )
    except (ValueError, AssertionError) as e:
        return False
    return True


def try_compare_with_numeric(ser1, ser2, rtol, atol):
    try:
        ser1_num = pd.to_numeric(ser1, errors="coerce")
        ser2_num = pd.to_numeric(ser2, errors="coerce")
        pdt.assert_series_equal(
            ser1_num, ser2_num, check_dtype=False, rtol=rtol, atol=atol
        )
    except AssertionError as e:
        return False
    return True


def try_compare_with_strings(ser1, ser2):
    try:
        pdt.assert_series_equal(ser1.astype(str), ser2.astype(str), check_dtype=False)
    except AssertionError as e:
        return False
    return True


def try_compare_with_combination(ser1, ser2, rtol=1e-5, atol=1e-8):
    """
    This is necessary because we have columns like this:
    [index]: [0, 1, 2, 3, 4, 5, ...]
    [left]:  [False, False, 1, 6, AM, -1.3037984712857993, EA, ...]
    [right]: [False, False, 1, 6, AM, -1.3037984712857994, EA, ...]
    (notice the machine precision difference in the float)
    """
    # replace annoying string values of bools
    ser1 = ser1.replace({"True": True, "False": False})
    ser2 = ser2.replace({"True": True, "False": False})
    # Separate numerical and non-numerical values
    ser1_num = pd.to_numeric(ser1, errors="coerce")
    ser2_num = pd.to_numeric(ser2, errors="coerce")
    ser1_non_num = ser1[ser1_num.isna()]
    ser2_non_num = ser2[ser2_num.isna()]
    ser1_num = ser1_num.dropna()
    ser2_num = ser2_num.dropna()
    try:
        pdt.assert_series_equal(
            ser1_num, ser2_num, check_dtype=False, rtol=rtol, atol=atol
        )
        pdt.assert_series_equal(ser1_non_num, ser2_non_num, check_dtype=False)
    except AssertionError as e:
        return False
    return True


def compare_dataframes_with_tolerance(df1, df2, rtol=1e-3, atol=1e-3):
    dfs_are_same = True
    e_msg = ""
    try:
        pdt.assert_frame_equal(df1, df2, check_dtype=False, rtol=rtol, atol=atol)
        return dfs_are_same, e_msg
    except AssertionError as e:
        print(e)
        print("trying to compare columns individually")
        for col in df1.columns:
            try:
                if try_compare_with_same_dtypes(df1[col], df2[col], rtol, atol):
                    continue
                elif try_compare_with_numeric(df1[col], df2[col], rtol, atol):
                    continue
                elif try_compare_with_strings(df1[col], df2[col]):
                    continue
                elif try_compare_with_combination(
                    df1[col], df2[col], rtol=rtol, atol=atol
                ):
                    continue
                else:
                    dfs_are_same = False
                    e_msg += f"Column '{col}' failed: {df1[col]} vs {df2[col]}\n"
                    print(f"Column '{col}' failed:\n {df1[col]} vs {df2[col]}\n")
            except Exception as e:
                dfs_are_same = False
                e_msg = e

    return dfs_are_same, e_msg


def regress_EDBs(regress_folder, output_folder, fileformat="csv"):
    edb_file = os.path.join(base_output_dir, regress_folder, "estimation_data_bundle")

    edb_dirs = find_lowest_level_directories(edb_file)

    for dir in edb_dirs:
        dir_basename = dir.split("estimation_data_bundle")[1][1:]
        output_dir = os.path.join(
            base_output_dir, output_folder, "estimation_data_bundle", dir_basename
        )

        for file in os.listdir(dir):
            if file.endswith(".yaml"):
                continue
            print(f"Regressing {file}")

            regress_path = os.path.join(dir, file)
            output_path = os.path.join(output_dir, file)

            # regress against csv for parquet, but regress against parquet for pkl (faster)
            if not os.path.exists(output_path) and (fileformat == "parquet"):
                output_path = output_path.replace(".csv", ".parquet")
            if not os.path.exists(output_path) and (fileformat == "pkl"):
                output_path = output_path.replace(".parquet", ".pkl")

            try:
                regress_df = read_table(regress_path)
                output_df = read_table(output_path)
            except FileNotFoundError as e:
                assert False, f"File not found: {e}"

            regress_df = process_table(regress_df)
            output_df = process_table(output_df)

            dfs_are_same, e = compare_dataframes_with_tolerance(regress_df, output_df)
            if not dfs_are_same:
                assert False, f"Regression test failed for {file} with error: {e}"
    return


def test_infer():
    infer_file = r"activitysim/examples/example_estimation/scripts/infer.py"
    data_dir = r"activitysim/estimation/test/test_edb_creation"
    configs_dir = r"activitysim/examples/prototype_mtc/configs"
    output_dir = r"activitysim/estimation/test/test_edb_creation/outputs/infer_output"
    run_cmd = f"python {infer_file} {data_dir} {configs_dir} {output_dir}"
    print(f"launching {run_cmd} from {os.getcwd()}")
    result = os.system(run_cmd)
    assert result == 0, "Infer.py run failed"

    for file in os.listdir(output_dir):
        if not file.endswith(".csv"):
            continue
        print("Regressing ", file)
        regress_df = pd.read_csv(os.path.join(data_dir, "survey_data", file))
        output_df = pd.read_csv(os.path.join(output_dir, file))

        pdt.assert_frame_equal(regress_df, output_df)


def test_generating_sp_csv():
    # first generate original tables
    output_dir = "output_single_csv"
    launch_est_example(output_dir, False, "csv")


def test_sp_parquet():
    # multiprocess = False, fileformat = "parquet"
    output_dir = "output_single_parquet"
    launch_est_example(output_dir, False, "parquet")
    regress_EDBs("output_single_csv", output_dir, "parquet")


def test_sp_pkl():
    # multiprocess = False, fileformat = "pkl"
    output_dir = "output_single_pkl"
    launch_est_example(output_dir, False, "pkl")
    regress_EDBs("output_single_parquet", output_dir, "pkl")


def test_mp_parquet():
    # multiprocess = True, fileformat = "parquet"
    output_dir = "output_multiprocess_parquet"
    launch_est_example(output_dir, True, "parquet")
    regress_EDBs("output_single_parquet", output_dir, "parquet")


if __name__ == "__main__":

    test_infer()
    test_generating_sp_csv()
    test_sp_pkl()
    test_sp_parquet()
    test_mp_parquet()
