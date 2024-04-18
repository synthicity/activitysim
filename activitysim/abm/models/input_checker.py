from __future__ import annotations

import logging
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pandera as pa
import pydantic

from activitysim.core import workflow
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)
file_logger = logger.getChild("logfile")

global TABLE_STORE
TABLE_STORE = {}

_log_infos = {}

global cur_table
cur_table = None


def create_table_store(state, input_checker_settings):
    """
    creating a global variable called TABLE_STORE to be able to access
    all tables at any point in the input checker python code

    FIXME: can we do this better with the state object?
    """
    # looping through all tables listed in input_checker.yaml
    for table_settings in input_checker_settings["table_list"]:
        table = None
        path = None

        table_name = table_settings["name"]
        logger.info("reading in table for input checking: %s" % table_name)

        # read with ActivitySim's input reader if ActivitySim input
        if table_settings["is_activitysim_input"]:
            table = read_input_table(state, table_name).reset_index()

        # otherwise read csv file directly with pandas
        else:
            path = table_settings.get("path", None)
            if path:
                if os.path.isabs(path):
                    table = pd.read_csv(os.path.join(path, table_name + ".csv"))
                elif os.path.exists(os.path.join(os.getcwd(), table_name + ".csv")):
                    table = pd.read_csv(os.path.join(os.getcwd(), table_name + ".csv"))
                else:
                    for directory in state.filesystem.get_data_dir():
                        file = os.path.join(directory, path, table_name + ".csv")
                        if os.path.exists(file):
                            table = pd.read_csv(file)
                            break

            else:
                table = pd.read_csv(
                    state.filesystem.get_data_file_path(
                        table_name, alternative_suffixes=[".csv"]
                    )
                )
        if table is None:
            raise FileNotFoundError(
                f"Input table {table_name} could not be found" + f"\nPath: {path}"
                if path
                else ""
            )
        # add pandas dataframes to TABLE_STORE dictionary with table name as key
        TABLE_STORE[table_name] = table
        _log_infos[table_name] = list()

    # FIXME: need to have state object available in the input checker. Is there a better way to pass this?
    TABLE_STORE["state"] = state


def validate_with_pandera(
    input_checker, table_name, validation_settings, v_errors, v_warnings
):
    """
    Validating with pandera.  Grabs the relevant class for the table and runs the validate() function.

    Structure of the code is as follows:
    households = pd.DataFrame()
    in input_checker:
        class Household(pa.DataFrameModel)...
    validator_class = input_checker.Household()
    validator_class.validate(households)

    Warnings & errors are captured and written out in full after all tables are checked.
    """

    validator_class = getattr(input_checker, validation_settings["class"])

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            validator_class.validate(TABLE_STORE[table_name], lazy=True)
        except pa.errors.SchemaErrors as e:
            v_errors[table_name].append(e)

        for warning in caught_warnings:
            v_warnings[table_name].append(warning)

    return v_errors, v_warnings


def add_child_to_parent_list(pydantic_lists, parent_table_name, children_settings):
    """
    Code to efficiently add children to the parent list.  Used for input checking with pydantic.

    If "parent" is household and "child" is persons, the following code is equivalent to:
    for household in h_list:
        person_list = []
        for person in p_list:
            if household["household_id"] == person["household_id"]:
                person_list.append(person)
        household["persons"] = person_list

    Above code is slow with nested for loops. Actual implementation is much faster.
    """
    child_table_name = children_settings["table_name"]
    child_variable_name = children_settings["child_name"]
    merge_variable = children_settings["merged_on"]

    # need to create child list if it does not yet exist
    if child_table_name not in pydantic_lists.keys():
        pydantic_lists[child_table_name] = TABLE_STORE[child_table_name].to_dict(
            orient="records"
        )

    child_list = pydantic_lists[child_table_name]
    parent_list = pydantic_lists[parent_table_name]

    logger.info(
        f"Adding {child_table_name} to {parent_table_name} based on {merge_variable}"
    )

    parent_children = defaultdict(list)
    for child in child_list:
        children = child[merge_variable]
        parent_children[children].append(child)

    for parent in parent_list:
        children = parent[merge_variable]
        parent[child_variable_name] = parent_children.get(children, [])

    pydantic_lists[parent_table_name] = parent_list

    return pydantic_lists


def validate_with_pydantic(
    input_checker,
    table_name,
    validation_settings,
    v_errors,
    v_warnings,
    pydantic_lists,
):
    """
    FIXME: Not fully built out!! Went with Pandera instead of pydantic,
           but leaving this in for future development. Code is functional,
           but not setup for warnings and errors handling yet.

    Validing table with pydantic. Uses a helper class to perform the validation.

    Strucutre of the code is as follows:
    households = pd.DataFrame()
    in input_checker:
        class Household(pydantic.BaseModel)...
        class HouseholdValidator(pydantic.BaseModel) list_of_households...
    validator_class = input_checker.HouseholdValidator()
    h_list = households.to_dict() with optional addition of child class (aka Persons)
    validator_class(list_of_households=h_list)

    Warnings & errors are captured and written out in full after all tables are checked.
    """
    if table_name not in pydantic_lists.keys():
        pydantic_lists[table_name] = TABLE_STORE[table_name].to_dict(orient="records")

    if validation_settings.get("children") is not None:
        # FIXME need to add functionality for if there are multiple children
        pydantic_lists = add_child_to_parent_list(
            pydantic_lists, table_name, validation_settings["children"]
        )

    validator_class = getattr(input_checker, validation_settings["helper_class"])

    attr = validation_settings["helper_class_attribute"]

    try:
        validator_instance = validator_class(**{attr: pydantic_lists[table_name]})
    except pydantic.error_wrappers.ValidationError as e:
        v_errors[table_name].append(e)
    # FIXME need to implement validation warning
    # except ValidationWarning as w:
    # v_warnings[table_name].append(w)

    return v_errors, v_warnings, pydantic_lists


def report_errors(state, input_checker_settings, v_warnings, v_errors):

    # logging overall statistics first before printing details
    for table_settings in input_checker_settings["table_list"]:
        table_name = table_settings["name"]

        # printing out any errors
        errors = v_errors[table_name]
        warns = v_warnings[table_name]

        msg = f"Encountered {sum(len(entry.schema_errors) for entry in errors)} errors and {len(warns)} warnings in table {table_name}"

        # first printing to activitysim log file
        if len(errors) > 0:
            logger.error(msg)
        elif len(warns) > 0:
            logger.warning(msg)
        else:
            logger.info(msg)

        # printing to input_checker.log file
        file_logger.info(msg)
    file_logger.info("\n")

    # now reporting details to just input_checker.log
    input_check_failure = False
    # looping through each input table first
    for table_settings in input_checker_settings["table_list"]:
        table_name = table_settings["name"]

        if (
            len(v_errors[table_name]) > 0
            or len(v_warnings[table_name]) > 0
            or len(_log_infos[table_name]) > 0
        ):
            file_logger.info("#" * (len(table_name) + 4))
            file_logger.info("  " + table_name)
            file_logger.info("#" * (len(table_name) + 4))

        # printing out any errors
        errors = v_errors[table_name]
        if len(errors) > 0:
            input_check_failure = True
            file_logger.error(
                f"\n{table_name} errors:\n" + ("-" * (8 + len(table_name)))
            )

            for error_group in errors:
                file_logger.error("Error Counts\n------------")
                for error_type in error_group.error_counts:
                    file_logger.error(
                        f"{error_type}\t{error_group.error_counts[error_type]}\n",
                    )

                for error in error_group.schema_errors:
                    if "dataframe validator" in str(error):
                        file_logger.error(
                            "Failed dataframe validator: " + str(error).split("\n")[-1]
                        )
                    elif "element-wise validator" in str(error):
                        if "DataFrameSchema" in str(error):
                            file_logger.error(
                                "Failed element-wise validator: <"
                                + str(error).split("\n")[0].split(" ")[1]
                                + table_name
                                + ")>\n\t"
                                + str(error)
                                .split("failure cases:\n")[0]
                                .split("\n")[-2]
                                + "\n\tfailure cases:\n\t"
                                + "\n\t".join(
                                    str(error).split("failure cases:\n")[1].split("\n")
                                )
                            )
                        else:
                            file_logger.error(
                                "Failed element-wise validator: <"
                                + " ".join(str(error).split("\n")[0].split(" ")[1:3])
                                + "\n\t"
                                + "\n\t".join(str(error).split("\n")[1:])
                            )
                    else:
                        file_logger.error(str(error))
                    file_logger.error("\n")

        # printing out any warnings
        warns = v_warnings[table_name]
        if len(warns) > 0:
            file_logger.warning(
                f"\n{table_name} warnings:\n" + ("-" * (10 + len(table_name)))
            )

            for warn in warns:
                if "dataframe validator" in str(warn.message):
                    try:
                        file_logger.warning(
                            "Failed dataframe validator: "
                            + str(warn.message).split("\n")[-1]
                        )
                    except Exception:
                        file_logger.warning(warn)
                elif "element-wise validator" in str(warn.message):
                    try:
                        if "DataFrameSchema" in str(warn.message):
                            file_logger.warning(
                                "Failed element-wise validator: <"
                                + str(warn.message).split("\n")[0].split(" ")[1]
                                + table_name
                                + ")>\n\t"
                                + str(warn.message)
                                .split("failure cases:\n")[0]
                                .split("\n")[-2]
                                + "\n\tfailure cases:\n\t"
                                + "\n\t".join(
                                    str(warn.message)
                                    .split("failure cases:\n")[1]
                                    .split("\n")
                                )
                            )
                        else:
                            file_logger.warning(
                                "Failed element-wise validator: <"
                                + " ".join(
                                    str(warn.message).split("\n")[0].split(" ")[1:3]
                                )
                                + "\n\t"
                                + "\n\t".join(str(warn.message).split("\n")[1:])
                            )
                    except Exception:
                        file_logger.warning(warn)
                else:
                    file_logger.warning(warn)
            file_logger.warning("\n")

        infos = _log_infos[table_name]
        if len(infos) > 0:
            file_logger.info(
                f"\n{table_name} additional messages:\n"
                + ("-" * (21 + len(table_name)))
            )

            for info in infos:
                file_logger.info(info)
            file_logger.info("\n")

    if (len(v_warnings) > 0) | (len(v_errors) > 0):
        logger.info("See the input_checker.log for full details on errors and warnings")

    return input_check_failure


def log_info(text: str):
    _log_infos[cur_table].append(text)


@workflow.step()
def input_checker(state: workflow.State):
    """
    Input checker model is designed to be a stand-alone model that gets run at the
    start of each ActivitySim run to quickly check the inputs for potential errors.

    Users are able to write python code built on the pandera package to perform the checks.
    The ActivitySim code written in this module then imports the data according to the user
    specification and the input checks and passes them to pandera.

    Pandera will output warnings and errors which are captured and written to the input_checker.log file.
    Additional info can be written to the output log file from the user by calling the log_info() inside
    their input checking code.  See the ActivitySim model documentation for more detailed user instructions.
    """

    # creating a new log file to report out warnings and errors
    out_log_file = state.get_log_file_path("input_checker.log")
    if os.path.exists(out_log_file):
        os.remove(out_log_file)

    file_logger.addHandler(logging.FileHandler(out_log_file))
    file_logger.propagate = False

    input_checker_settings = state.filesystem.read_model_settings(
        "input_checker.yaml", mandatory=True
    )

    input_checker_file = input_checker_settings.get(
        "input_checker_code", "input_checks.py"
    )

    data_model_dir = state.get_injectable("data_model_dir")[0]
    logger.info("Data model directory: %s", data_model_dir)

    # FIXME: path doesn't recognize windows path object, so converting to string.
    # Is there a better way to get the data model directory than just adding it to the path?
    # Can't just import input_checker because enums.py also needs to be recognized.
    sys.path.append(str(data_model_dir))

    input_checker_file_full = os.path.join(
        state.filesystem.working_dir if state.filesystem.working_dir else ".",
        data_model_dir,
        input_checker_file,
    )

    assert os.path.exists(
        input_checker_file_full
    ), f"Cannot find input checker file {input_checker_file} in data_model_dir {data_model_dir}"

    create_table_store(state, input_checker_settings)

    # import the input checker code after the TABLE_STORE is initialized so functions have access to the variable
    prior_dir = os.getcwd()
    if state.filesystem.working_dir:
        os.chdir(state.filesystem.working_dir)
    input_checker = __import__(os.path.splitext(input_checker_file)[0])
    os.chdir(prior_dir)

    # intializing data objects for errors, warnings, and pydantic data
    v_errors = {}
    v_warnings = {}
    pydantic_lists = {}

    # loop through each input table checker
    for table_settings in input_checker_settings["table_list"]:
        validation_settings = table_settings["validation"]
        table_name = table_settings["name"]

        # creating current table global so that logger can access automatically without the
        # user having to specify when calling log_info
        global cur_table
        cur_table = table_name

        # initializing validation error and warning tracking
        v_errors[table_name] = []
        v_warnings[table_name] = []

        assert validation_settings["method"] in ["pandera", "pydantic"]

        if validation_settings["method"] == "pandera":
            logger.info(f"performing Pandera check on {table_settings['name']}")
            v_errors, v_warnings = validate_with_pandera(
                input_checker, table_name, validation_settings, v_errors, v_warnings
            )

        if validation_settings["method"] == "pydantic":
            logger.info(f"performing Pydantic check on {table_settings['name']}")
            logger.warning(f"Pydantic validation is not fully implemented yet!!")
            v_errors, v_warnings, pydantic_lists = validate_with_pydantic(
                input_checker,
                table_name,
                validation_settings,
                v_errors,
                v_warnings,
                pydantic_lists,
            )

    input_check_failure = report_errors(
        state, input_checker_settings, v_warnings, v_errors
    )

    # free memory from input checker tables
    for key, value in TABLE_STORE.items():
        del value
    TABLE_STORE.clear()

    if input_check_failure:
        logger.error("Run is killed due to input checker failure!!")
        raise RuntimeError(
            "Encountered error in input checker, see input_checker.log for details"
        )
