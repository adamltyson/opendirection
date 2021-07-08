import pandas as pd
import logging
from functools import reduce


class ConditionError(Exception):
    pass


def condition_select(
    df, all_conditions, chosen_condition, column="condition", stability=None
):
    """
    Takes a dataframe, and returns rows based on the conditions in the given
    'column'. If a specific chosen_condition is given, only those rows are
    returned. If 'all_conditions' is given, then all of the conditions in
    'all_conditions' are merged into one. If 'all_data' is given, then no
    selection is performed, and all of the data is returned.
    :param df: dataframe
    :param all_conditions: list of conditions to choose from
    :param chosen_condition: Selected condition, or 'all_conditions' or
    'all_data'
    :param str column: Condition column heading. Default: 'condition'
    :param str stability: If None, return full df. If "first", return the first
    half. If "last", return the second half.
    :return: Dataframe with only the rows corresponding to 'chosen_condition'
    """
    logging.info("Selecting behavioural conditions")
    if chosen_condition == "all_data":
        logging.info(
            'Condition: "{}" specified, keeping all data'.format(
                chosen_condition
            )
        )

    elif chosen_condition == "all_conditions":
        df = df[df[column].isin(all_conditions)]
        logging.info(
            'Condition: "{}" specified, analysing all conditions'
            "as one.".format(chosen_condition)
        )

    elif chosen_condition in all_conditions:
        df = df[df[column] == chosen_condition]
        logging.info(
            'Condition: "{}" specified, analysing condition'
            " separately".format(chosen_condition)
        )
    else:
        logging.error(
            'Condition: "{}" not found. Raising error.'.format(
                chosen_condition
            )
        )
        raise ConditionError(
            'Condition: "{}" not found. Please ensure that the list of '
            'conditions to analyse matches those in "conditions.ini"'.format(
                chosen_condition
            )
        )
    if stability == "first":
        return df.iloc[: len(df) // 2]
    elif stability == "last":
        return df.iloc[len(df) // 2 :]
    else:
        return df


def add_conditions(df, config):
    # supress copy of df warning
    pd.options.mode.chained_assignment = None  # default='warn'
    starts = [
        i / config.camera_frames_per_sec
        for i in config.condition_timing_starts
    ]
    ends = [
        i / config.camera_frames_per_sec for i in config.condition_timing_ends
    ]

    # deal with the EXCLUSION section
    # parse the list of exclusion times
    excludes = []
    for condition_exclusion_list in config.condition_timing_excludes:
        indv_exclusion_criteria = []
        for exclusion_criteria in condition_exclusion_list:
            if exclusion_criteria != "None":
                exclusion_criteria = [
                    int(i) / config.camera_frames_per_sec
                    for i in exclusion_criteria.split("-")
                ]
            indv_exclusion_criteria.append(exclusion_criteria)
        excludes.append(indv_exclusion_criteria)

    # set the default condition to NONE
    df["condition"] = "NONE"

    # make masks based on all of the start, ends and exclusion conditions
    for idx, index in enumerate(config.conditions):
        mask = df.index > starts[idx]
        mask = mask * (df.index < ends[idx])
        for exclusion_condition in excludes[idx]:
            if exclusion_condition != "None":
                temp_mask = df.index > exclusion_condition[0]
                temp_mask = temp_mask * (df.index < exclusion_condition[1])
                mask = mask * ~temp_mask
        df["condition"][mask] = index

    return df


def cell_list_combine(conditions, cell_condition_inclusion):
    # choose which cells to keep. If neither, do nothing and keep only
    # cells that meet criteria for each condition
    cells_in_all_conditions = []
    for condition in conditions:
        cells_in_all_conditions.append(condition.cell_list)

    if cell_condition_inclusion == "any":
        final_cell_list = list(set().union(*cells_in_all_conditions))
    elif cell_condition_inclusion == "all":
        final_cell_list = list(
            reduce(
                set.intersection,
                [set(item) for item in cells_in_all_conditions],
            )
        )

    for condition in conditions:
        condition.cell_list = final_cell_list

    return conditions
