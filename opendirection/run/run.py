import logging
from fancylog import fancylog
import pandas as pd
import numpy as np
import os.path
import sys

from imlib.general.system import (
    get_num_processes,
    ensure_directory_exists,
    filename_from_path,
)
from imlib.pandas.misc import move_column_first
from imlib.general.parsing import convert_string_to_operation as str_op
import opendirection.tools.tools as tools
from opendirection.tools import cli_parser, config_parser, experiment_parser
import opendirection.behaviour.behaviour as behaviour
import opendirection.spikes.spikes as spikes
from opendirection.combine import clean, combine
import opendirection.combine.condition as condition_select
import opendirection.plot.plot as plotting
import opendirection.run.analyse as analyse


import opendirection as program_for_log


def setup(config_path):
    if len(sys.argv) == 2 and sys.argv[1] != "-h":
        args = experiment_parser.GetArgs(sys.argv[1])
    else:
        args = cli_parser.get_args()

    ensure_directory_exists(args.output_dir)

    if args.config_path is None:
        args.config_path = config_path

    config = config_parser.GetConfig(args.config_path)
    options = config_parser.GetOptions(args.config_path)
    options.num_processes = get_num_processes(
        min_free_cpu_cores=options.n_free_cpus
    )

    fancylog.start_logging(
        args.output_dir,
        program_for_log,
        variables=[args],
        verbose=args.verbose,
        log_header="OPENDIRECTION LOG",
    )

    return args, options, config


def load_align_data(args, options, config):
    # Load raw data and calculate behaviours
    behaviour_data = behaviour.get_behaviour(
        args.dlc_files,
        config.meters_per_pixel,
        config.camera_frames_per_sec,
        options,
    )
    num_behaviour_measurements = len(behaviour_data.index)

    # Load spiking data
    spike_data = spikes.get_spikes(
        args.spike_time_file,
        args.spike_clusters_file,
        args.cluster_groups_file,
        args.sync_param_file,
        num_behaviour_measurements,
        config.camera_frames_per_sec,
        config.probe_samples_per_sec,
    )

    # Align spiking data and behavioural data to a single dataframe
    total_df = combine.make_combined_df(spike_data, behaviour_data)

    # Remove any frames based on global conditions
    total_df = clean.clean_df(total_df, speed_cutoff=options.speed_cut_off)
    # Align data with externally supplied conditions
    total_df = condition_select.add_conditions(total_df, config)
    return total_df


def analysis(args, options, config, stability=None):

    total_df = load_align_data(args, options, config)

    # Initialise conditions class
    conditions = [
        analyse.ConditionAnalyse(
            total_df, condition, config, options, stability=stability
        )
        for condition in options.conditions_list
    ]

    # Combine cell lists if necessary
    if options.cell_condition_inclusion in ["all", "any"]:
        conditions = condition_select.cell_list_combine(
            conditions, options.cell_condition_inclusion
        )

    # Calculate remaining parameters based on new cell lists
    for condition in conditions:
        condition.cell_specific(options, config)

    return conditions, total_df


def save(
    condition, output_directory, cell_name_var="cell_name", suffix=".csv"
):
    """
    Goes through all the stats objects for each cell, gathers the name of
    each parameter and its value, then saves to .csv.

    :param condition:
    :param output_directory:
    :param cell_name_var:
    :param suffix:
    :return:
    """
    logging.info("Saving results for condition: " + condition.name)

    stats_type_objects = tools.get_cleaned_attrs_from_object(
        condition.cell_specific_stats[0], ignore_types=str
    )

    all_vars_dict = tools.get_attrs_multiple_sub_objects(
        condition.cell_specific_stats[0],
        stats_type_objects,
        return_dict_with_sub_objects=True,
        ignore_types=(tuple, np.ndarray),
    )

    save_df = pd.DataFrame(columns=all_vars_dict.keys())

    for cell_property, object_name in all_vars_dict.items():
        property_list = []
        for cell in condition.cell_specific_stats:
            cell_specific_obj = getattr(cell, object_name)
            property_list.append(getattr(cell_specific_obj, cell_property))
        save_df[cell_property] = property_list

    save_df = move_column_first(save_df, cell_name_var)
    filename = os.path.join(output_directory, condition.name + suffix)
    save_df.to_csv(filename, encoding="utf-8", index=False)
    return save_df


def summary(
    summary_file_path,
    condition_dfs,
    output_directory,
    cell_name_var="cell_name",
    file_suffix=".csv",
):
    """
    Generates a summary csv file, putting all attributes of cells that meet
    certain criteria into a single row.

    :param summary_file_path:
    :param condition_dfs:
    :param output_directory:
    :param cell_name_var:
    :param file_suffix:
    :return:
    """
    filename = filename_from_path(summary_file_path, remove_extension=True)
    logging.info("Saving summary csv for summary file: {}".format(filename))
    summary_params = config_parser.GetSummaryConfig(summary_file_path)
    for idx, section in enumerate(summary_params.conditions):

        condition, parameter, direction, value = get_indv_summary_params(
            summary_params, idx
        )

        df = condition_dfs[condition]

        # query based on operation defined as string
        df = df[str_op(direction)(df[parameter], value)]

        condition_dfs[condition] = df

    if summary_params.parameters_keep:
        params_keep = summary_params.parameters_keep
        params_keep.append("cell_name")
        for df in condition_dfs:
            condition_dfs[df] = condition_dfs[df].filter(items=params_keep)

    for condition_df in condition_dfs:
        condition_suffix = "_" + condition_df
        df = condition_dfs[condition_df]
        df.columns = [
            "{}{}".format(c, "" if c in cell_name_var else condition_suffix)
            for c in df.columns
        ]

    for idx, condition_df in enumerate(condition_dfs):
        if idx == 0:
            df = condition_dfs[condition_df]
        else:
            df = pd.merge(df, condition_dfs[condition_df], on=[cell_name_var])

    out_filename = os.path.join(output_directory, filename + file_suffix)
    df.to_csv(out_filename, encoding="utf-8", index=False)


def plot(
    total_df,
    cell_specific_data,
    cell_specific_stats,
    hd,
    ahv,
    velocity,
    place,
    colors,
    camera_frames_per_sec,
    options,
    condition_name=None,
):
    plotting.plot(
        total_df,
        cell_specific_data,
        cell_specific_stats,
        hd,
        ahv,
        velocity,
        place,
        colors,
        camera_frames_per_sec,
        options,
        condition_name=condition_name,
    )


def get_indv_summary_params(summary_params, idx):
    condition = summary_params.conditions[idx]
    parameter = summary_params.parameters[idx]
    direction = summary_params.directions[idx]
    value = summary_params.values[idx]

    return condition, parameter, direction, value
