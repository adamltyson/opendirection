import logging
import os.path

import numpy as np
import multiprocessing as mp

from datetime import datetime
from fancylog import fancylog
from glob import glob


from imlib.radial.temporal import angle_bin_occupancy
from imlib.general.system import (
    get_num_processes,
    ensure_directory_exists,
)

from opendirection.combine.cell_select import get_matching_cells
import opendirection as program_for_log
from opendirection.combine.condition import condition_select
import opendirection.spikes.tools as spike_tools
import opendirection.tools.tools as tools
from opendirection.tools import config_parser, experiment_parser
from opendirection.stats.tools import apply_random_sign
import opendirection.run.run as run
from opendirection.tools.multi_exp_parser import get_args

MIN_PROCESSES = 1
PERCENTILES = [50, 75, 90, 95, 96, 97, 98, 99, 99.5, 99.9, 99.95, 99.99]


def mult_exp_setup():
    args = get_args()
    ensure_directory_exists(args.output_dir)
    options = config_parser.GetOptions(args.options)

    num_processes = get_num_processes(min_free_cpu_cores=args.n_free_cpus)
    options.num_processes = num_processes
    fancylog.start_logging(
        args.output_dir,
        program_for_log,
        variables=[args],
        verbose=args.verbose,
        log_header="OPENDIRECTION MULTI EXPERIMENT LOG",
    )

    experiment_files = glob(os.path.join(args.exp_files, "*.txt"))
    logging.info(f"Found {len(experiment_files)} experiment files")
    experiment_config_list = [
        experiment_parser.GetArgs(experiment_file)
        for experiment_file in experiment_files
    ]

    return args, options, experiment_config_list, num_processes


def shuffle_single_timeseries(
    head_angles_all,
    spike_train,
    head_angle_sampling,
    min_shuffle_dist_time=20,
    max_shuffle_dist_time=0,
    camera_frames_per_sec=None,
    bin_spacing=0.105,
    num_iterations=1000,
    threshold=1,
    smooth_width=None,
):
    """
    :param head_angles_all: Head angle at bin (wrapped)
    :param spike_train: Number of spikes per bin
    :param head_angle_sampling: Relative occupancy of bins
    :param camera_frames_per_sec: Calibration
    :param bin_spacing: In radians
    :param num_iterations: How many shuffling iterations (default: 1000)
    :param threshold: Number of spikes per bin for it to be
    classed as active (default: 1)
    :return:
    """
    # todo: combine with opendirection.spikes.tools.get_direction_per_spike

    head_angles_all = np.array(np.deg2rad(head_angles_all))
    spike_train = np.array(spike_train)

    min_shuffle_dist = tools.shuffle_distances_in_units(
        min_shuffle_dist_time, camera_frames_per_sec, spike_train
    )
    if max_shuffle_dist_time == 0:
        max_shuffle_dist = len(spike_train) - min_shuffle_dist
    else:
        max_shuffle_dist = tools.shuffle_distances_in_units(
            max_shuffle_dist_time, camera_frames_per_sec, spike_train
        )

    angles_w_firing = head_angles_all[(spike_train >= threshold)]
    firing_weighting = spike_train[(spike_train >= threshold)]

    spikes_per_bin, bin_centers = spike_tools.get_spike_hist_single(
        np.rad2deg(angles_w_firing),
        firing_weighting,
        np.rad2deg(bin_spacing),
        head_angle_sampling,
    )

    logging.debug("Running random shuffling")

    vec_length_shuffled = np.empty(num_iterations)
    stability_index_shuffled = np.empty(num_iterations)

    for iteration in range(0, num_iterations):
        logging.debug("Iteration: " + str(iteration))
        rand_shuffle = apply_random_sign(
            np.random.randint(min_shuffle_dist, high=max_shuffle_dist)
        )
        spikes_shuffled = np.roll(spike_train, rand_shuffle)

        (
            vec_length_shuffled[iteration],
            stability_index_shuffled[iteration],
        ) = spike_tools.mean_vec_length_and_stability(
            head_angles_all,
            spikes_shuffled,
            bin_centers,
            bin_spacing,
            head_angle_sampling,
            camera_frames_per_sec,
            smooth_width=smooth_width,
            threshold=threshold,
        )
    return vec_length_shuffled, stability_index_shuffled


def run_all_series(
    args,
    options,
    experiment_config_list,
    bin_spacing_rad,
    angle_query="absolute_head_angle",
):
    vec_lengths = np.array([])
    stability_indices = np.array([])
    for experiment_config in experiment_config_list:
        (
            vec_lengths_single_exp,
            stability_index_single_exp,
        ) = run_single_experiment(
            args,
            options,
            bin_spacing_rad,
            experiment_config,
            angle_query=angle_query,
        )

        vec_lengths = np.append(vec_lengths, vec_lengths_single_exp)
        stability_indices = np.append(
            stability_indices, stability_index_single_exp
        )

    return vec_lengths, stability_indices


def run_single_experiment(
    args,
    options,
    bin_spacing_rad,
    experiment_config,
    angle_query="absolute_head_angle",
):

    logging.info(f"Analysing experiment: {experiment_config.experiment_name}")
    vec_lengths = np.array([])
    stability_indices = np.array([])

    experiment_config.verbose = args.verbose
    config = config_parser.GetConfig(experiment_config.config_path)

    total_df = run.load_align_data(experiment_config, options, config)

    cell_list = get_matching_cells(total_df)
    for condition in args.conditions:
        logging.info(f"Analysing condition: {condition}")
        df = condition_select(total_df, args.conditions, condition)
        (time_spent_each_head_angle_hist, _,) = angle_bin_occupancy(
            df[angle_query],
            config.camera_frames_per_sec,
            bin_size=options.direction_bin,
        )
        for cell_name in cell_list:
            logging.info(f"Analysing cell: {cell_name}")

            (
                vec_length_shuffled,
                stability_index_shuffled,
            ) = shuffle_single_timeseries(
                df[angle_query],
                df[cell_name],
                time_spent_each_head_angle_hist,
                min_shuffle_dist_time=options.multiexp_shuffle_min_magnitude,
                max_shuffle_dist_time=options.multiexp_shuffle_max_magnitude,
                num_iterations=options.multiexp_shuffle_iterations,
                bin_spacing=bin_spacing_rad,
                camera_frames_per_sec=config.camera_frames_per_sec,
                smooth_width=options.hd_smooth_sigma,
            )
            vec_lengths = np.append(vec_lengths, vec_length_shuffled)
            stability_indices = np.append(
                stability_indices, stability_index_shuffled
            )
    return vec_lengths, stability_indices


def run_single_experiment_parallel(
    args,
    options,
    bin_spacing_rad,
    experiment_config,
    output,
    angle_query="absolute_head_angle",
):
    logging.info(f"Analysing experiment: {experiment_config.experiment_name}")
    vec_lengths = np.array([])
    stability_indices = np.array([])

    experiment_config.verbose = args.verbose
    config = config_parser.GetConfig(experiment_config.config_path)

    total_df = run.load_align_data(experiment_config, options, config)

    cell_list = get_matching_cells(total_df)
    for condition in args.conditions:
        logging.info(f"Analysing condition: {condition}")
        df = condition_select(total_df, args.conditions, condition)
        (time_spent_each_head_angle_hist, _,) = angle_bin_occupancy(
            df[angle_query],
            config.camera_frames_per_sec,
            bin_size=options.direction_bin,
        )
        for cell_name in cell_list:
            logging.info(f"Analysing cell: {cell_name}")

            (
                vec_length_shuffled,
                stability_index_shuffled,
            ) = shuffle_single_timeseries(
                df[angle_query],
                df[cell_name],
                time_spent_each_head_angle_hist,
                min_shuffle_dist_time=options.multiexp_shuffle_min_magnitude,
                max_shuffle_dist_time=options.multiexp_shuffle_max_magnitude,
                num_iterations=options.multiexp_shuffle_iterations,
                bin_spacing=bin_spacing_rad,
                camera_frames_per_sec=config.camera_frames_per_sec,
                smooth_width=options.hd_smooth_sigma,
            )
            vec_lengths = np.append(vec_lengths, vec_length_shuffled)
            stability_indices = np.append(
                stability_indices, stability_index_shuffled
            )
    output.put([vec_lengths, stability_indices])


def run_all_parallel(
    args,
    options,
    experiment_config_list,
    bin_spacing_rad,
    angle_query="absolute_head_angle",
    num_processes=10,
):

    output = mp.Queue()
    num_processes = min(num_processes, len(experiment_config_list))
    processes = [
        mp.Process(
            target=run_single_experiment_parallel,
            args=(
                args,
                options,
                bin_spacing_rad,
                experiment_config_list[process],
                output,
                angle_query,
            ),
        )
        for process in range(0, num_processes)
    ]

    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    output_tmp = [output.get() for p in processes]

    vec_lengths = output_tmp[0][0]
    stability_indices = output_tmp[0][1]
    for i in range(1, len(output_tmp)):
        vec_lengths = np.append(vec_lengths, output_tmp[i][0])
        stability_indices = np.append(stability_indices, output_tmp[i][1])

    return vec_lengths, stability_indices


def main(angle_query="absolute_head_angle"):
    start_time = datetime.now()
    args, options, experiment_config_list, num_processes = mult_exp_setup()

    # TODO: define the size of these at the beginning

    bin_spacing_rad = np.deg2rad(options.direction_bin)

    vec_lengths, stability_indices = run_all_series(
        args,
        options,
        experiment_config_list,
        bin_spacing_rad,
        angle_query=angle_query,
    )
    # vec_lengths, stability_indices = run_all_parallel(
    #     args,
    #     options,
    #     experiment_config_list,
    #     bin_spacing_rad,
    #     angle_query=angle_query,
    # )

    vec_length_percentiles = np.array([])
    stability_index_percentiles = np.array([])
    for percentiles in PERCENTILES:
        vec_length_percentiles = np.append(
            vec_length_percentiles, np.percentile(vec_lengths, percentiles)
        )
        # taking absolute because can be negative
        stability_index_percentiles = np.append(
            stability_index_percentiles,
            np.percentile(abs(stability_indices), percentiles),
        )

    logging.info(f"Percentiles calculated: {PERCENTILES}")
    logging.info(f"Mean vector length: {vec_length_percentiles}")
    logging.info(f"Stability index: {stability_index_percentiles}")

    logging.info(
        "Finished calculations. Total time taken: %s",
        datetime.now() - start_time,
    )

    fancylog.disable_logging()


if __name__ == "__main__":
    main()
