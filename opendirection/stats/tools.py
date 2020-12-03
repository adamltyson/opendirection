import logging
import random
import numpy as np
import multiprocessing as mp

from scipy.stats import percentileofscore
from imlib.general.system import sanitize_num_processes

import opendirection.spikes.tools as spike_tools
import opendirection.tools.tools as tools

MIN_PROCESSES = 1


def apply_random_sign(number):
    sign = 1 if random.random() < 0.5 else -1
    return sign * number


def generic_parallel_shuffle_test(
    specific_test,
    min_shuffle_dist,
    max_shuffle_dist,
    num_iterations,
    num_processes,
    args,
    two_lists=False,
):
    # Define an output queue
    output = mp.Queue()
    args = (output, *args)  # add output to args
    # array of shuffle magnitudes to go through
    shuffle_dists = np.random.randint(
        min_shuffle_dist, high=max_shuffle_dist, size=num_iterations
    )

    # split up the shuffle magnitudes into blocks for each process
    shuffle_dist_blocks = np.array_split(shuffle_dists, num_processes)

    # Setup a list of processes
    processes = [
        mp.Process(
            target=specific_test, args=(shuffle_dist_blocks[process], *args)
        )
        for process in range(0, num_processes)
    ]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    if two_lists:
        # Get process results from the output queue
        output_tmp = [output.get() for p in processes]

        all_vals_1 = output_tmp[0][0]
        all_vals_2 = output_tmp[0][1]
        for i in range(1, len(output_tmp)):
            all_vals_1 = np.append(all_vals_1, output_tmp[i][0])
            all_vals_2 = np.append(all_vals_2, output_tmp[i][1])

        return all_vals_1, all_vals_2

    else:
        # Get process results from the output queue
        vals_tmp = [output.get() for p in processes]

        all_vals = vals_tmp[0]
        for i in range(1, len(vals_tmp)):
            all_vals = np.append(all_vals, vals_tmp[i])

        return all_vals


def run_hd_shuffled_stats(
    vec_length_real,
    stability_index_real,
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
    parallel=False,
    num_processes=10,
):
    """
    :param vec_length_real: "Real" (i.e. unshuffled) mean vector length
    :param head_angles_all: Head angle at bin (wrapped)
    :param spike_train: Number of spikes per bin
    :param head_angle_sampling: Relative occupancy of bins
    :param camera_frames_per_sec: Calibration
    :param bin_spacing: In radians
    :param num_iterations: How many shuffling iterations (default: 1000)
    :param threshold: Number of spikes per bin for it to be
    classed as active (default: 1)
    :param bool parallel: If true, split up the shuffle iterations across
    multiple CPU cores.
    :param int num_processes: If 'parallel', how many processes to use.
    Default: 10
    :return:
    """
    # todo: combine with opendirection.spikes.tools.get_direction_per_spike

    head_angles_all = np.array(np.deg2rad(head_angles_all))
    spike_train = np.array(spike_train)

    parallel = sanitize_num_processes(
        num_processes, MIN_PROCESSES, parallel=parallel
    )

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
    if parallel:
        args = (
            spike_train,
            bin_centers,
            bin_spacing,
            head_angles_all,
            head_angle_sampling,
            threshold,
            camera_frames_per_sec,
            smooth_width,
        )
        (
            vec_length_shuffled,
            stability_index_shuffled,
        ) = generic_parallel_shuffle_test(
            hd_shuffle_parallel,
            min_shuffle_dist,
            max_shuffle_dist,
            num_iterations,
            num_processes,
            args,
            two_lists=True,
        )

    else:
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

    vec_length_percentile = percentileofscore(
        vec_length_shuffled, vec_length_real
    )

    stability_index_percentile = percentileofscore(
        abs(stability_index_shuffled), abs(stability_index_real)
    )

    return vec_length_percentile, stability_index_percentile


def hd_shuffle_parallel(
    shuffle_dists,
    output,
    spike_train,
    bin_centers,
    bin_spacing,
    head_angles_all,
    head_angle_sampling,
    threshold,
    camera_frames_per_sec,
    smooth_width,
):

    vec_lengths = []
    stability_indices = []

    for i in range(0, len(shuffle_dists)):
        spikes_shuffled = np.roll(
            spike_train, apply_random_sign(shuffle_dists[i])
        )
        (
            vec_length,
            stability_index,
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
        vec_lengths.append(vec_length)
        stability_indices.append(stability_index)

    vec_lengths = np.array(vec_lengths)
    stability_indices = np.array(stability_indices)
    output.put([vec_lengths, stability_indices])


def is_ahv_cell_sig(
    pearson_r_neg_real,
    pearson_r_pos_real,
    bin_centers,
    spike_train,
    ahv_vals_timecourse,
    bin_times_in_range,
    min_shuffle_dist_time=20,
    max_shuffle_dist_time=0,
    camera_frames_per_sec=None,
    num_iterations=1000,
    num_processes=10,
    correlation_mag_force=True,
    parallel=False,
):

    parallel = sanitize_num_processes(
        num_processes, MIN_PROCESSES, parallel=parallel
    )

    min_shuffle_dist = tools.shuffle_distances_in_units(
        min_shuffle_dist_time, camera_frames_per_sec, spike_train
    )
    if max_shuffle_dist_time == 0:
        max_shuffle_dist = len(spike_train) - min_shuffle_dist
    else:
        max_shuffle_dist = tools.shuffle_distances_in_units(
            max_shuffle_dist_time, camera_frames_per_sec, spike_train
        )

    logging.debug("Running random shuffling")
    if parallel:

        args = (
            spike_train,
            bin_centers,
            ahv_vals_timecourse,
            bin_times_in_range,
        )
        pearson_r_neg, pearson_r_pos = generic_parallel_shuffle_test(
            ahv_shuffle_parallel,
            min_shuffle_dist,
            max_shuffle_dist,
            num_iterations,
            num_processes,
            args,
            two_lists=True,
        )

    else:  # if parallel doesn't work
        logging.warning("Running serial shuffling")

        pearson_r_neg = np.empty(num_iterations)
        pearson_r_pos = np.empty(num_iterations)

        for iteration in range(0, num_iterations):
            logging.debug("Iteration: " + str(iteration))
            rand_shuffle = apply_random_sign(
                np.random.randint(min_shuffle_dist, high=max_shuffle_dist)
            )
            spikes_shuffled = np.roll(spike_train, rand_shuffle)
            (
                pearson_r_neg[iteration],
                pearson_r_pos[iteration],
            ) = spike_tools.get_correlations(
                spikes_shuffled,
                bin_centers,
                ahv_vals_timecourse,
                bin_times_in_range,
                pos_neg_separate=True,
            )

    # if only care about magnitude of correlation
    if correlation_mag_force:
        pearson_r_neg = abs(pearson_r_neg)
        pearson_r_pos = abs(pearson_r_pos)

        pearson_r_neg_real = abs(pearson_r_neg_real)
        pearson_r_pos_real = abs(pearson_r_pos_real)

    real_percentile_neg = percentileofscore(pearson_r_neg, pearson_r_neg_real)
    real_percentile_pos = percentileofscore(pearson_r_pos, pearson_r_pos_real)

    return real_percentile_neg, real_percentile_pos


def ahv_shuffle_parallel(
    shuffle_dists,
    output,
    spike_train,
    bin_centers,
    ahv_vals_timecourse,
    bin_times_in_range,
):

    pearson_r_neg = []
    pearson_r_pos = []

    for i in range(0, len(shuffle_dists)):
        spikes_shuffled = np.roll(
            spike_train, apply_random_sign(shuffle_dists[i])
        )
        r_neg, r_pos = spike_tools.get_correlations(
            spikes_shuffled,
            bin_centers,
            ahv_vals_timecourse,
            bin_times_in_range,
            pos_neg_separate=True,
        )
        pearson_r_neg.append(r_neg)
        pearson_r_pos.append(r_pos)

    pearson_r_neg = np.array(pearson_r_neg)
    pearson_r_pos = np.array(pearson_r_pos)
    output.put([pearson_r_neg, pearson_r_pos])


def is_velocity_cell_sig(
    pearson_real,
    bin_centers,
    spike_train,
    velocity_vals_timecourse,
    bin_times_in_range,
    min_shuffle_dist_time=20,
    max_shuffle_dist_time=0,
    camera_frames_per_sec=None,
    num_iterations=1000,
    num_processes=10,
    parallel=False,
    correlation_mag_force=False,
):

    parallel = sanitize_num_processes(
        num_processes, MIN_PROCESSES, parallel=parallel
    )

    min_shuffle_dist = tools.shuffle_distances_in_units(
        min_shuffle_dist_time, camera_frames_per_sec, spike_train
    )
    if max_shuffle_dist_time == 0:
        max_shuffle_dist = len(spike_train) - min_shuffle_dist
    else:
        max_shuffle_dist = tools.shuffle_distances_in_units(
            max_shuffle_dist_time, camera_frames_per_sec, spike_train
        )

    logging.debug("Running random shuffling")
    if parallel:
        args = (
            spike_train,
            bin_centers,
            velocity_vals_timecourse,
            bin_times_in_range,
        )
        pearson = generic_parallel_shuffle_test(
            velocity_shuffle_parallel,
            min_shuffle_dist,
            max_shuffle_dist,
            num_iterations,
            num_processes,
            args,
        )

    else:  # if parallel doesn't work
        logging.warning("Not running serial shuffling")
        pearson = np.empty(num_iterations)

        for iteration in range(0, num_iterations):
            logging.debug("Iteration: " + str(iteration))
            rand_shuffle = apply_random_sign(
                np.random.randint(min_shuffle_dist, high=max_shuffle_dist)
            )
            spikes_shuffled = np.roll(spike_train, rand_shuffle)
            pearson[iteration] = spike_tools.get_correlations(
                spikes_shuffled,
                bin_centers,
                velocity_vals_timecourse,
                bin_times_in_range,
                sanitise_values=True,
            )

    if correlation_mag_force:
        pearson = abs(pearson)
        pearson_real = abs(pearson_real)
    real_percentile_val = percentileofscore(pearson, pearson_real)

    return real_percentile_val


def velocity_shuffle_parallel(
    shuffle_dists,
    output,
    spike_train,
    bin_centers,
    vals_timecourse,
    bin_times_in_range,
):

    pearson = []

    for i in range(0, len(shuffle_dists)):
        spikes_shuffled = np.roll(
            spike_train, apply_random_sign(shuffle_dists[i])
        )
        r = spike_tools.get_correlations(
            spikes_shuffled,
            bin_centers,
            vals_timecourse,
            bin_times_in_range,
            sanitise_values=True,
        )
        pearson.append(r)

    pearson = np.array(pearson)
    output.put(pearson)


def is_place_cell_sig(
    real_peak,
    bin_centers,
    spike_train,
    x,
    y,
    bin_occupancy,
    smoothing=None,
    min_shuffle_dist_time=20,
    max_shuffle_dist_time=0,
    camera_frames_per_sec=None,
    num_iterations=1000,
    num_processes=10,
    sanitise_values=True,
    min_time_in_spatial_bin=0,
    parallel=False,
):

    parallel = sanitize_num_processes(
        num_processes, MIN_PROCESSES, parallel=parallel
    )

    min_shuffle_dist = tools.shuffle_distances_in_units(
        min_shuffle_dist_time, camera_frames_per_sec, spike_train
    )
    if max_shuffle_dist_time == 0:
        max_shuffle_dist = len(spike_train) - min_shuffle_dist
    else:
        max_shuffle_dist = tools.shuffle_distances_in_units(
            max_shuffle_dist_time, camera_frames_per_sec, spike_train
        )

    bin_centers_x, bin_centers_y = bin_centers
    bin_size = bin_centers_x[1] - bin_centers_x[0]

    logging.debug("Running random shuffling")
    if parallel:

        args = (
            spike_train,
            bin_centers,
            bin_size,
            x,
            y,
            bin_occupancy,
            smoothing,
            sanitise_values,
            min_time_in_spatial_bin,
        )
        peaks = generic_parallel_shuffle_test(
            place_shuffle_parallel,
            min_shuffle_dist,
            max_shuffle_dist,
            num_iterations,
            num_processes,
            args,
        )

    else:  # if parallel doesn't work
        logging.warning("Not running parallel shuffling")
        peaks = np.empty(num_iterations)

        for iteration in range(0, num_iterations):
            logging.debug("Iteration: " + str(iteration))
            rand_shuffle = apply_random_sign(
                np.random.randint(min_shuffle_dist, high=max_shuffle_dist)
            )
            spikes_shuffled = np.roll(spike_train, rand_shuffle)

            peaks[iteration] = spike_tools.place_peak_response(
                spikes_shuffled,
                bin_centers,
                bin_size,
                x,
                y,
                sanitise_values=sanitise_values,
                min_time_in_spatial_bin=min_time_in_spatial_bin,
                smoothing=smoothing,
                bin_occupancy=bin_occupancy,
            )

    real_percentile_val = percentileofscore(peaks, real_peak)

    return real_percentile_val


def place_shuffle_parallel(
    shuffle_dists,
    output,
    spike_train,
    bin_centers,
    bin_size,
    x,
    y,
    bin_occupancy,
    smoothing,
    sanitise_values,
    min_time_in_spatial_bin,
):

    peaks = []

    for i in range(0, len(shuffle_dists)):
        spikes_shuffled = np.roll(
            spike_train, apply_random_sign(shuffle_dists[i])
        )

        peak = spike_tools.place_peak_response(
            spikes_shuffled,
            bin_centers,
            bin_size,
            x,
            y,
            sanitise_values=sanitise_values,
            min_time_in_spatial_bin=min_time_in_spatial_bin,
            smoothing=smoothing,
            bin_occupancy=bin_occupancy,
        )
        peaks.append(peak)

    peaks = np.array(peaks)
    output.put(peaks)
