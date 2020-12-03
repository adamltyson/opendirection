import numpy as np

from spikey.histogram.radial import radial_spike_histogram_multiple
import opendirection.spikes.tools as spike_tools


def check_min_firing_per_bin(
    df, cells, bin_size, time_spent_each_head_angle, min_hz_bin
):
    minimum_firing_per_bin = min_hz_bin * time_spent_each_head_angle

    # TODO: how to stop getting angle bins twice? normalise after histogram?

    list_angles, weights = spike_tools.get_direction_per_spike(
        df, cells, "absolute_head_angle", threshold=0
    )

    spikes_per_bin, _ = radial_spike_histogram_multiple(
        list_angles, weights, bin_width=bin_size
    )

    new_cells = []
    for idx, cell in enumerate(cells):
        meet_criteria_per_bin = spikes_per_bin[idx] > minimum_firing_per_bin
        if np.any(meet_criteria_per_bin):
            new_cells.append(cell)

    return new_cells


def get_head_angles_with_firing(
    df,
    query="absolute_head_angle",
    min_firing_per_frame=1,
    min_firing_total=None,
):
    list_angles = []
    cell_names = []
    weights = []
    threshold = min_firing_per_frame - 1
    for column in list(df):
        if "ell" in column:
            degrees = df[query][(df[column] > threshold)]
            weights_cell = df[column][(df[column] > threshold)]
            total_spikes = np.sum(weights_cell)
            if total_spikes > min_firing_total:
                list_angles.append(degrees)
                cell_names.append(column)
                weights.append(weights_cell)
    return list_angles, cell_names, weights


def get_spike_time_list(df, cells):
    all_spikes = []
    threshold = 0
    for column in cells:
        spikes = df[column][(df[column] > threshold)]
        all_spikes.append(spikes)
    spike_array = []
    for spikes in all_spikes:
        spike_array.append(spikes.index.values)
    return spike_array
