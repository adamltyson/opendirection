import numpy as np
import pycircstat
import scipy.stats as stats

from imlib.array.misc import (
    sanitise_array,
    mask_array_by_array_val,
    midpoints_of_series,
)
from imlib.array.locate import peak_nd_array
from spikey.descriptive.radial import radial_tuning_stability


def get_firing_cells(df, cells, min_firing_frequency=1):
    """
    Only returns the cells that fire over a certain frequency
    (across entire recording)
    :param df:
    :param cells:
    :param min_firing_frequency: in hz
    :return: Cells that meet the criteria
    """
    total_time = df.index.max() - df.index.min()
    min_firing_total = min_firing_frequency * total_time

    new_cells = []
    for cell in cells:
        total = df[cell].sum()
        if total > min_firing_total:
            new_cells.append(cell)

    return new_cells


def get_direction_per_spike(df, cells, value_query, threshold=1):
    """
    For a list of cells in a dataframe, return a list of values that are
    associated with spiking activity.
    :param df: Dataframe containing the spiking activity, and the values
    to be queried
    :param cells: List of cells (dataframe column headings)
    :param value_query: Dataframe column heading containing values
    :param threshold: Spiking threshold to be reached. Default: 1 spike per bin
    :return: List of values, and their weights (if there are >1 spike per
    temporal bin)
    """
    value_list = []
    weights = []
    for cell in cells:
        values = df[value_query][(df[cell] >= threshold)]
        weights_cell = df[cell][(df[cell] >= threshold)]
        value_list.append(values)
        weights.append(weights_cell)
    return value_list, weights


def get_spike_hist_single(
    angles_w_firing, firing_weighting, bin_spacing, head_angle_sampling
):
    spikes_per_bin, b = np.histogram(
        angles_w_firing,
        weights=firing_weighting,
        bins=np.arange(0, 360 + bin_spacing, bin_spacing),
        density=False,
    )
    spikes_per_bin = np.divide(spikes_per_bin, head_angle_sampling)
    bin_centers = np.deg2rad(midpoints_of_series(b))

    return spikes_per_bin, bin_centers


def binned_spikes_from_train(
    spike_train,
    bin_centers,
    bin_sizes,
    vals,
    bin_times_in_range=None,
    remove_nan=True,
):
    """
    For the ND case, when you have bins in x (also maybe y) ,
    calculate the spike frequency for each bin.
    :param spike_train: (maybe tuple of) 1D spike values in each bin
    (e.g. 0,0,0,1,0,1,2,0)
    :param bin_centers: (maybe tuple of) 1D arrays of bin centers
    :param bin_sizes: (maybe tuple of) bin length scalars
    :param vals: (maybe tuple of) 1D arrays of the value that's binned
    :param bin_times_in_range: ND array of length of time
    (in seconds) spent in each bin, to return in hz
    :param remove_nan: should NaN values (due to dividing by 0 occupancy)
     be removed?
    :return: spike_freq: ND array of spike frequencies
    """

    if type(bin_centers) is tuple:
        if len(bin_sizes) == 2:
            spikes = binned_spikes_from_train_2d(
                spike_train, bin_centers, bin_sizes, vals
            )
        else:
            raise NotImplementedError(
                "binned_spikes_from_train not "
                "implemented for dimensions > 2"
            )
    else:
        spikes = binned_spikes_from_train_1d(
            spike_train, bin_centers, bin_sizes, vals
        )

    if bin_times_in_range is not None:
        spike_freq = np.divide(spikes, bin_times_in_range)  # hz

    if remove_nan:
        spike_freq = np.nan_to_num(spike_freq)
    return spike_freq


def binned_spikes_from_train_1d(spike_train, bin_centers, bin_size, vals):
    """
    For the 1D case, when you have bins in x, count the spikes for each bin.
    :param spike_train: 1D spike values in each bin (e.g. 0,0,0,1,0,1,2,0)
    :param bin_centers: 1D array of bin centers
    :param bin_size: bin length scalar
    :param vals: 1D array of the value that's binned
    (in seconds) spent in each bin, to return in hz
    :return: 1D array of spike counts
    """

    spikes = []
    for cntr in bin_centers:
        bin_min = cntr - (bin_size / 2)
        bin_max = cntr + (bin_size / 2)
        spikes_in_bin = spike_train[(vals > bin_min) & (vals < bin_max)]

        spikes.append(spikes_in_bin.sum())
    spikes = np.array(spikes)
    return spikes


def binned_spikes_from_train_2d(spike_train, bin_centers, bin_sizes, vals):
    """
    For the 2D case, when you have bins in x & y, count the
    spikes for each bin.
    :param spike_train: 1D spike values in each bin (e.g. 0,0,0,1,0,1,2,0)
    :param bin_centers: tuple of 1D arrays of bin centers
    :param bin_sizes: tuple of bin length scalars
    :param vals: tuple of 1D arrays of the value that's binned
    (in seconds) spent in each bin, to return in hz
    :return: spikes: 2D array of spike counts
    """

    spikes = np.zeros((len(bin_centers[0]), len(bin_centers[1])))
    for i, center_i in enumerate(bin_centers[0]):
        bin_min_i = center_i - (bin_sizes[0] / 2)
        bin_max_i = center_i + (bin_sizes[0] / 2)

        for j, center_j in enumerate(bin_centers[1]):
            bin_min_j = center_j - (bin_sizes[1] / 2)
            bin_max_j = center_j + (bin_sizes[1] / 2)
            spikes_in_bin = spike_train[
                (vals[0] > bin_min_i)
                & (vals[0] < bin_max_i)
                & (vals[1] > bin_min_j)
                & (vals[1] < bin_max_j)
            ]
            spikes[i, j] = spikes_in_bin.sum()
    return spikes


def place_spike_freq(
    spike_train,
    bin_centers,
    bin_size,
    x,
    y,
    sanitise_values=False,
    min_time_in_spatial_bin=0,
    bin_occupancy=None,
):
    spike_freq = binned_spikes_from_train(
        spike_train,
        bin_centers,
        (bin_size, bin_size),
        (x, y),
        bin_times_in_range=bin_occupancy,
    )
    if sanitise_values:
        spike_freq = sanitise_array(
            spike_freq, extreme_multiplier=10 ** 10, exclude_zeros=True
        )
    if min_time_in_spatial_bin > 0:
        spike_freq = mask_array_by_array_val(
            spike_freq, bin_occupancy, min_time_in_spatial_bin
        )
    return spike_freq


def get_correlations(
    spike_train,
    bin_centers,
    value_timecourse,
    bin_times_in_range,
    pos_neg_separate=False,
    sanitise_values=False,
    extreme_multiplier=10 ** 10,
):
    """
    For a given spike train, and given bins, return the correlation of spiking
    in bins with a value_timecourse.
    :param spike_train: Spiking activity in spikes per temporal bin
    (e.g. 0 1 0 0 2 1)
    :param bin_centers: Bins defined by their central values.
    :param value_timecourse: Timecourse of some value to be correlated
    with spiking activity
    :param bin_times_in_range: ND array of length of time
    (in seconds) spent in each bin, to return in hz
    :param pos_neg_separate: Run separate correlations for bin values above
    and below zero
    :param sanitise_values: If True, remove very large values and Infs etc
    :param extreme_multiplier: How big do values need to be to be removed if
    using `sanitise_values=True`
    :return: Pearson correlation coefficient, or tuple of positive/negative
    correlation values
    """

    if type(bin_centers) is tuple:
        raise NotImplementedError("get_correlations requires 1D data")

    bin_size = bin_centers[1] - bin_centers[0]
    spike_freq = binned_spikes_from_train(
        spike_train,
        bin_centers,
        bin_size,
        value_timecourse,
        bin_times_in_range=bin_times_in_range,
    )
    if sanitise_values:
        spike_freq = sanitise_array(
            spike_freq, extreme_multiplier=extreme_multiplier
        )

    if pos_neg_separate:
        x_neg = bin_centers[bin_centers <= 0]
        x_pos = bin_centers[bin_centers >= 0]

        y_neg = spike_freq[bin_centers <= 0]
        y_pos = spike_freq[bin_centers >= 0]

        pearson_r_neg, _ = stats.pearsonr(x_neg, y_neg)
        pearson_r_pos, _ = stats.pearsonr(x_pos, y_pos)

        pearson = pearson_r_neg, pearson_r_pos

    else:
        pearson, _ = stats.pearsonr(bin_centers, spike_freq)

    return pearson


def place_peak_response(
    spike_train,
    bin_centers,
    bin_size,
    x,
    y,
    sanitise_values=False,
    min_time_in_spatial_bin=0,
    smoothing=None,
    bin_occupancy=None,
):
    """
    Calculate the peak firing rate (in hz) of putative place cells.
    :param spike_train: Spiking activity in spikes per temporal bin
    (e.g. 0 1 0 0 2 1)
    :param bin_centers: Tuple of bin centers in x & y
    :param bin_size: Bin size (assuming isotropic)
    :param x: list of x values (in space)
    :param y: list of y values (in space
    :param sanitise_values: If True, remove very large values and Infs etc
    :param min_time_in_spatial_bin: How long must a bin be occupied for,
    for the firing rate in that bin to be calculated
    :param int smoothing: Smoothing kernel size (assumes isotropic)
    :param bin_occupancy: Array of occupancies of bins (in seconds) to
    calculate firing in hz
    :return: peak firing value (in hz)
    """
    if type(bin_size) is tuple:
        raise NotImplementedError(
            "bin_size must be a single value (assuming isotropic bins"
        )
    if type(smoothing) is tuple:
        raise NotImplementedError(
            "smoothing must be a single value (assuming isotropic smoothing"
        )

    spike_freq = place_spike_freq(
        spike_train,
        bin_centers,
        bin_size,
        x,
        y,
        sanitise_values=sanitise_values,
        min_time_in_spatial_bin=min_time_in_spatial_bin,
        bin_occupancy=bin_occupancy,
    )
    peak = peak_nd_array(spike_freq, smoothing_kernel=smoothing)

    return peak


def mean_vec_length_and_stability(
    head_angles,
    spike_train,
    bin_centers,
    bin_spacing,
    head_angle_sampling,
    camera_frames_per_sec,
    smooth_width=None,
    threshold=1,
):
    angles_w_firing = head_angles[(spike_train >= threshold)]
    firing_weighting = spike_train[(spike_train >= threshold)]

    spikes_per_bin, _ = get_spike_hist_single(
        np.rad2deg(angles_w_firing),
        firing_weighting,
        np.rad2deg(bin_spacing),
        head_angle_sampling,
    )

    vec_length = pycircstat.descriptive.resultant_vector_length(
        bin_centers, w=spikes_per_bin, d=bin_spacing
    )

    stability_index = radial_tuning_stability(
        np.rad2deg(head_angles),
        spike_train,
        np.rad2deg(bin_spacing),
        camera_frames_per_sec,
        nan_correct=True,
        smooth_width=smooth_width,
    )

    return vec_length, stability_index
