import opendirection.plot.setup as plot_setup
import opendirection.spikes.tools as spike_tools


def get_matching_cells(
    df, options=None, time_spent_each_head_angle=None, bin_size=6
):

    # TODO: add error - "No cells meet firing criteria"
    cells = [column for column in df.columns if "ell" in column]

    if options is not None:
        if options.min_fire_freq is not None:
            min_firing_frequency = options.min_fire_freq
            cells = spike_tools.get_firing_cells(
                df, cells, min_firing_frequency=min_firing_frequency
            )

        if options.min_fire_freq_per_direction_bin is not None:
            min_hz_bin = options.min_fire_freq_per_direction_bin
            cells = plot_setup.check_min_firing_per_bin(
                df, cells, bin_size, time_spent_each_head_angle, min_hz_bin
            )

    return cells
