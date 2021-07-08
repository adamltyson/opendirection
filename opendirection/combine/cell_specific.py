import numpy as np
import logging


from imlib.radial.distributions import vonmises_kde
from imlib.array.misc import sanitise_array, weight_array
from spikey.histogram.radial import radial_spike_histogram_multiple

import opendirection.spikes.tools as spike_tools
from opendirection.combine.clean import clean_df


class CellSpecific:
    def __init__(self, df, hd, ahv, velocity, place, cell_list, options):
        # TODO: check what self.ahv_* are - don't seem to be one per cell
        # TODO: organise by vel, ahv, place etc
        self.bin_size = options.direction_bin
        self.cell_list = cell_list

        self.hd_list_angles = []
        self.hd_weights = []
        self.hd_hist_bar_centers = []
        self.hd_spikes_per_bin = []
        self.hd_list_angles_weighted = []

        self.hd_kde_centers = []
        self.hd_kde_spikes = []

        self.ahv_cell_spikes_freq = []
        self.ahv_centers_in_range = []  # bin centers
        self.ahv_bin_times_in_range = []
        self.ahv_shuffled_binned_data = []

        self.velocity_cell_spikes_freq = []
        self.velocity_centers_in_range = []
        self.velocity_bin_times_in_range = []
        self.velocity_hist_centers = []
        self.velocity_shuffled_binned_data = []

        self.place_cell_spikes_freq = []

        self.calc_all(df, hd, ahv, velocity, place, options)

    def calc_all(self, df, hd, ahv, velocity, place, options):
        self.calc_hd(df, hd, options)
        self.calc_ahv(df, ahv, options)
        self.calc_velocity(df, velocity, options)
        self.calc_place(df, place, options)

    def calc_hd(self, df, hd, options):
        df = clean_df(df, speed_cutoff=options.hd_speed_cut_off, copy=True)
        self.get_firing_directions(df)
        self.get_direction_firing_rates_bins(
            hd, smooth_width=options.hd_smooth_sigma
        )
        self.direction_firing_kde(kappa=options.hd_kde_kappa)
        self.kde_normalise(hd.time_spent_each_head_angle_kde)

    def calc_ahv(self, df, ahv, options):
        df = clean_df(df, speed_cutoff=options.ahv_speed_cut_off, copy=True)
        self.get_ahv_firing_rates_bins(df, ahv)

    def calc_velocity(self, df, velocity, options):
        df = clean_df(
            df, speed_cutoff=options.velocity_speed_cut_off, copy=True
        )
        self.get_velocity_firing_rates_bins(df, velocity)

    def calc_place(self, df, place, options):
        df = clean_df(df, speed_cutoff=options.place_speed_cut_off, copy=True)
        self.get_place_firing_bins(
            df, place, min_time_in_spatial_bin=options.min_time_in_spatial_bin
        )

    def get_firing_directions(self, df):
        (
            self.hd_list_angles,
            self.hd_weights,
        ) = spike_tools.get_direction_per_spike(
            df, self.cell_list, "absolute_head_angle", threshold=1
        )

    def get_direction_firing_rates_bins(self, hd, smooth_width=None):
        (
            self.hd_spikes_per_bin,
            self.hd_hist_bar_centers,
        ) = radial_spike_histogram_multiple(
            self.hd_list_angles,
            self.hd_weights,
            bin_width=self.bin_size,
            bin_occupancy=hd.time_spent_each_head_angle_hist,
            smooth_width=smooth_width,
        )

        # if smooth_width is not None:
        #     smooth_width_sigma = int(round(smooth_width / self.bin_size))
        #     # if the smooth width is less than the bin size, set it to
        #     # the bin size
        #     if smooth_width_sigma < 1:
        #         smooth_width_sigma = 1
        #     self.hd_spikes_per_bin = filters.gaussian_filter1d(
        #         self.hd_spikes_per_bin, smooth_width_sigma, mode="wrap"
        #     )

    def direction_firing_kde(self, kappa=200, n_bins=None):
        for idx, list_angles in enumerate(self.hd_list_angles):
            weights = self.hd_weights[idx]

            if n_bins is None:
                n_bins = int(round(360 / self.bin_size))

            angles_weighted = weight_array(list_angles, weights)
            self.hd_list_angles_weighted.append(angles_weighted)
            bins, kde = vonmises_kde(
                np.deg2rad(angles_weighted), kappa, n_bins
            )
            self.hd_kde_spikes.append(kde)
            self.hd_kde_centers.append(bins)

    def kde_normalise(self, hd_total_kde):
        for idx, kde in enumerate(self.hd_kde_spikes):
            self.hd_kde_spikes[idx] = np.divide(kde, hd_total_kde)

    def get_ahv_firing_rates_bins(
        self, df, ahv, query="angular_head_velocity"
    ):
        logging.info("Calculating firing rates per AHV bin")
        cntrs = ahv.ahv_hist_centers
        ahv_min = ahv.min_ahv - (ahv.bin_size / 2)
        ahv_max = ahv.max_ahv + (ahv.bin_size / 2)

        # find centers that meet both criteria
        self.ahv_centers_in_range = cntrs[
            (cntrs > ahv_min) * (cntrs < ahv_max)
        ]
        self.ahv_bin_times_in_range = ahv.ahv_hist_seconds[
            (cntrs > ahv_min) * (cntrs < ahv_max)
        ]  # in s

        for cell in self.cell_list:
            logging.debug("Calculating firing rates for {}".format(cell))
            spike_freq = spike_tools.binned_spikes_from_train(
                df[cell],
                self.ahv_centers_in_range,
                ahv.bin_size,
                df[query],
                bin_times_in_range=self.ahv_bin_times_in_range,
            )
            self.ahv_cell_spikes_freq.append(spike_freq)

    def get_velocity_firing_rates_bins(
        self, df, velocity, sanitise_values=True, query="total_speed"
    ):
        logging.info("Calculating firing rates per velocity bin")
        # to match ahv
        bin_centers = velocity.velocity_hist_centers
        min_vel = velocity.min_velocity
        max_vel = velocity.max_velocity

        self.velocity_centers_in_range = bin_centers[
            (bin_centers > min_vel) * (bin_centers < max_vel)
        ]
        self.velocity_bin_times_in_range = velocity.velocity_hist_seconds[
            (bin_centers > min_vel) * (bin_centers < max_vel)
        ]  # in s

        for cell in self.cell_list:
            logging.debug("Calculating firing rates for {}".format(cell))
            spike_freq = self.bin_spike_trains(
                df,
                cell,
                velocity.bin_size,
                sanitise_values=sanitise_values,
                query=query,
            )
            self.velocity_cell_spikes_freq.append(spike_freq)

    def bin_spike_trains(
        self, df, cell, bin_size, sanitise_values=True, query="total_speed"
    ):
        spike_freq = spike_tools.binned_spikes_from_train(
            df[cell],
            self.velocity_centers_in_range,
            bin_size,
            df[query],
            bin_times_in_range=self.velocity_bin_times_in_range,
        )
        if sanitise_values:
            spike_freq = sanitise_array(
                spike_freq, extreme_multiplier=10 ** 10
            )
        return spike_freq

    def get_place_firing_bins(
        self, df, place, sanitise_values=True, min_time_in_spatial_bin=0
    ):
        logging.info("Calculating firing rates per spatial bin")
        # TODO: add functionality to decide what bins to use
        bin_centers_x = place.x_bin_centers
        bin_centers_y = place.y_bin_centers

        for cell in self.cell_list:
            logging.debug("Calculating firing rates for {}".format(cell))
            spike_freq = spike_tools.place_spike_freq(
                df[cell],
                (bin_centers_x, bin_centers_y),
                place.bin_size,
                place.x,
                place.y,
                sanitise_values=sanitise_values,
                min_time_in_spatial_bin=min_time_in_spatial_bin,
                bin_occupancy=place.place_hist_seconds,
            )
            self.place_cell_spikes_freq.append(spike_freq)
