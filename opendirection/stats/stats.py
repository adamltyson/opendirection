import numpy as np
import pycircstat
import logging

import scipy.ndimage.filters as filters
import scipy.stats as stats

from imlib.radial.misc import opposite_angle
from imlib.array.locate import peak_nd_array
from imlib.array.fit import max_polyfit

from spikey.descriptive.radial import radial_tuning_stability

import opendirection.stats.tools as stats_tools

from opendirection.combine.calculate_general import get_positions
from opendirection.combine.clean import clean_df


class Stats:
    def __init__(
        self,
        cell_specific_data,
        df,
        hd_behaviour,
        place,
        cell_name,
        options,
        config,
    ):
        self.cell_name = cell_name
        logging.info("Calculating statistics for cell: " + self.cell_name)
        logging.debug("Calculating head direction statistics")
        self.hd = HeadDirectionStats(
            cell_specific_data, df, hd_behaviour, cell_name, options, config
        )

        logging.debug("Calculating angular head velocity statistics")
        self.ahv = AHVStats(cell_specific_data, df, cell_name, options, config)

        logging.debug("Calculating velocity statistics")
        self.velocity = VelocityStats(
            cell_specific_data, df, cell_name, options, config
        )

        logging.debug("Calculating place statistics")
        self.place = PlaceStats(
            cell_specific_data, df, place, cell_name, options, config
        )


class HeadDirectionStats:
    # TODO: clean up once cell_specific_data has been refactored
    #  to cell specificity
    def __init__(
        self, cell_specific_data, df, hd_behaviour, cell_name, options, config
    ):
        df = clean_df(df, speed_cutoff=options.hd_speed_cut_off, copy=True)
        self.mean_vec_percentile = []
        self.hd_stability_percentile = []
        self.rayleigh_test_p = []
        self.rayleigh_test_z = []
        self.omnibus_test_p = []
        self.omnibus_test_m = []
        self.mean_vec_length = []
        self.mean_direction = []
        self.hd_background_hz = []
        self.hd_peak_hz = []
        self.hd_preferred_firing_angle = []
        self.hd_snr = []
        self.stability_index = []
        self.cell_name = cell_name

        idx = get_idx(cell_specific_data, cell_name)
        bins, spikes_per_bin = self.get_angles(cell_specific_data, idx)
        head_angle_sampling = hd_behaviour.time_spent_each_head_angle_hist
        bin_spacing_rad = np.deg2rad(options.direction_bin)
        self.descriptive(bins, spikes_per_bin, bin_spacing_rad)
        self.firing(
            cell_specific_data,
            idx,
            df,
            options,
            config,
            calc_hd_peak_method=options.calc_hd_peak_method,
            smooth_width=options.calc_hd_hist_smooth_width,
            baseline_bin=options.direction_baseline_bin,
            bin_size=options.direction_bin,
        )

        self.directionality(bins, spikes_per_bin)

        self.calc_stability_index(
            df,
            cell_name,
            config.camera_frames_per_sec,
            bin_width=options.direction_bin,
            smooth_width=options.hd_smooth_sigma,
        )
        if options.hd_shuffle_test:
            self.hd_shuffled_stats(
                df,
                cell_name,
                head_angle_sampling,
                min_shuffle_dist_time=options.hd_shuffle_min_magnitude,
                max_shuffle_dist_time=options.hd_shuffle_max_magnitude,
                num_iterations=options.hd_shuffle_iterations,
                bin_spacing=bin_spacing_rad,
                camera_frames_per_sec=config.camera_frames_per_sec,
                num_processes=options.num_processes,
                parallel=options.parallel,
                smooth_width=options.hd_smooth_sigma,
            )

    def firing(
        self,
        all_cells,
        idx,
        df,
        options,
        config,
        calc_hd_peak_method="hist",
        smooth_width=None,
        baseline_bin=60,
        bin_size=6,
    ):
        logging.debug(
            "Calculating head direction firing properties "
            "of cell: " + self.cell_name
        )

        if calc_hd_peak_method == "hist":
            self.peak_firing_rate_calc_hist(
                all_cells, idx, smooth_width=smooth_width, bin_size=bin_size
            )
            self.get_baseline_hist(
                all_cells, idx, baseline_bin=baseline_bin, bin_size=bin_size
            )
        else:
            if calc_hd_peak_method != "mean":
                logging.warning(
                    "No peak firing option given, " "defaulting to ''mean'"
                )
            self.peak_firing_rate_calc_mean(df, options, config)
            self.background_firing_rate_calc_mean(df, options, config)

        self.get_snr()

    def background_firing_rate_calc_mean(self, df, options, config):
        """
        Uses 180 + mean firing angle to generate a "new" bin around this angle
        to calculate the background firing rate
        :param df:
        :param options:
        :param config:
        :return:
        """

        self.hd_preferred_firing_angle = self.mean_direction
        non_preferred_angle = opposite_angle(self.hd_preferred_firing_angle)

        min_angle = non_preferred_angle - (options.direction_baseline_bin / 2)
        max_angle = non_preferred_angle + (options.direction_baseline_bin / 2)

        spikes_in_ang_range = df[self.cell_name][
            (df["absolute_head_angle"] > min_angle)
            & (df["absolute_head_angle"] < max_angle)
        ]
        num_spikes = spikes_in_ang_range.sum()
        time_in_range = len(spikes_in_ang_range) / config.camera_frames_per_sec
        self.hd_background_hz = num_spikes / time_in_range

    def peak_firing_rate_calc_mean(
        self, df, options, config, hd_query="absolute_head_angle"
    ):
        """
        Uses the mean firing angle to generate a "new" bin around this angle
        to calculate the peak firing rate

        """

        self.hd_preferred_firing_angle = self.mean_direction
        min_angle = self.hd_preferred_firing_angle - (
            options.direction_bin / 2
        )
        max_angle = self.hd_preferred_firing_angle + (
            options.direction_bin / 2
        )

        spikes_in_ang_range = df[self.cell_name][
            (df[hd_query] > min_angle) & (df[hd_query] < max_angle)
        ]
        num_spikes = spikes_in_ang_range.sum()
        time_in_range = len(spikes_in_ang_range) / config.camera_frames_per_sec
        self.hd_peak_hz = num_spikes / time_in_range

    def peak_firing_rate_calc_hist(
        self, all_cells, idx, smooth_width=5, bin_size=6
    ):

        direction_binned_spikes = all_cells.hd_spikes_per_bin[idx]

        if smooth_width is not None:
            smooth_width_sigma = int(round(smooth_width / bin_size))
            # if the smooth width is less than the bin size, set it to
            # the bin size
            if smooth_width_sigma < 1:
                smooth_width_sigma = 1
            direction_binned_spikes = filters.gaussian_filter1d(
                direction_binned_spikes, smooth_width_sigma, mode="wrap"
            )

        peak_fire_idx = np.argmax(direction_binned_spikes)
        self.hd_peak_hz = all_cells.hd_spikes_per_bin[idx][peak_fire_idx]
        self.hd_preferred_firing_angle = np.rad2deg(
            all_cells.hd_hist_bar_centers[peak_fire_idx]
        )

    def get_baseline_hist(self, all_cells, idx, baseline_bin=60, bin_size=6):
        baseline_num = int(round(baseline_bin / bin_size))
        spike_counts = all_cells.hd_spikes_per_bin[idx]
        part = np.argpartition(spike_counts, baseline_num)
        k_smallest = spike_counts[part[:baseline_num]]
        self.hd_background_hz = np.mean(k_smallest)

    def get_snr(self,):
        self.hd_snr = float(
            float(self.hd_peak_hz) / float(self.hd_background_hz)
        )

    def directionality(self, bins, spikes_per_bin):
        self.rayleigh_test_p, self.rayleigh_test_z = pycircstat.tests.rayleigh(
            bins, w=spikes_per_bin
        )
        self.omnibus_test_p, self.omnibus_test_m = pycircstat.tests.omnibus(
            bins, w=spikes_per_bin
        )

    def descriptive(self, bins, spikes_per_bin, bin_spacing):
        self.mean_vec_length = pycircstat.descriptive.resultant_vector_length(
            bins, w=spikes_per_bin, d=bin_spacing
        )
        self.mean_direction = np.rad2deg(
            pycircstat.descriptive.mean(bins, w=spikes_per_bin, d=bin_spacing)
        )

    @staticmethod
    def get_angles(all_cells, idx):
        bins = all_cells.hd_hist_bar_centers
        # spike_frequency doesn't need to be scaled, fractions are fine for
        # pycircstat
        spike_freq = all_cells.hd_spikes_per_bin[idx]
        return bins, spike_freq

    def hd_shuffled_stats(
        self,
        df,
        cell_name,
        head_angle_sampling,
        min_shuffle_dist_time=None,
        max_shuffle_dist_time=None,
        num_iterations=1000,
        camera_frames_per_sec=40,
        bin_spacing=0.105,
        angle_query="absolute_head_angle",
        num_processes=10,
        parallel=False,
        smooth_width=None,
    ):
        logging.info("Testing HD significance for cell: " + cell_name)
        head_angles = df[angle_query]
        spike_train = df[cell_name]
        (
            self.mean_vec_percentile,
            self.hd_stability_percentile,
        ) = stats_tools.run_hd_shuffled_stats(
            self.mean_vec_length,
            self.stability_index,
            head_angles,
            spike_train,
            head_angle_sampling,
            min_shuffle_dist_time=min_shuffle_dist_time,
            max_shuffle_dist_time=max_shuffle_dist_time,
            num_iterations=num_iterations,
            bin_spacing=bin_spacing,
            camera_frames_per_sec=camera_frames_per_sec,
            num_processes=num_processes,
            parallel=parallel,
            smooth_width=smooth_width,
        )

    def calc_stability_index(
        self,
        df,
        cell_name,
        frames_per_sec,
        bin_width=6,
        angle_query="absolute_head_angle",
        smooth_width=None,
    ):
        """
        Calculate the stabilty index (the correlation between the tuning in
        the first half of the recording and the second)

        :param df: pandas dataframe containing the head angles as a series
        (angle_query), and the spike train (cell_name).
        :param cell_name: String label of the spike train for the individual
        cell
        :param frames_per_sec: How many angle values are recorded each second
        :param bin_width: Size of bin used for histogram
        :param angle_query: String label of the series of head angles
        :param smooth_width: If not None, smooth with a kernel of this size
        """

        logging.info("Testing HD stability for cell: " + cell_name)
        head_angles = df[angle_query].to_numpy()
        spike_train = df[cell_name].to_numpy()

        self.stability_index = radial_tuning_stability(
            head_angles,
            spike_train,
            bin_width,
            frames_per_sec,
            nan_correct=True,
            smooth_width=smooth_width,
        )


class AHVStats:
    def __init__(self, cell_specific_data, df, cell_name, options, config):
        df = clean_df(df, speed_cutoff=options.ahv_speed_cut_off, copy=True)
        self.cell_name = cell_name
        self.ahv_baseline_hz = []
        self.ahv_peak_hz = []
        self.ahv_pearson_r_neg = []
        self.ahv_pearson_p_neg = []
        self.ahv_pearson_r_pos = []
        self.ahv_pearson_p_pos = []
        self.ahv_fit_intercept_neg = []
        self.ahv_fit_intercept_pos = []
        self.ahv_fit_slope_neg = []
        self.ahv_fit_slope_pos = []
        self.pearson_neg_percentile = []
        self.pearson_pos_percentile = []

        idx = get_idx(cell_specific_data, cell_name)

        cw_ccw_firing = GetCWandCCW(cell_specific_data, idx)

        self.firing(df, cw_ccw_firing, options, config)
        self.get_correlation(cw_ccw_firing)
        self.get_fit(cw_ccw_firing)

        if options.ahv_shuffle_test:
            self.get_ahv_cell_sig(
                cell_specific_data,
                df,
                cell_name,
                min_shuffle_dist_time=options.ahv_shuffle_min_magnitude,
                max_shuffle_dist_time=options.ahv_shuffle_max_magnitude,
                num_iterations=options.ahv_shuffle_iterations,
                camera_frames_per_sec=config.camera_frames_per_sec,
                num_processes=options.num_processes,
                correlation_mag_force=options.ahv_correlation_magnitude,
                parallel=options.parallel,
            )

    def firing(self, df, cw_ccw_firing, options, config):
        logging.debug(
            "Calculating ahv firing properties " "of cell: " + self.cell_name
        )
        self.get_baseline(
            df, options.ang_vel_bin_size, config.camera_frames_per_sec
        )

        self.get_peak(cw_ccw_firing)

    def get_baseline(
        self,
        df,
        ang_vel_bin_size,
        camera_hz,
        ahv_query="angular_head_velocity",
    ):
        logging.debug("Calculating ahv baseline")
        min_ahv = -ang_vel_bin_size
        max_ahv = ang_vel_bin_size
        spikes_in_ahv_range = df[self.cell_name][
            (df[ahv_query] > min_ahv) & (df[ahv_query] < max_ahv)
        ]
        num_spikes = spikes_in_ahv_range.sum()
        time_in_range = len(spikes_in_ahv_range) / camera_hz
        self.ahv_baseline_hz = num_spikes / time_in_range

    def get_peak(self, cw_ccw_firing, polyfit_deg=3):
        logging.debug("Calculating ahv peak")

        max_neg = max_polyfit(
            cw_ccw_firing.x_neg, cw_ccw_firing.y_neg, fit_degree=polyfit_deg
        )
        max_pos = max_polyfit(
            cw_ccw_firing.x_pos, cw_ccw_firing.y_pos, fit_degree=polyfit_deg
        )

        self.ahv_peak_hz = max(max_neg, max_pos)

    def get_correlation(self, cw_ccw_firing):
        self.ahv_pearson_r_neg, self.ahv_pearson_p_neg = stats.pearsonr(
            cw_ccw_firing.x_neg, cw_ccw_firing.y_neg
        )
        self.ahv_pearson_r_pos, self.ahv_pearson_p_pos = stats.pearsonr(
            cw_ccw_firing.x_pos, cw_ccw_firing.y_pos
        )

    def get_fit(self, cw_ccw_firing, degree=1):
        neg_coef = np.polyfit(cw_ccw_firing.x_neg, cw_ccw_firing.y_neg, degree)
        pos_coef = np.polyfit(cw_ccw_firing.x_pos, cw_ccw_firing.y_pos, degree)

        self.ahv_fit_intercept_neg = neg_coef[1]
        self.ahv_fit_slope_neg = neg_coef[0]
        self.ahv_fit_intercept_pos = pos_coef[1]
        self.ahv_fit_slope_pos = pos_coef[0]

    def get_ahv_cell_sig(
        self,
        all_cells,
        df,
        cell_name,
        min_shuffle_dist_time=None,
        max_shuffle_dist_time=None,
        camera_frames_per_sec=40,
        num_iterations=1000,
        query="angular_head_velocity",
        num_processes=10,
        correlation_mag_force=True,
        parallel=False,
    ):
        logging.info("Testing AHV significance for cell: " + cell_name)
        spike_train = df[cell_name]
        ahv_vals_timecourse = df[query]

        (
            self.pearson_neg_percentile,
            self.pearson_pos_percentile,
        ) = stats_tools.is_ahv_cell_sig(
            self.ahv_pearson_r_neg,
            self.ahv_pearson_r_pos,
            all_cells.ahv_centers_in_range,
            spike_train,
            ahv_vals_timecourse,
            all_cells.ahv_bin_times_in_range,
            num_processes=num_processes,
            min_shuffle_dist_time=min_shuffle_dist_time,
            max_shuffle_dist_time=max_shuffle_dist_time,
            num_iterations=num_iterations,
            camera_frames_per_sec=camera_frames_per_sec,
            correlation_mag_force=correlation_mag_force,
            parallel=parallel,
        )


class VelocityStats:
    def __init__(self, cell_specific_data, df, cell_name, options, config):
        df = clean_df(
            df, speed_cutoff=options.velocity_speed_cut_off, copy=True
        )

        self.velocity_pearson_r = []
        self.velocity_pearson_p = []
        self.pearson_percentile = []
        self.velocity_fit_intercept = []
        self.velocity_fit_slope = []

        idx = get_idx(cell_specific_data, cell_name)

        self.get_correlations(idx, cell_specific_data)
        self.get_fit(idx, cell_specific_data)

        if options.velocity_shuffle_test:
            self.get_velocity_cell_sig(
                cell_specific_data,
                df,
                cell_name,
                min_shuffle_dist_time=options.velocity_shuffle_min_magnitude,
                max_shuffle_dist_time=options.velocity_shuffle_max_magnitude,
                num_iterations=options.velocity_shuffle_iterations,
                camera_frames_per_sec=config.camera_frames_per_sec,
                num_processes=options.num_processes,
                parallel=options.parallel,
                correlation_mag_force=options.velocity_correlation_magnitude,
            )

    def get_correlations(self, idx, all_cells):
        self.velocity_pearson_r, self.velocity_pearson_p = stats.pearsonr(
            all_cells.velocity_centers_in_range,
            all_cells.velocity_cell_spikes_freq[idx],
        )


    def get_velocity_cell_sig(
        self,
        all_cells,
        df,
        cell_name,
        min_shuffle_dist_time=None,
        max_shuffle_dist_time=None,
        camera_frames_per_sec=40,
        num_iterations=1000,
        query="total_speed",
        num_processes=10,
        parallel=False,
        correlation_mag_force=False,
    ):
        logging.info("Testing velocity significance for cell: " + cell_name)
        spike_train = df[cell_name]
        velocity_vals_timecourse = df[query]

        self.pearson_percentile = stats_tools.is_velocity_cell_sig(
            self.velocity_pearson_r,
            all_cells.velocity_centers_in_range,
            spike_train,
            velocity_vals_timecourse,
            all_cells.velocity_bin_times_in_range,
            num_processes=num_processes,
            min_shuffle_dist_time=min_shuffle_dist_time,
            max_shuffle_dist_time=max_shuffle_dist_time,
            num_iterations=num_iterations,
            camera_frames_per_sec=camera_frames_per_sec,
            parallel=parallel,
            correlation_mag_force=correlation_mag_force,
        )

    def get_fit(self,idx, all_cells, degree=1):
        coef = np.polyfit(all_cells.velocity_centers_in_range,
                          all_cells.velocity_cell_spikes_freq[idx],
                          degree)

        self.velocity_fit_intercept = coef[1]
        self.velocity_fit_slope = coef[0]


class PlaceStats:
    def __init__(
        self, cell_specific_data, df, place, cell_name, options, config
    ):
        df = clean_df(df, speed_cutoff=options.place_speed_cut_off, copy=True)
        idx = get_idx(cell_specific_data, cell_name)
        self._smoothing = options.place_firing_smooth_width
        self._min_time_in_spatial_bin = options.min_time_in_spatial_bin
        self.place_peak_magnitude = []
        self.peak_percentile = []

        self.place_firing = cell_specific_data.place_cell_spikes_freq[idx]

        self.get_peak_firing_magnitude()

        if options.place_shuffle_test:
            self.get_place_cell_sig(
                df,
                place,
                cell_name,
                options,
                config,
                min_shuffle_dist_time=options.place_shuffle_min_magnitude,
                max_shuffle_dist_time=options.place_shuffle_max_magnitude,
                num_iterations=options.place_shuffle_iterations,
                camera_frames_per_sec=config.camera_frames_per_sec,
                num_processes=options.num_processes,
                parallel=options.parallel,
            )

    def smooth_place_firing(self):
        if self._smoothing is not None:
            self.place_firing = filters.gaussian_filter(
                self.place_firing, self._smoothing
            )

    def get_peak_firing_magnitude(self):
        self.place_peak_magnitude = peak_nd_array(
            self.place_firing, smoothing_kernel=self._smoothing
        )

    def get_place_cell_sig(
        self,
        df,
        place,
        cell_name,
        options,
        config,
        min_shuffle_dist_time=None,
        max_shuffle_dist_time=None,
        camera_frames_per_sec=40,
        num_iterations=1000,
        num_processes=10,
        parallel=False,
    ):
        logging.info("Testing place significance for cell: " + cell_name)
        spike_train = df[cell_name]
        x, y = get_positions(
            df, use_head_as_position=options.spatial_position_head
        )
        x = x * config.meters_per_pixel
        y = y * config.meters_per_pixel

        self.peak_percentile = stats_tools.is_place_cell_sig(
            self.place_peak_magnitude,
            (place.x_bin_centers, place.y_bin_centers),
            spike_train,
            x,
            y,
            place.place_hist_seconds,
            smoothing=self._smoothing,
            min_time_in_spatial_bin=self._min_time_in_spatial_bin,
            sanitise_values=True,
            num_processes=num_processes,
            min_shuffle_dist_time=min_shuffle_dist_time,
            max_shuffle_dist_time=max_shuffle_dist_time,
            num_iterations=num_iterations,
            camera_frames_per_sec=camera_frames_per_sec,
            parallel=parallel,
        )


def get_idx(all_cells, cell_name):
    return all_cells.cell_list.index(cell_name)


class GetCWandCCW:
    def __init__(self, all_cells, idx):
        self.x_neg = all_cells.ahv_centers_in_range[
            all_cells.ahv_centers_in_range <= 0
        ]
        self.x_pos = all_cells.ahv_centers_in_range[
            all_cells.ahv_centers_in_range >= 0
        ]

        self.y_neg = all_cells.ahv_cell_spikes_freq[idx][
            all_cells.ahv_centers_in_range <= 0
        ]
        self.y_pos = all_cells.ahv_cell_spikes_freq[idx][
            all_cells.ahv_centers_in_range >= 0
        ]
