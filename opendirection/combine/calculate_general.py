import numpy as np
import logging
import scipy.ndimage.filters as filters

from imlib.pandas.query import column_as_array
from imlib.radial.distributions import vonmises_kde
from imlib.radial.temporal import angle_bin_occupancy
from imlib.array.misc import midpoints_of_series

import opendirection.tools.tools as tools
from opendirection.combine.clean import clean_df


class HeadDirection:
    def __init__(self, df, options, camera_frames_per_sec):
        df = clean_df(df, speed_cutoff=options.hd_speed_cut_off, copy=True)
        self.camera_frames_per_sec = camera_frames_per_sec
        self.bin_size = options.direction_bin
        self.kde_kappa = options.hd_kde_kappa

        self.time_spent_each_head_angle_hist = []
        self.hd_bin_centers_hist = []
        self.time_spent_each_head_angle_kde = []
        self.hd_bin_centers_kde = []

        self.angles = column_as_array(df, "absolute_head_angle")

        # todo: combine bin centers (standardise to a range) ???
        # i.e. combine kde and hist bin numbers?

        self.hd_hist(
            bin_size=options.direction_bin,
            smooth_width=options.hd_smooth_sigma,
        )
        self.hd_kde(kappa=self.kde_kappa)

    def hd_hist(self, bin_size=6, smooth_width=None):
        (
            self.time_spent_each_head_angle_hist,
            self.hd_bin_centers_hist,
        ) = angle_bin_occupancy(
            self.angles, self.camera_frames_per_sec, bin_size=self.bin_size
        )

        if smooth_width is not None:
            smooth_width_sigma = int(round(smooth_width / bin_size))
            # if the smooth width is less than the bin size, set it to
            # the bin size
            if smooth_width_sigma < 1:
                smooth_width_sigma = 1
            self.time_spent_each_head_angle_hist = filters.gaussian_filter1d(
                self.time_spent_each_head_angle_hist,
                smooth_width_sigma,
                mode="wrap",
            )

    def hd_kde(self, kappa=200, n_bins=None):
        if n_bins is None:
            n_bins = int(round(360 / self.bin_size))
            (
                self.hd_bin_centers_kde,
                self.time_spent_each_head_angle_kde,
            ) = vonmises_kde(np.deg2rad(self.angles), kappa, n_bins)


class PlaceVals:
    def __init__(self, df, options, config):
        df = clean_df(df, speed_cutoff=options.place_speed_cut_off, copy=True)
        self.x = []
        self.y = []
        self.x_max = config.camera_x
        self.y_max = config.camera_y
        self.x_bins = []
        self.y_bins = []
        self.x_bin_centers = []
        self.y_bin_centers = []
        self.place_hist_seconds = []

        self.camera_frames_per_sec = config.camera_frames_per_sec
        self.bin_size = options.spatial_bin_size

        self.x, self.y = get_positions(
            df, use_head_as_position=options.spatial_position_head
        )
        self.convert_positions_to_m(config.meters_per_pixel)
        self.get_bins()
        self.place_hist()

    def convert_positions_to_m(self, real_length_to_pixel_conversion):
        self.x = self.x * real_length_to_pixel_conversion
        self.y = self.y * real_length_to_pixel_conversion

        self.x_max = self.x_max * real_length_to_pixel_conversion
        self.y_max = self.y_max * real_length_to_pixel_conversion

    def get_bins(self):
        self.x_bins = tools.get_bins(
            self.x, self.bin_size, min_val=0, max_val=self.x_max
        )
        self.y_bins = tools.get_bins(
            self.y, self.bin_size, min_val=0, max_val=self.y_max
        )
        self.x_bin_centers = midpoints_of_series(self.x_bins)
        self.y_bin_centers = midpoints_of_series(self.y_bins)

    def place_hist(self):
        logging.info("Calculating 2D place histogram")
        place_hist, _, _ = np.histogram2d(
            self.x, self.y, bins=[self.x_bins, self.y_bins]
        )
        self.place_hist_seconds = place_hist / self.camera_frames_per_sec


class VelocityVals:
    def __init__(self, df, options, camera_frames_per_sec):
        df = clean_df(
            df, speed_cutoff=options.velocity_speed_cut_off, copy=True
        )
        self.velocity = df["total_speed"]
        self.camera_frames_per_sec = camera_frames_per_sec
        self.bin_size = options.velocity_bin_size
        self.bins = []
        self.velocity_hist_seconds = []
        self.velocity_hist_centers = []

        self.min_velocity = 0
        self.max_velocity = options.max_velocity

        self.velocity_hist()

    def velocity_hist(self):
        logging.info("Calculating velocity histogram")
        self.bins = tools.get_bins(self.velocity, self.bin_size)

        frames_per_velocity_bin, b = np.histogram(self.velocity, self.bins)
        self.velocity_hist_seconds = (
            frames_per_velocity_bin / self.camera_frames_per_sec
        )
        self.velocity_hist_centers = midpoints_of_series(b)


class AngularHeadVelocityVals:
    # NB: min/max ahv will be inclusive, and based on raw, not binned data
    def __init__(
        self, df, options, camera_frames_per_sec, border_choice="max"
    ):
        df = clean_df(df, speed_cutoff=options.ahv_speed_cut_off, copy=True)
        self.ang_vel = df["angular_head_velocity"]
        self.camera_frames_per_sec = camera_frames_per_sec
        self.bin_size = options.ang_vel_bin_size
        self.central_ahv_fraction = options.central_ahv_fraction
        self.border_choice = border_choice
        self.max_ahv = options.max_ahv
        self.min_ahv = []
        self.bins = []
        self.ahv_hist_seconds = []
        self.ahv_hist_centers = []
        self.retained_ahv_timepoint_fraction = []

        self.dp = 2  # decimal points for reporting values

        self.ahv_hist()
        self.calculate_ahc_cutoffs()
        self.calculate_retained_timepoints()

    def ahv_hist(self):
        logging.info("Calculating angular head velocity histogram")
        self.bins = tools.get_bins(
            self.ang_vel, self.bin_size, deal_with_decimals=False
        )

        frames_per_ang_vel_bin, b = np.histogram(self.ang_vel, self.bins)
        self.ahv_hist_seconds = (
            frames_per_ang_vel_bin / self.camera_frames_per_sec
        )
        self.ahv_hist_centers = midpoints_of_series(b)

    def calculate_ahc_cutoffs(self):
        logging.info("Calculating angular head velocity cutoffs")

        if self.central_ahv_fraction is not None:
            min_keep, max_keep = tools.keep_n_central_bins(
                self.ahv_hist_seconds,
                self.ahv_hist_centers,
                fraction_keep=self.central_ahv_fraction,
            )

            borders = (abs(min_keep), abs(max_keep))
            if self.border_choice is "raw":
                self.min_ahv = min_keep
                self.max_ahv = max_keep
            elif self.border_choice is "max":
                self.min_ahv = -np.max(borders)
                self.max_ahv = np.max(borders)
            elif self.border_choice is "min":
                self.min_ahv = -np.min(borders)
                self.max_ahv = np.min(borders)

        elif self.max_ahv is not None:
            self.min_ahv = -self.max_ahv

        logging.debug(
            "AHV cutoffs - minimum: {} [deg/s], "
            "maximum: {} [deg/s]".format(self.min_ahv, self.max_ahv)
        )

    def calculate_retained_timepoints(self):
        ahv_kept = self.ang_vel[
            (self.ang_vel > self.min_ahv) & (self.ang_vel < self.max_ahv)
        ]
        self.retained_ahv_timepoint_fraction = len(ahv_kept) / len(
            self.ang_vel
        )
        logging.info(
            "AHV timepoints retained: {}%".format(
                round((self.retained_ahv_timepoint_fraction * 100), self.dp)
            )
        )


def get_positions(df, use_head_as_position=True):
    """
    Return positions of the animal. Float conversion is for dlc csv support.
    """
    if use_head_as_position:
        x = (
            df.Hear_L_x.astype(np.float).to_numpy()
            + df.Hear_R_x.astype(np.float).to_numpy()
        ) / 2
        y = (
            df.Hear_L_y.astype(np.float).to_numpy()
            + df.Hear_R_y.astype(np.float).to_numpy()
        ) / 2
    else:
        x = df.Back_x.astype(np.float).to_numpy()
        y = df.Back_y.astype(np.float).to_numpy()

    return x, y
