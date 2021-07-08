import configparser
import os.path

from imlib.general.list import unique_elements_lists, strip_spaces_list


class GetOptions:
    def __init__(self, options_path):
        if os.path.isdir(options_path):
            options_config = os.path.join(options_path, "options.ini")
        else:
            options_config = options_path

        self.parallel = False
        self.n_free_cpus = 6
        self.stability = False

        self.conditions_list = []
        self.cell_condition_inclusion = "individual"

        self.min_fire_freq = []
        self.min_fire_freq_per_direction_bin = []

        self.speed_cut_off = []

        self.ang_vel_filter_hd = False
        self.ang_vel_filter_hd_filt_width = None

        self.hd_speed_cut_off = None
        self.direction_bin = None
        self.direction_baseline_bin = None
        self.hd_smooth = False
        self.hd_smooth_sigma = None
        self.hd_kde_kappa = None

        self.ahv_speed_cut_off = None
        self.ang_vel_local_derivative_window = None
        self.ang_vel_bin_size = []
        self.calculate_central_ahv = False
        self.central_ahv_fraction = None
        self.max_ahv = None

        self.velocity_speed_cut_off = None
        self.velocity_position_head = False
        self.velocity_smooth = False
        self.velocity_smooth_sigma = None
        self.velocity_bin_size = []
        self.max_velocity = []

        self.place_speed_cut_off = None
        self.spatial_position_head = False
        self.spatial_bin_size = []
        self.min_time_in_spatial_bin = []
        self.place_firing_smooth = False
        self.place_firing_smooth_width = None

        self.plot_hist_or_kde = "both"
        self.plot_head_direction = False
        self.plot_cell_direction_overlay = False
        self.plot_cell_direction_subplot = False
        self.plot_all_behaviour = False
        self.plot_raw_spikes = False
        self.plot_velocity = False
        self.plot_velocity_log = False
        self.plot_velocity_firing_rate = False
        self.plot_velocity_remove_zeros = False
        self.plot_angular_velocity = False
        self.plot_ahv_firing_rate = False
        self.plot_trajectory = False
        self.plot_spatial_occupancy = False
        self.plot_space_firing = False

        self.plot_ang_vel_log = False
        self.plot_ahv_remove_zeros = False
        self.plot_ahv_fit = (False,)
        self.plot_median_filter = False
        self.filter_width_median_plot = None

        self.hd_shuffle_test = False
        self.hd_shuffle_min_magnitude = []
        self.hd_shuffle_max_magnitude = []
        self.hd_shuffle_iterations = []
        self.calc_hd_peak_method = "mean"
        self.calc_hd_hist_smooth = False
        self.calc_hd_hist_smooth_width = None
        self.ahv_shuffle_test = False
        self.ahv_shuffle_min_magnitude = []
        self.ahv_shuffle_max_magnitude = []
        self.ahv_shuffle_iterations = []
        self.ahv_correlation_magnitude = True
        self.velocity_shuffle_test = False
        self.velocity_shuffle_min_magnitude = []
        self.velocity_shuffle_max_magnitude = []
        self.velocity_shuffle_iterations = []
        self.velocity_correlation_magnitude = True
        self.place_shuffle_test = False
        self.place_shuffle_min_magnitude = []
        self.place_shuffle_max_magnitude = []
        self.place_shuffle_iterations = []

        self.color_palette = "husl"
        self.plot_transparency = []

        # multiexp
        self.multiexp_shuffle_min_magnitude = []
        self.multiexp_shuffle_max_magnitude = []
        self.multiexp_shuffle_iterations = []

        config = configparser.ConfigParser()
        config.read(options_config)

        self.general_parse(config)
        self.condition_option_parse(config)
        self.cell_selection_parse(config)
        self.behaviour_selection_parse(config)
        self.head_direction_parse(config)
        self.angular_head_velocity_parse(config)
        self.velocity_parse(config)
        self.place_parse(config)
        self.plot_option_parse(config)
        self.stats_parse(config)
        self.display_parse(config)
        self.multiexp_parse(config)

    def general_parse(self, config):
        self.n_free_cpus = int(config["GENERAL"]["N_FREE_CPUS"])
        self.parallel = config["GENERAL"].getboolean("PARALLEL")
        self.stability = config["GENERAL"].getboolean("STABILITY")

    def condition_option_parse(self, config):
        self.conditions_list = config.get(
            "CONDITIONS", "CONDITIONS_LIST"
        ).split(",")
        self.cell_condition_inclusion = config["CONDITIONS"][
            "CELL_CONDITION_INCLUSION"
        ]

    def cell_selection_parse(self, config):
        self.min_fire_freq = config["CELL_SELECTION"].getfloat("MIN_FIRE_FREQ")
        self.min_fire_freq_per_direction_bin = config[
            "CELL_SELECTION"
        ].getfloat("MIN_HZ_IN_ANY_DIRECTION_BIN")

    def behaviour_selection_parse(self, config):
        self.speed_cut_off = config["BEHAVIOUR_SELECTION"].getfloat(
            "SPEED_CUT_OFF"
        )

    def head_direction_parse(self, config):
        self.hd_speed_cut_off = config["HEAD_DIRECTION"].getfloat(
            "HD_SPEED_CUT_OFF"
        )
        self.direction_bin = config["HEAD_DIRECTION"].getfloat("DIRECTION_BIN")
        self.direction_baseline_bin = config["HEAD_DIRECTION"].getfloat(
            "BASELINE_BIN"
        )
        self.ang_vel_filter_hd = config["HEAD_DIRECTION"].getboolean(
            "FILTER_HEAD_DIRECTION"
        )
        self.hd_kde_kappa = config["HEAD_DIRECTION"].getfloat("HD_KDE_KAPPA")

        if self.ang_vel_filter_hd:
            self.ang_vel_filter_hd_filt_width = config[
                "HEAD_DIRECTION"
            ].getfloat("HEAD_DIRECTION_MED_FILT_WIDTH")

        self.hd_smooth = config["HEAD_DIRECTION"].getboolean("HD_HIST_SMOOTH")
        if self.hd_smooth:
            self.hd_smooth_sigma = int(
                config["HEAD_DIRECTION"]["HD_HIST_SMOOTH_WIDTH"]
            )

    def angular_head_velocity_parse(self, config):
        self.ahv_speed_cut_off = config["ANGULAR_HEAD_VELOCITY"].getfloat(
            "AHV_SPEED_CUT_OFF"
        )
        self.ang_vel_local_derivative_window = config[
            "ANGULAR_HEAD_VELOCITY"
        ].getfloat("LOCAL_DERIVATIVE_WINDOW")
        self.ang_vel_bin_size = config["ANGULAR_HEAD_VELOCITY"].getfloat(
            "ANG_VEL_BIN_SIZE"
        )
        self.calculate_central_ahv = config[
            "ANGULAR_HEAD_VELOCITY"
        ].getboolean("CALCULATE_CENTRAL_AHV")
        if self.calculate_central_ahv:
            self.central_ahv_fraction = config[
                "ANGULAR_HEAD_VELOCITY"
            ].getfloat("CENTRAL_AHV_FRACTION")
        else:
            self.max_ahv = config["ANGULAR_HEAD_VELOCITY"].getfloat("MAX_AHV")

    def velocity_parse(self, config):
        self.velocity_speed_cut_off = config["VELOCITY"].getfloat(
            "AHV_SPEED_CUT_OFF"
        )
        self.velocity_position_head = config["VELOCITY"].getboolean(
            "VELOCITY_POSITION_HEAD"
        )
        self.velocity_smooth = config["VELOCITY"].getboolean("FILTER_VELOCITY")
        if self.velocity_smooth:
            self.velocity_smooth_sigma = float(
                config["VELOCITY"]["VELOCITY_MED_FILT_WIDTH"]
            )

        self.velocity_bin_size = config["VELOCITY"].getfloat(
            "VELOCITY_BIN_SIZE"
        )
        self.max_velocity = config["VELOCITY"].getfloat("MAX_VELOCITY")

    def place_parse(self, config):
        self.place_speed_cut_off = config["PLACE"].getfloat(
            "AHV_SPEED_CUT_OFF"
        )
        self.spatial_position_head = config["PLACE"].getboolean(
            "SPATIAL_POSITION_HEAD"
        )
        self.spatial_bin_size = config["PLACE"].getfloat("SPATIAL_BIN_SIZE")
        self.min_time_in_spatial_bin = config["PLACE"].getfloat(
            "MIN_TIME_IN_SPATIAL_BIN"
        )
        self.place_firing_smooth = config["PLACE"].getboolean(
            "PLACE_FIRING_SMOOTH"
        )
        if self.place_firing_smooth:
            self.place_firing_smooth_width = config["PLACE"].getfloat(
                "PLACE_FIRING_SMOOTH_WIDTH"
            )

    def plot_option_parse(self, config):
        self.plot_hist_or_kde = config["PLOTTING"]["HIST_OR_KDE"]
        self.plot_head_direction = config["PLOTTING"].getboolean(
            "HEAD_DIRECTION"
        )
        self.plot_cell_direction_overlay = config["PLOTTING"].getboolean(
            "CELL_DIRECTION_OVERLAY"
        )
        self.plot_cell_direction_subplot = config["PLOTTING"].getboolean(
            "CELL_DIRECTION_SUBPLOT"
        )
        self.plot_all_behaviour = config["PLOTTING"].getboolean(
            "ALL_BEHAVIOUR"
        )
        self.plot_raw_spikes = config["PLOTTING"].getboolean("PLOT_RAW_SPIKES")
        self.plot_velocity = config["PLOTTING"].getboolean("PLOT_VELOCITY")
        self.plot_velocity_firing_rate = config["PLOTTING"].getboolean(
            "PLOT_VELOCITY_FIRING_RATE"
        )
        self.plot_angular_velocity = config["PLOTTING"].getboolean(
            "PLOT_ANGULAR_VELOCITY"
        )
        self.plot_ahv_firing_rate = config["PLOTTING"].getboolean(
            "PLOT_AHV_FIRING_RATE"
        )
        self.plot_trajectory = config["PLOTTING"].getboolean("PLOT_TRAJECTORY")
        self.plot_spatial_occupancy = config["PLOTTING"].getboolean(
            "PLOT_SPATIAL_OCCUPANCY"
        )
        self.plot_space_firing = config["PLOTTING"].getboolean(
            "PLOT_SPACE_FIRING"
        )

        if self.plot_all_behaviour:
            self.plot_median_filter = config["PLOTTING"].getboolean(
                "MEDIAN_FILTER_BEHAVIOUR_PLOT"
            )
            if self.plot_median_filter:
                # otherwise defaults to none, and no filtering is done
                self.filter_width_median_plot = config["PLOTTING"].getfloat(
                    "FILTER_WIDTH_BEHAVIOUR_PLOT"
                )

        if self.plot_angular_velocity:
            self.plot_ang_vel_log = config["PLOTTING"].getboolean(
                "PLOT_ANG_VEL_LOG"
            )

        if self.plot_ahv_firing_rate:
            self.plot_ahv_remove_zeros = config["PLOTTING"].getboolean(
                "PLOT_AHV_REMOVE_ZEROS"
            )
            self.plot_ahv_fit = config["PLOTTING"].getboolean("PLOT_AHV_FIT")

        if self.plot_velocity:
            self.plot_velocity_log = config["PLOTTING"].getboolean(
                "PLOT_VELOCITY_LOG"
            )

        if self.plot_velocity_firing_rate:
            self.plot_velocity_remove_zeros = config["PLOTTING"].getboolean(
                "PLOT_VELOCITY_REMOVE_ZEROS"
            )

    def display_parse(self, config):
        self.color_palette = config["DISPLAY"]["PALETTE"]
        self.plot_transparency = config["DISPLAY"].getfloat(
            "PLOT_TRANSPARENCY"
        )

    def stats_parse(self, config):
        self.hd_shuffle_test = config["STATISTICS"].getboolean(
            "HD_SHUFFLE_TEST"
        )

        self.calc_hd_peak_method = config["STATISTICS"]["CALC_HD_PEAK_METHOD"]

        if self.calc_hd_peak_method == "hist":
            self.calc_hd_hist_smooth = config["STATISTICS"].getboolean(
                "CALC_HD_HIST_SMOOTH"
            )
            if self.calc_hd_hist_smooth:
                self.calc_hd_hist_smooth_width = int(
                    config["STATISTICS"]["CALC_HD_HIST_SMOOTH_WIDTH"]
                )

        if self.hd_shuffle_test:
            self.hd_shuffle_min_magnitude = int(
                config["STATISTICS"]["HD_SHUFFLE_MIN_MAGNITUDE"]
            )
            self.hd_shuffle_max_magnitude = int(
                config["STATISTICS"]["HD_SHUFFLE_MAX_MAGNITUDE"]
            )
            self.hd_shuffle_iterations = int(
                config["STATISTICS"]["HD_SHUFFLE_ITERATIONS"]
            )

        self.ahv_shuffle_test = config["STATISTICS"].getboolean(
            "AHV_SHUFFLE_TEST"
        )

        if self.ahv_shuffle_test:
            self.ahv_shuffle_min_magnitude = int(
                config["STATISTICS"]["AHV_SHUFFLE_MIN_MAGNITUDE"]
            )
            self.ahv_shuffle_max_magnitude = int(
                config["STATISTICS"]["AHV_SHUFFLE_MAX_MAGNITUDE"]
            )
            self.ahv_shuffle_iterations = int(
                config["STATISTICS"]["AHV_SHUFFLE_ITERATIONS"]
            )
            self.ahv_correlation_magnitude = config["STATISTICS"].getboolean(
                "AHV_CORRELATION_MAGNITUDE"
            )

        self.velocity_shuffle_test = config["STATISTICS"].getboolean(
            "VELOCITY_SHUFFLE_TEST"
        )

        if self.velocity_shuffle_test:
            self.velocity_shuffle_min_magnitude = int(
                config["STATISTICS"]["VELOCITY_SHUFFLE_MIN_MAGNITUDE"]
            )
            self.velocity_shuffle_max_magnitude = int(
                config["STATISTICS"]["VELOCITY_SHUFFLE_MAX_MAGNITUDE"]
            )
            self.velocity_shuffle_iterations = int(
                config["STATISTICS"]["VELOCITY_SHUFFLE_ITERATIONS"]
            )
            self.velocity_correlation_magnitude = config[
                "STATISTICS"
            ].getboolean("VELOCITY_CORRELATION_MAGNITUDE")

        self.place_shuffle_test = config["STATISTICS"].getboolean(
            "PLACE_SHUFFLE_TEST"
        )

        if self.place_shuffle_test:
            self.place_shuffle_min_magnitude = int(
                config["STATISTICS"]["PLACE_SHUFFLE_MIN_MAGNITUDE"]
            )
            self.place_shuffle_max_magnitude = int(
                config["STATISTICS"]["PLACE_SHUFFLE_MAX_MAGNITUDE"]
            )
            self.place_shuffle_iterations = int(
                config["STATISTICS"]["PLACE_SHUFFLE_ITERATIONS"]
            )

    def multiexp_parse(self, config):
        self.multiexp_shuffle_min_magnitude = int(
            config["MULTIEXPERIMENT"]["MULTIEXP_SHUFFLE_MIN_MAGNITUDE"]
        )
        self.multiexp_shuffle_max_magnitude = int(
            config["MULTIEXPERIMENT"]["MULTIEXP_SHUFFLE_MAX_MAGNITUDE"]
        )
        self.multiexp_shuffle_iterations = int(
            config["MULTIEXPERIMENT"]["MULTIEXP_SHUFFLE_ITERATIONS"]
        )


class GetConfig:
    def __init__(self, options_path):
        condition_timing_config = os.path.join(options_path, "conditions.ini")
        variables_config = os.path.join(options_path, "config.ini")
        self.conditions = []
        self.condition_timing_starts = []
        self.condition_timing_ends = []
        self.condition_timing_excludes = []

        self.probe_samples_per_sec = []
        self.camera_frames_per_sec = []
        self.meters_per_pixel = []
        self.camera_x = []
        self.camera_y = []

        self.conditions_parse(condition_timing_config)
        self.config_parse(variables_config)

    def conditions_parse(self, condition_timing_config):
        config = configparser.ConfigParser()
        config.read(condition_timing_config)
        self.conditions = config.sections()
        for condition in self.conditions:
            start = int(config[condition]["START"])
            end = int(config[condition]["END"])
            exclude = config[condition]["EXCLUDE"].split(",")
            self.condition_timing_starts.append(start)
            self.condition_timing_ends.append(end)
            self.condition_timing_excludes.append(exclude)

    def config_parse(self, variables_config):
        config = configparser.ConfigParser()
        config.read(variables_config)

        self.probe_samples_per_sec = int(
            config["SETUP"]["PROBE_SAMPLES_PER_SEC"]
        )
        self.camera_frames_per_sec = int(
            config["SETUP"]["CAMERA_FRAMES_PER_SEC"]
        )
        self.meters_per_pixel = float(config["SETUP"]["METERS_PER_PIXEL"])
        self.camera_x = float(config["SETUP"]["CAMERA_X"])
        self.camera_y = float(config["SETUP"]["CAMERA_Y"])


class GetSummaryConfig:
    def __init__(
        self,
        summary_file,
        direction_default="higher",
        parameter_section_name="SAVE_PARAMETERS",
        strip_spaces=True,
    ):
        self._direction_default = direction_default
        self._strip_spaces = strip_spaces
        self.parameter_section_name = parameter_section_name
        self.sections = []
        self.conditions = []
        self.parameters = []
        self.values = []
        self.directions = []

        self.parameters_keep = []

        self.conditions_parse(summary_file)

    def conditions_parse(self, summary_config):
        config = configparser.ConfigParser()
        config.read(summary_config)
        self.sections = config.sections()
        for section in self.sections:
            if section == self.parameter_section_name:
                self.get_parameters_to_save(config)
            else:
                condition = config[section]["CONDITION"]
                parameter = config[section]["PARAMETER"]
                value = float(config[section]["VALUE"])

                if config.has_option(section, "DIRECTION"):
                    direction = config[section]["DIRECTION"]
                else:
                    direction = self._direction_default

                self.conditions.append(condition)
                self.parameters.append(parameter)
                self.values.append(value)
                self.directions.append(direction)

        if self._strip_spaces:
            self.conditions = strip_spaces_list(self.conditions)

    def get_parameters_to_save(self, config):
        for subsection in config[self.parameter_section_name]:
            parameters_keep_tmp = config.get(
                self.parameter_section_name, subsection
            ).split(",")
            self.parameters_keep = self.parameters_keep + parameters_keep_tmp

        self.parameters_keep = unique_elements_lists(self.parameters_keep)

        if self._strip_spaces:
            self.parameters_keep = strip_spaces_list(self.parameters_keep)
