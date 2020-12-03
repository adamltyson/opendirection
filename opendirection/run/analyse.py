from imlib.plotting.colors import get_n_colors

import opendirection.combine.calculate_general as combine_calculate
import opendirection.combine.cell_specific as cell_calc
import opendirection.stats.stats as stats
import opendirection.combine.condition as condition_select
import opendirection.combine.cell_select as cell_select


class ConditionAnalyse:
    def __init__(self, total_df, condition, config, options):
        self.name = condition
        self.colors = []

        self.hd = []
        self.ahv = []
        self.velocity = []
        self.place = []
        self.cell_specific_data = []
        self.cell_specific_stats = []
        self.all_data = []
        self.cell_list = []

        self.cell_agnostic(total_df, config, options)

    def cell_agnostic(self, df, config, options):
        # Runs before cell list is calculated.
        self.select_condition(df, config.conditions)
        self.get_condition_specific_cell_agnostic(config, options)
        self.get_cell_list(options)

    def cell_specific(self, options, config):
        # Is called after a new, combined cell list is calculated
        self.cell_list.sort()
        self.get_colors(options.color_palette)
        self.get_cell_specific_data(options)
        self.get_cell_specific_stats(options, config)

    def select_condition(self, df, conditions):
        self.all_data = condition_select.condition_select(
            df, conditions, self.name
        )

    def get_condition_specific_cell_agnostic(self, config, options):
        self.hd = combine_calculate.HeadDirection(
            self.all_data, options, config.camera_frames_per_sec
        )

        self.ahv = combine_calculate.AngularHeadVelocityVals(
            self.all_data, options, config.camera_frames_per_sec
        )

        self.velocity = combine_calculate.VelocityVals(
            self.all_data, options, config.camera_frames_per_sec
        )

        self.place = combine_calculate.PlaceVals(
            self.all_data, options, config
        )

    def get_cell_specific_data(self, options):
        self.cell_specific_data = cell_calc.CellSpecific(
            self.all_data,
            self.hd,
            self.ahv,
            self.velocity,
            self.place,
            self.cell_list,
            options,
        )

    def get_cell_specific_stats(self, options, config):
        self.cell_specific_stats = [
            stats.Stats(
                self.cell_specific_data,
                self.all_data,
                self.hd,
                self.place,
                cell_name,
                options,
                config,
            )
            for cell_name in self.cell_list
        ]

    def get_cell_list(self, options):
        self.cell_list = cell_select.get_matching_cells(
            self.all_data,
            options,
            time_spent_each_head_angle=self.hd.time_spent_each_head_angle_hist,
            bin_size=options.direction_bin,
        )

    def get_colors(self, palette):
        self.colors = get_n_colors(object_list=self.cell_list, palette=palette)
