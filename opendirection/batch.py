import logging
import pickle

import os.path
from pathlib import Path
from datetime import datetime

import opendirection.run.run as run
from opendirection.main import run_stability_analyses


def run_analysis_single_bin(args, options, config):
    conditions, total_df = run.analysis(args, options, config)
    if options.stability:
        conditions = run_stability_analyses(args, options, config, conditions)

    results_df = {}
    for condition in conditions:
        results_df[condition.name] = run.save(condition, args.output_dir)

    return results_df, conditions


def export(options, args, ang_vel_bin_sizes, conditions, results_dfs):
    output = {}
    output["AHV_max_magnitude"] = options.max_ahv
    output["AHV_bin_size"] = ang_vel_bin_sizes
    output["experiment_name"] = args.experiment_name
    output["cell_data"] = []
    tmp_output = {}
    for i, ang_vel_bin_size in enumerate(ang_vel_bin_sizes):
        for condition in conditions[i]:
            condition_dict = {}
            condition_dict["cell_information"] = results_dfs[i][condition.name]
            condition_dict[
                "AHV_spiking_frequency"
            ] = condition.cell_specific_data.ahv_cell_spikes_freq
            condition_dict[
                "cell_list_order"
            ] = condition.cell_specific_data.cell_list
            tmp_output[condition.name] = condition_dict
        output["cell_data"].append(tmp_output)
        tmp_output = {}
    filename = os.path.join(args.output_dir, args.experiment_name + ".opend")
    pickle.dump(output, open(filename, "wb"))


def main():
    start_time = datetime.now()
    config_path = os.path.join(str(Path(__file__).parents[1]), "options")

    args, options, config = run.setup(config_path)

    ang_vel_bin_sizes = [5.0, 10.0]
    results_dfs = []
    conditions = []
    for ang_vel_bin_size in ang_vel_bin_sizes:
        options.ang_vel_bin_size = ang_vel_bin_size
        results_df_bin, conditions_bin = run_analysis_single_bin(
            args, options, config
        )
        results_dfs.append(results_df_bin)
        conditions.append(conditions_bin)

    logging.info("Exporting")
    export(options, args, ang_vel_bin_sizes, conditions, results_dfs)

    logging.info(
        "Finished calculations. Total time taken: %s",
        datetime.now() - start_time,
    )


if __name__ == "__main__":
    results = main()
