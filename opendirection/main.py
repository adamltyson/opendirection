import logging
import os.path
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from pathlib import Path
from datetime import datetime
from fancylog import fancylog
import opendirection.run.run as run


def main():
    start_time = datetime.now()
    config_path = os.path.join(str(Path(__file__).parents[1]), "options")

    args, options, config = run.setup(config_path)
    logging.info("Running full analysis of each condition")
    conditions, total_df = run.analysis(args, options, config)

    if options.stability:
        conditions, _, _ = run_stability_analyses(
            args, options, config, conditions
        )

    if args.save:
        save_dfs = {}
        logging.info("Saving results")
        for condition in conditions:
            save_dfs[condition.name] = run.save(condition, args.output_dir)
        if args.summary_paths is not None:
            logging.info("saving summary csvs")
            for summary_path in args.summary_paths:
                run.summary(summary_path, save_dfs, args.output_dir)

    logging.info(
        "Finished calculations. Total time taken: %s",
        datetime.now() - start_time,
    )

    fancylog.disable_logging()

    if args.plot:
        logging.info("PLOTTING:")
        for condition in conditions:
            run.plot(
                condition.all_data,
                condition.cell_specific_data,
                condition.cell_specific_stats,
                condition.hd,
                condition.ahv,
                condition.velocity,
                condition.place,
                condition.colors,
                config.camera_frames_per_sec,
                options,
                condition_name=condition.name,
            )
    plt.show()


def run_stability_analyses(args, options, config, conditions):
    logging.info("Running analysis on first half of each condition")
    conditions_first, _ = run.analysis(
        args, options, config, stability="first"
    )

    logging.info("Running analysis on second half of each condition")
    conditions_last, _ = run.analysis(args, options, config, stability="last")

    (
        conditions,
        condition_r_velocity_distributions,
        condition_r_ahv_distributions,
    ) = add_stability_indices(conditions, conditions_first, conditions_last)
    return (
        conditions,
        condition_r_velocity_distributions,
        condition_r_ahv_distributions,
    )


def stability_calcs(conditions_first, conditions_last):
    condition_velocity_correlations = []
    condition_ahv_correlations = []

    condition_r_velocity_distributions = []
    condition_r_ahv_distributions = []

    condition_velocity_null_percentiles = []
    condition_ahv_null_percentiles = []

    for idx, condition in enumerate(conditions_first):

        velocity_correlations = []
        ahv_correlations = []

        r_velocity_distributions = []
        r_ahv_distributions = []

        velocity_null_percentiles = []
        ahv_null_percentiles = []
        for cell_id, cell in enumerate(condition.cell_list):
            _r_velocity_distributions = correlate_null_distributions(
                conditions_first[idx]
                .cell_specific_stats[cell_id]
                .velocity._VelocityStats__shuffled_binned_data,
                conditions_last[idx]
                .cell_specific_stats[cell_id]
                .velocity._VelocityStats__shuffled_binned_data,
            )

            min_length = min(
                len(
                    conditions_first[
                        idx
                    ].cell_specific_data.velocity_cell_spikes_freq[cell_id]
                ),
                len(
                    conditions_last[
                        idx
                    ].cell_specific_data.velocity_cell_spikes_freq[cell_id]
                ),
            )
            r, p = stats.pearsonr(
                conditions_first[
                    idx
                ].cell_specific_data.velocity_cell_spikes_freq[cell_id][
                    :min_length
                ],
                conditions_last[
                    idx
                ].cell_specific_data.velocity_cell_spikes_freq[cell_id][
                    :min_length
                ],
            )

            velocity_correlations.append((r, p))
            velocity_null_percentile = stats.percentileofscore(
                abs(_r_velocity_distributions), abs(r)
            )
            velocity_null_percentiles.append(velocity_null_percentile)
            r_velocity_distributions.append(_r_velocity_distributions)

            _r_ahv_distributions = correlate_null_distributions(
                conditions_first[idx]
                .cell_specific_stats[cell_id]
                .ahv._AHVStats__shuffled_binned_data,
                conditions_last[idx]
                .cell_specific_stats[cell_id]
                .ahv._AHVStats__shuffled_binned_data,
            )

            r, p = stats.pearsonr(
                conditions_first[idx].cell_specific_data.ahv_cell_spikes_freq[
                    cell_id
                ],
                conditions_last[idx].cell_specific_data.ahv_cell_spikes_freq[
                    cell_id
                ],
            )

            ahv_correlations.append((r, p))
            ahv_null_percentile = stats.percentileofscore(
                abs(_r_ahv_distributions), abs(r)
            )
            ahv_null_percentiles.append(ahv_null_percentile)
            r_ahv_distributions.append(_r_ahv_distributions)

        condition_velocity_correlations.append(velocity_correlations)
        condition_ahv_correlations.append(ahv_correlations)

        condition_r_velocity_distributions.append(r_velocity_distributions)
        condition_r_ahv_distributions.append(r_ahv_distributions)

        condition_velocity_null_percentiles.append(velocity_null_percentiles)
        condition_ahv_null_percentiles.append(ahv_null_percentiles)
    return (
        condition_velocity_correlations,
        condition_ahv_correlations,
        condition_velocity_null_percentiles,
        condition_ahv_null_percentiles,
        condition_r_velocity_distributions,
        condition_r_ahv_distributions,
    )


def correlate_null_distributions(
    shuffled_tunings_first, shuffled_tunings_last
):
    r_values = []
    for first, last in zip(shuffled_tunings_first, shuffled_tunings_last):
        min_length = min(len(first), len(last))
        r_values.append(
            stats.pearsonr(first[:min_length], last[:min_length])[0]
        )
    return np.array(r_values)


def add_stability_indices(conditions, conditions_first, conditions_last):
    (
        condition_velocity_correlations,
        condition_ahv_correlations,
        condition_velocity_null_percentiles,
        condition_ahv_null_percentiles,
        condition_r_velocity_distributions,
        condition_r_ahv_distributions,
    ) = stability_calcs(conditions_first, conditions_last)

    for idx, condition in enumerate(conditions):
        # r is first output of stats.pearsonr
        for cell_id, cell in enumerate(condition.cell_specific_stats):
            # ahv
            cell.ahv.ahv_stability_index = condition_velocity_correlations[
                idx
            ][cell_id][0]

            cell.ahv.ahv_pearson_r_first_half_neg = (
                conditions_first[idx]
                .cell_specific_stats[cell_id]
                .ahv.ahv_pearson_r_neg
            )
            cell.ahv.ahv_pearson_r_first_half_pos = (
                conditions_first[idx]
                .cell_specific_stats[cell_id]
                .ahv.ahv_pearson_r_pos
            )

            cell.ahv.ahv_pearson_r_second_half_neg = (
                conditions_last[idx]
                .cell_specific_stats[cell_id]
                .ahv.ahv_pearson_r_neg
            )
            cell.ahv.ahv_pearson_r_second_half_pos = (
                conditions_last[idx]
                .cell_specific_stats[cell_id]
                .ahv.ahv_pearson_r_pos
            )

            cell.ahv.ahv_r_percentile_first_half_neg = (
                conditions_first[idx]
                .cell_specific_stats[cell_id]
                .ahv.pearson_neg_percentile
            )
            cell.ahv.ahv_r_percentile_first_half_pos = (
                conditions_first[idx]
                .cell_specific_stats[cell_id]
                .ahv.pearson_pos_percentile
            )

            cell.ahv.ahv_r_percentile_second_half_neg = (
                conditions_last[idx]
                .cell_specific_stats[cell_id]
                .ahv.pearson_neg_percentile
            )
            cell.ahv.ahv_r_percentile_second_half_pos = (
                conditions_last[idx]
                .cell_specific_stats[cell_id]
                .ahv.pearson_pos_percentile
            )

            cell.ahv.ahv_null_correlation_percentile = condition_ahv_null_percentiles[
                idx
            ][
                cell_id
            ]

            # velocity
            cell.velocity.velocity_stability_index = condition_ahv_correlations[
                idx
            ][
                cell_id
            ][
                0
            ]

            cell.velocity.velocity_r_percentile_first_half = (
                conditions_first[idx]
                .cell_specific_stats[cell_id]
                .velocity.pearson_percentile
            )
            cell.velocity.velocity_r_percentile_second_half = (
                conditions_last[idx]
                .cell_specific_stats[cell_id]
                .velocity.pearson_percentile
            )

            cell.velocity.velocity_pearson_r_first_half = (
                conditions_first[idx]
                .cell_specific_stats[cell_id]
                .velocity.velocity_pearson_r
            )

            cell.velocity.velocity_pearson_r_second_half = (
                conditions_last[idx]
                .cell_specific_stats[cell_id]
                .velocity.velocity_pearson_r
            )

            cell.velocity.velocity_null_correlation_percentile = condition_velocity_null_percentiles[
                idx
            ][
                cell_id
            ]

    return (
        conditions,
        condition_r_velocity_distributions,
        condition_r_ahv_distributions,
    )


if __name__ == "__main__":
    results = main()
