import logging
import os.path
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from fancylog import fancylog
import opendirection.run.run as run


def main():
    start_time = datetime.now()
    config_path = os.path.join(str(Path(__file__).parents[1]), "options")

    args, options, config = run.setup(config_path)
    conditions, total_df = run.analysis(args, options, config)

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


if __name__ == "__main__":
    results = main()
