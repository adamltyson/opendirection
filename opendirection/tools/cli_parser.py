from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def file_parse(parser):
    file_parser = parser.add_argument_group("Files")
    file_parser.add_argument(
        "--spiketimes",
        dest="spike_time_file",
        type=str,
        required=True,
        help="Spike timing file (.npy)",
    )
    file_parser.add_argument(
        "--spikeclusters",
        dest="spike_clusters_file",
        type=str,
        required=True,
        help="Spike clusters file (.npy)",
    )
    file_parser.add_argument(
        "--clustergroups",
        dest="cluster_groups_file",
        type=str,
        required=True,
        help="Cluster groups file (.csv)",
    )
    file_parser.add_argument(
        "--syncparams",
        dest="sync_param_file",
        type=str,
        required=True,
        help="Synchronisation parameters (.mat)",
    )
    file_parser.add_argument(
        "--dlcfiles",
        dest="dlc_files",
        type=str,
        required=True,
        default=[],
        nargs="*",
        help="DeepLabCut .xlsx files",
    )
    return parser


def main_parse(parser):
    main_parser = parser.add_argument_group("General options")
    main_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    main_parser.add_argument(
        "--config",
        type=str,
        dest="config_path",
        help="External configuration directory",
    )
    main_parser.add_argument(
        "--summary-config",
        type=str,
        dest="summary_paths",
        nargs="+",
        help="Paths to N summary configuration files (in the style of"
        "opendirection/options/summary_eg.ini",
    )
    main_parser.add_argument(
        "--experiment-name",
        type=str,
        dest="experiment_name",
        required=True,
        help="Experiment name. Used for correlating with other experiments",
    )
    return parser


def misc_parse(parser):
    misc_parser = parser.add_argument_group("Misc options")
    misc_parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Increase verbosity. For debugging.",
    )
    misc_parser.add_argument(
        "-P", "--plot", action="store_true", help="Whether to plot figures"
    )
    misc_parser.add_argument(
        "-S",
        "--save",
        action="store_true",
        help="Whether to save results as .csv",
    )
    return parser


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = file_parse(parser)
    parser = main_parse(parser)
    parser = misc_parse(parser)
    args = parser.parse_args()
    return args
