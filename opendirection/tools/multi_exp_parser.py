from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main_parse(parser):
    main_parser = parser.add_argument_group("General options")
    main_parser.add_argument(
        type=str, dest="exp_files", help="Experiment file directories",
    )
    main_parser.add_argument(
        type=str, dest="options", help="Options file",
    )
    main_parser.add_argument(
        dest="output_dir", type=str, help="Output directory",
    )

    main_parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        required=True,
        help="Names of conditions to include. Must be exact matches",
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
        "--n-free-cpus",
        dest="n_free_cpus",
        type=int,
        default=2,
        help="How many CPU cores to leave free",
    )
    return parser


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = main_parse(parser)
    parser = misc_parse(parser)
    args = parser.parse_args()
    return args
