import numpy as np

import matplotlib.pyplot as plt

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from imlib.array.size import pad_with_number_1d
from imlib.general.exceptions import ArgumentError
from imlib.math.trig import sine_min_max, get_scaled_sine


def sine_from_peak(
    peak,
    num_samples=6000,
    sample_duration=0.001,
    target_integral=118,
    plot=False,
    title="Timecourse",
    xlabel="Time",
    ylabel="Measure",
    save_to=None,
    delimiter=",",
    frequency=1,
    xunit="s",
    yunit="m",
    annotation_x=0.6,
    annotation_y=0.9,
):
    """
    Calculates a sine wave with a specified max, with the integral fixed.

    If the maximum value is not sufficient to reach the specified fixed
    integral, then another sine wave will be appended.

    If the maximum value is too high, and would overshoot the specified
    integral, then the duration of the sine wave is reduced, and the
    returned array will be zero in all other areas.

    :param peak: Maximum value to be reached. If the a second sine wave is
    appended, then this only applies to the first sine wave. Specified in
    (units)
    :param num_samples: Length of the final array
    :param sample_duration: How long (in seconds) is each sample.
    :param target_integral: Specified integral of the returned function
    :param plot: If true, plot the function. Default: False
    :param title: Default: "Timecourse"
    :param xlabel: Default: "s"
    :param ylabel: Default: "m"
    :param save_to: Save to this file. Default: None
    :param delimiter: Save file delimiter. Default: ","
    :param frequency: Frequency of the sine function. Default: 1
    :param xunit: Unit for the x axis. Default: s
    :param yunit: Unit for they axis. Default: cm
    :param annotation_x: Where on the axis to place information
    :param annotation_y: Where on the axis to place information
    """
    min_peak = ((2 / sample_duration) * target_integral) / num_samples

    # if the peak value if sufficient to reach the target integral
    if peak >= min_peak:
        frequency = 1 / peak
        num_samples_tmp = int(
            round(((2 / sample_duration) * target_integral) / peak)
        )

        min_val, max_val = sine_min_max(frequency)
        angles = np.linspace(min_val, max_val, num_samples_tmp)

        y = get_scaled_sine(angles, peak, frequency)

        y = pad_with_number_1d(y, num_samples)

    # otherwise an additional sine wave is needed
    else:
        num_samples_tmp = int(round(0.5 * num_samples * (peak / min_peak)))
        min_val, max_val = sine_min_max(frequency)

        angles = np.linspace(min_val, max_val, num_samples_tmp)
        y = get_scaled_sine(angles, peak, frequency)
        x = np.arange(0, sample_duration * num_samples_tmp, sample_duration)
        area = np.trapz(y, x)
        area_diff = target_integral - area
        space_left = num_samples - num_samples_tmp
        angles = np.linspace(min_val, max_val, space_left)
        velocity_extra = np.sin(frequency * angles) + 1
        x = np.arange(0, sample_duration * space_left, sample_duration)
        area = np.trapz(velocity_extra, x)
        velocity_extra = velocity_extra * (area_diff / area)
        y = np.append(y, velocity_extra)

    x = np.arange(0, num_samples * sample_duration, sample_duration)
    final_area = np.trapz(y, x)
    peak = np.max(y)

    if save_to is not None:
        np.savetxt(save_to, y, delimiter=delimiter)
    else:
        print("No output file specified. Not saving results.")
    if plot:
        print("Plotting profile")
        plt.figure()
        xaxis = np.linspace(0, num_samples * sample_duration, num_samples)
        plt.plot(xaxis, y)
        plt.title(title)
        plt.xlabel(xlabel + "({})".format(xunit))
        plt.ylabel(ylabel + "({})".format(yunit))
        annotation = "Max vel: {}({}) \n Displacement: {} ({})".format(
            round(peak), yunit, round(final_area), xunit
        )

        plt.text(
            annotation_x,
            annotation_y,
            annotation,
            transform=plt.gca().transAxes,
        )
        plt.show()
    else:
        print("--plot not specified. Not plotting")
    print("Done!")


def parse():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        dest="max_value",
        type=float,
        help="Maximum value to be reached. If the a second sine wave is "
        "appended, then this only applies to the first sine wave.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        type=str,
        help="Output file path (.csv.)",
        default=None,
    )
    parser.add_argument(
        "--plot", action="store_true", help="If used, plot the results"
    )
    parser.add_argument(
        "--target-integral",
        dest="target_integral",
        type=float,
        help="Specified area under the resulting function.",
        default=118,
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        type=float,
        help="Number of timepoints to calculate over",
        default=6000,
    )
    parser.add_argument(
        "--sample-duration",
        dest="sample_duration",
        type=float,
        help="How long (in seconds) is each sample.",
        default=0.001,
    )
    return parser


def main():
    args = parse().parse_args()
    if args.output_file is None and not args.plot:
        raise ArgumentError(
            "Neither plotting nor saving, " "are specified, aborting."
        )
    else:

        sine_from_peak(
            args.max_value,
            plot=args.plot,
            save_to=args.output_file,
            title="Velocity profile",
            ylabel="Velocity",
            yunit="cm/s",
            num_samples=args.num_samples,
            sample_duration=args.sample_duration,
            target_integral=args.target_integral,
        )


if __name__ == "__main__":
    main()
