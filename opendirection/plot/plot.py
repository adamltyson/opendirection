import logging
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors

from scipy.ndimage.filters import gaussian_filter

from imlib.plotting.colors import get_random_colors
from imlib.plotting.matplotlib import make_legend
from spikey.histogram.radial import radial_spike_histogram_multiple

import opendirection.plot.setup as plot_setup

# TODO: organise hd cell plotting into one function
# TODO: Combine similar plotting functions


def plot(
    df,
    cell_specific_data,
    cell_specific_stats,
    hd,
    ahv,
    velocity,
    place,
    cell_colors,
    camera_frames_per_sec,
    options,
    condition_name=None,
    power_normalisation=0.17,
):
    logging.info("Plotting behaviour")

    global global_plot_color
    global_plot_color = "c"

    global global_plot_transparency
    global_plot_transparency = options.plot_transparency

    global global_plot_more_transparent
    global_plot_more_transparent = global_plot_transparency * 0.7

    if condition_name is not None:
        global global_condition_name
        global_condition_name = condition_name

    radial_bin_size = options.direction_bin

    overlay_direction = options.plot_cell_direction_overlay
    subplot_direction = options.plot_cell_direction_subplot

    legend = make_legend(cell_specific_data.cell_list, cell_colors)

    if options.plot_all_behaviour:
        if options.filter_width_median_plot is not None:
            filter_window = int(
                np.round(
                    options.filter_width_median_plot * camera_frames_per_sec
                )
            )
        else:
            filter_window = None

        behaviour_plot_all(df, filter_window=filter_window)

    if options.plot_head_direction:
        if options.plot_hist_or_kde in ["hist", "both"]:
            head_direction_plot_hist(
                hd.time_spent_each_head_angle_hist,
                hd.hd_bin_centers_hist,
                bin_size=radial_bin_size,
            )

        if options.plot_hist_or_kde in ["kde", "both"]:
            head_direction_plot_kde(
                hd.time_spent_each_head_angle_kde, hd.hd_bin_centers_kde
            )

    if overlay_direction or subplot_direction:
        plot_all_direction(
            cell_specific_data,
            cell_colors,
            normalise_overlay=True,
            weighted=True,
            legend=legend,
            overlay=overlay_direction,
            subplot=subplot_direction,
            bin_size=options.direction_bin,
            time_spent_each_bin=hd.time_spent_each_head_angle_hist,
            plot_type=options.plot_hist_or_kde,
        )

    if options.plot_raw_spikes:
        plot_spikes(
            df,
            cells=cell_specific_data.cell_list,
            colors=cell_colors,
            legend=legend,
        )

    if options.plot_angular_velocity:
        plot_angular_velocity(ahv, log=options.plot_ang_vel_log)

    if options.plot_ahv_firing_rate:
        plot_ahv_firing_rate(
            cell_specific_data,
            cell_specific_stats,
            cell_colors,
            remove_zero_bins=options.plot_ahv_remove_zeros,
            plot_fit=options.plot_ahv_fit,
        )

    if options.plot_trajectory:
        if options.spatial_position_head:
            trajectory_position = "head"
        else:
            trajectory_position = "body"
        plot_trajectory(df, position=trajectory_position)

    if options.plot_velocity:
        plot_velocity(velocity, log=options.plot_velocity_log)

    if options.plot_velocity_firing_rate:
        plot_velocity_firing_rate(
            cell_specific_data,
            cell_colors,
            remove_zero_bins=options.plot_velocity_remove_zeros,
        )

    if options.plot_spatial_occupancy:
        plot_spatial_occupancy(
            place.place_hist_seconds,
            power_normalisation=power_normalisation,
            smoothing=options.place_firing_smooth_width,
        )

    if options.plot_space_firing:
        plot_spatial_firing(
            cell_specific_data.place_cell_spikes_freq,
            cell_specific_data.cell_list,
            power_normalisation=power_normalisation,
            smoothing=options.place_firing_smooth_width,
        )


def plot_spatial_firing(
    place_arrays,
    cell_list,
    title="Firing rate in each spatial bin",
    power_normalisation=1,
    transpose=True,
    colormap=plt.cm.inferno,
    colormap_label="Firing rate (hz)",
    smoothing=None,
):

    logging.info("Plotting velocity firing rates")
    number_of_cells = len(cell_list)

    if smoothing is not None:
        logging.info("Smoothing place cell firing plot")
        for idx, v in enumerate(range(number_of_cells)):
            place_arrays[idx] = gaussian_filter(place_arrays[idx], smoothing)

    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name
    shape_array = place_arrays[0].shape
    aspect_ratio = shape_array[1] / shape_array[0]
    fig = plt.figure(figsize=(number_of_cells * 3, 4))

    all_vals = []
    for idx, v in enumerate(range(number_of_cells)):
        all_vals = all_vals + place_arrays[idx].reshape(-1).tolist()

    min_val = min(all_vals)
    max_val = max(all_vals)

    for idx, v in enumerate(range(number_of_cells)):
        if transpose:
            firing = place_arrays[idx].T
        else:
            firing = place_arrays[idx]

        v = v + 1
        ax1 = fig.add_subplot(1, number_of_cells + 1, v)
        ax1.set_title(cell_list[idx], y=1)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        im = plt.imshow(
            firing,
            cmap=colormap,
            interpolation="none",
            norm=colors.PowerNorm(
                power_normalisation, vmin=min_val, vmax=max_val
            ),
        )

        ax1.set_aspect(aspect_ratio)

    ax1 = fig.add_subplot(1, number_of_cells + 1, v + 1)
    ax1.set_aspect(10)
    plt.axis("off")

    colorbar = plt.colorbar()
    colorbar.set_label(colormap_label, rotation=270)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )

    fig.suptitle(title, y=1)


def plot_spatial_occupancy(
    occupancy,
    title="Occupancy of spatial bins",
    power_normalisation=1,
    transpose=True,
    colormap=plt.cm.inferno,
    colormap_label="Occupancy (seconds)",
    smoothing=None,
):
    """
    Plots occupancy of spatial bins, based on a 2D histogram
    :param occupancy: 2D histogram of occupancy
    :param title: Plot title - default: 'Occupancy of spatial bins'
    :param power_normalisation: Gamma correction parameter
    :param bool transpose: Swap axes (for numpy defaults)
    :param colormap: matplotlib.pyplot.cm colormap
    :param colormap_label: Colormap label - default: 'Occupancy (seconds)'
    :param smoothing: Kernel size for smoothing. Default: None
    """

    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    if transpose:
        occupancy = occupancy.T

    if smoothing is not None:
        logging.info("Smoothing place cell occupancy plot")
        occupancy = gaussian_filter(occupancy, smoothing)

    fig, ax = plt.subplots()

    plt.imshow(
        occupancy,
        cmap=colormap,
        interpolation="none",
        norm=colors.PowerNorm(
            power_normalisation, vmin=occupancy.min(), vmax=occupancy.max()
        ),
    ),

    colorbar = plt.colorbar()
    colorbar.set_label(colormap_label, rotation=270)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)


def plot_velocity(velocity, log=True, title="Velocity distribution"):
    logging.info("Plotting velocity histogram")

    legend_elements = []

    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Velocity (m/s) ")
    ax.set_ylabel("Time spent in bin (s)")
    plt.gca().set_title(title)
    if log:
        ax.set_yscale("log")
        ax.set_ylabel("Time spent in bin (s - log scale)")

    plt.bar(
        velocity.velocity_hist_centers,
        velocity.velocity_hist_seconds,
        width=velocity.bin_size,
        bottom=0.0,
        color=global_plot_color,
        alpha=global_plot_transparency,
    )

    if velocity.max_velocity is not None:
        ax.axvline(velocity.max_velocity, color="red", lw=2)

        el = patches.Patch(color="red", label="Maximum velocity cutoff")
        legend_elements.append(el)
        plt.legend(handles=legend_elements)


def plot_velocity_firing_rate(
    all_cells,
    colors,
    title="Firing rate at each  velocity",
    xlabel="Velocity [m/s]",
    ylabel="Firing rate [hz]",
    area=15,
    remove_zero_bins=False,
):

    logging.info("Plotting velocity firing rates")

    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    number_of_cells = len(all_cells.cell_list)
    fig = plt.figure(figsize=(number_of_cells * 3, 4))

    for idx, v in enumerate(range(number_of_cells)):
        x = all_cells.velocity_centers_in_range
        y = all_cells.velocity_cell_spikes_freq[idx]

        if remove_zero_bins:
            # remove bins with zero firing rate
            x = x[y > 0]
            y = y[y > 0]

        v = v + 1
        ax1 = fig.add_subplot(1, number_of_cells, v)
        plt.gca().set_title(all_cells.cell_list[idx], y=1)
        plt.scatter(
            x, y, s=area, c=colors[idx], alpha=global_plot_transparency
        )

        ax1.spines["right"].set_color("none")
        ax1.spines["top"].set_color("none")

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle(title, y=1)


def plot_trajectory(
    df,
    plot_heading=False,
    position="head",
    invert_y_axis=True,
    title="Trajectory",
):
    """

    :param df: Raw (or cropped) dataframe of body positions over time
    :param bool plot_heading: Plot direction of animal at each point
    :param str position: Plot position of 'head' or 'body'
    :param bool invert_y_axis: Invert the y-axis to match numpy
    :param str title: Title of plot (appended with condition name)
    :return:
    """
    logging.info("Plotting trajectory")

    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name
    fig, ax = plt.subplots()
    ax.set_title(title)

    if position is "head":
        head_left_x = df["Hear_L_x"].to_numpy().astype(float)
        head_left_y = df["Hear_L_y"].to_numpy().astype(float)
        head_right_x = df["Hear_R_x"].to_numpy().astype(float)
        head_right_y = df["Hear_R_y"].to_numpy().astype(float)
        x = (head_left_x + head_right_x).astype(float) / 2
        y = (head_left_y + head_right_y).astype(float) / 2

    elif position is "body":
        x = df.Back_x.to_numpy().astype(float)
        y = df.Back_y.to_numpy().astype(float)
    else:
        logging.error(
            'Position marker: {} is not known. Please use "head" or '
            '"body"'.format(position)
        )

    if plot_heading:
        angles = df.absolute_head_angle.to_numpy()
        vec_x = np.ones(x.shape)
        vec_y = np.ones(x.shape)
        ax.quiver(x, y, vec_x, vec_y, angles=angles, pivot="middle", scale=50)
    else:
        ax.plot(x, y, color=global_plot_color)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if invert_y_axis:
        ax.invert_yaxis()


def plot_ahv_firing_rate(
    all_cells,
    cell_specific_stats,
    colors,
    title="Firing rate at each angular head velocity",
    xlabel="Angular head velocity [deg/s]",
    ylabel="Firing rate [hz]",
    area=15,
    centre_y_axis=True,
    remove_zero_bins=False,
    plot_fit=True,
):
    logging.info("Plotting AHV firing rates")

    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    number_of_cells = len(all_cells.cell_list)
    fig = plt.figure(figsize=(number_of_cells * 3, 4))

    for idx, v in enumerate(range(number_of_cells)):
        x = all_cells.ahv_centers_in_range
        y = all_cells.ahv_cell_spikes_freq[idx]

        if remove_zero_bins:
            # remove bins with zero firing rate
            x = x[y > 0]
            y = y[y > 0]

        v = v + 1
        ax1 = fig.add_subplot(1, number_of_cells, v)
        plt.gca().set_title(all_cells.cell_list[idx], y=1)
        plt.scatter(
            x, y, s=area, c=colors[idx], alpha=global_plot_transparency
        )

        if plot_fit:
            fit_neg = np.poly1d(
                [
                    cell_specific_stats[idx].ahv.ahv_fit_slope_neg,
                    cell_specific_stats[idx].ahv.ahv_fit_intercept_neg,
                ]
            )
            fit_pos = np.poly1d(
                [
                    cell_specific_stats[idx].ahv.ahv_fit_slope_pos,
                    cell_specific_stats[idx].ahv.ahv_fit_intercept_pos,
                ]
            )

            x_pos = x[x >= 0]
            x_neg = x[x <= 0]

            plt.plot(x_neg, fit_neg(x_neg), "--k")
            plt.plot(x_pos, fit_pos(x_pos), "--k")

        if centre_y_axis:
            # set the x-spine & turn off others
            ax1.spines["left"].set_position("zero")
            ax1.spines["right"].set_color("none")
            ax1.spines["top"].set_color("none")

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle(title, y=1)


def plot_angular_velocity(
    ahv, title="Time spent in each angular velocity bin", log=True
):
    logging.info("Plotting angular velocity histogram")
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    legend_elements = []

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Angular head velocity (deg/s) ")
    ax.set_ylabel("Time spent in bin (s)")
    plt.gca().set_title(title)
    if log:
        ax.set_yscale("log")
        ax.set_ylabel("Time spent in bin (s - log scale)")

    plt.bar(
        ahv.ahv_hist_centers,
        ahv.ahv_hist_seconds,
        width=ahv.bin_size,
        bottom=0.0,
        color=global_plot_color,
        alpha=global_plot_transparency,
    )

    if ahv.central_ahv_fraction is not None or ahv.max_ahv is not None:
        ax.axvline(ahv.min_ahv, color="red", lw=2)
        ax.axvline(ahv.max_ahv, color="red", lw=2)

        el = patches.Patch(color="red", label="Central AHV fraction")
        legend_elements.append(el)

        plt.legend(handles=legend_elements)


def head_direction_plot_hist(
    time_spent_each_head_angle,
    centers,
    title="Head direction - time(s) in each bin",
    bin_size=10,
):
    logging.info("Plotting head direction radial histogram")
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    plt.bar(
        centers,
        time_spent_each_head_angle,
        width=np.deg2rad(bin_size),
        bottom=0.0,
        color=global_plot_color,
        alpha=global_plot_transparency,
    )
    plt.gca().set_title(title)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)


def head_direction_plot_kde(kde, bins, title="Head direction - KDE"):
    logging.info("Plotting head direction radial KDE")
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")

    plt.plot(bins, kde, color=global_plot_color, linewidth=5)
    plt.gca().set_title(title)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)


def plot_spikes(
    df,
    cells=None,
    colors=None,
    legend=None,
    lineoffsets=10,
    linelengths=5,
    linewidths=1,
    title="All spiking",
):
    logging.info("Plotting all spikes")
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    if cells is None:
        cells = [column for column in df.columns if "ell" in column]

    spike_array = plot_setup.get_spike_time_list(df, cells)

    if colors is None:
        colors = get_random_colors(len(cells))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time (s)")
    ax.get_yaxis().set_visible(False)

    ax.eventplot(
        spike_array,
        colors=colors,
        lineoffsets=lineoffsets,
        linelengths=linelengths,
        linewidths=linewidths,
    )

    if legend is None:
        legend = make_legend(cells, colors)
    ax.legend(handles=legend, loc="upper left")

    plt.gca().set_title(title)


def behaviour_plot_all(
    df, height=2, aspect=8, filter_window=5, title="Behaviour over time"
):
    logging.info("Plotting all behaviour")
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    # TODO: remove hard codes
    tidy_df = pd.DataFrame(index=df.index)

    if filter_window is not None:
        tidy_df["Total Speed (median filtered)"] = (
            df["total_speed"]
            .rolling(filter_window, center=True, min_periods=1)
            .median()
        )
        tidy_df["Unwrapped Angle (median filtered)"] = (
            df["unwrapped_angle"]
            .rolling(filter_window, center=True, min_periods=1)
            .median()
        )
        tidy_df["Angular Velocity (median filtered)"] = (
            df["angular_head_velocity"]
            .rolling(filter_window, center=True, min_periods=1)
            .median()
        )
    else:
        tidy_df["Total Speed"] = df.total_speed
        tidy_df["Unwrapped Angle"] = df.unwrapped_angle
        tidy_df["Angular Velocity"] = df.angular_head_velocity

    tidy_df["Time (s)"] = pd.to_numeric(tidy_df.index)

    tidy_df = tidy_df.melt("Time (s)", var_name="Measure", value_name="vals")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    g = sns.FacetGrid(
        tidy_df, row="Measure", height=height, aspect=aspect, sharey=False
    )
    g = g.map(plt.plot, "Time (s)", "vals", color=global_plot_color)
    plt.subplots_adjust(top=0.85)
    g.set_titles(row_template="{row_name}")
    g.fig.suptitle(title)
    g.axes[0, 0].set_ylabel("m/s")
    g.axes[1, 0].set_ylabel("Degrees")
    g.axes[2, 0].set_ylabel("Degrees/s")


def plot_all_direction(
    all_cells,
    cell_colors,
    normalise_overlay=True,
    weighted=False,
    legend=None,
    overlay=True,
    subplot=True,
    bin_size=6,
    time_spent_each_bin=None,
    plot_type="both",
):

    # TODO: check weights etc works properly
    if not weighted:
        weights = None
    else:
        weights = all_cells.hd_weights

    if overlay:
        logging.info("Plotting overlay of firing directionality")

        if plot_type in ["hist", "both"]:

            (
                spikes_per_bin,
                hist_bar_centers,
            ) = radial_spike_histogram_multiple(
                all_cells.hd_list_angles,
                weights,
                bin_width=bin_size,
                normalise=normalise_overlay,
                bin_occupancy=time_spent_each_bin,
            )
            polar_hist_overlay(
                hist_bar_centers,
                spikes_per_bin,
                all_cells.cell_list,
                bin_size=bin_size,
                colors=cell_colors,
                legend=legend,
            )

        if plot_type in ["kde", "both"]:
            polar_kde_overlay(
                all_cells.hd_kde_centers,
                all_cells.hd_kde_spikes,
                all_cells.cell_list,
                colors=cell_colors,
                legend=legend,
            )

    if subplot:
        logging.info("Plotting subplots of firing directionality")

        if plot_type in ["hist", "both"]:

            polar_hist_subplot(
                all_cells.hd_hist_bar_centers,
                all_cells.hd_spikes_per_bin,
                all_cells.cell_list,
                bin_size=bin_size,
                colors=cell_colors,
            )

        if plot_type in ["kde", "both"]:
            polar_kde_subplot(
                all_cells.hd_kde_centers,
                all_cells.hd_kde_spikes,
                all_cells.cell_list,
                colors=cell_colors,
            )


def polar_kde_overlay(
    kde_centers,
    kde_spikes,
    cell_names,
    colors=None,
    legend=None,
    linewidth=3,
    title="Direction tuning firing in Hz",
):
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    number_of_cells = len(cell_names)
    if colors is None:
        colors = get_random_colors(number_of_cells)

    if legend is not None:
        legend = make_legend(cell_names, colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.get_yaxis().set_visible(False)
    for idx, v in enumerate(range(number_of_cells)):
        plt.plot(
            kde_centers[idx],
            kde_spikes[idx],
            color=colors[idx],
            linewidth=linewidth,
            alpha=global_plot_transparency,
        )

        plt.gca().set_title(title)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.legend(handles=legend, loc="upper left")


def polar_kde_subplot(
    kde_centers,
    kde_spikes,
    cell_names,
    colors=None,
    linewidth=3,
    title="Direction tuning firing in Hz",
):
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    number_of_cells = len(cell_names)

    if colors is None:
        colors = get_random_colors(number_of_cells)

    fig = plt.figure(figsize=(number_of_cells * 3, 5))
    fig.suptitle(title)

    for idx, v in enumerate(range(number_of_cells)):
        v = v + 1
        ax1 = fig.add_subplot(1, number_of_cells, v, projection="polar")
        plt.plot(
            kde_centers[idx],
            kde_spikes[idx],
            color=colors[idx],
            linewidth=linewidth,
            alpha=global_plot_transparency,
        )

        plt.gca().set_title(cell_names[idx], y=1.3)
        ax1.set_theta_zero_location("E")
        ax1.set_theta_direction(1)
    plt.subplots_adjust(wspace=0.5)


def polar_hist_subplot(
    hist_bar_centers,
    spikes_per_bin,
    cell_names,
    bin_size=6,
    colors=None,
    title="Direction tuning firing in Hz",
):
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    number_of_cells = len(cell_names)

    if colors is None:
        colors = get_random_colors(number_of_cells)

    fig = plt.figure(figsize=(number_of_cells * 3, 5))
    fig.suptitle(title)

    for idx, v in enumerate(range(number_of_cells)):
        v = v + 1
        ax1 = fig.add_subplot(1, number_of_cells, v, projection="polar")
        plt.bar(
            hist_bar_centers,
            spikes_per_bin[idx],
            width=np.deg2rad(bin_size),
            bottom=0.0,
            color=colors[idx],
            alpha=global_plot_more_transparent,
        )
        plt.gca().set_title(cell_names[idx], y=1.3)
        ax1.set_theta_zero_location("E")
        ax1.set_theta_direction(1)
    plt.subplots_adjust(wspace=0.5)


def polar_hist_overlay(
    hist_bar_centers,
    spikes_per_bin,
    cell_names,
    bin_size=6,
    colors=None,
    legend=None,
    title="Direction tuning, (normalised)",
):
    if "global_condition_name" in globals():
        title = title + " - " + global_condition_name

    number_of_cells = len(cell_names)

    if colors is None:
        colors = get_random_colors(number_of_cells)

    if legend is not None:
        legend = make_legend(cell_names, colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.get_yaxis().set_visible(False)
    for idx, v in enumerate(range(number_of_cells)):

        plt.bar(
            hist_bar_centers,
            spikes_per_bin[idx],
            width=np.deg2rad(bin_size),
            bottom=0.0,
            color=colors[idx],
            alpha=global_plot_more_transparent,
        )
        plt.gca().set_title(title)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.legend(handles=legend, loc="upper left")
