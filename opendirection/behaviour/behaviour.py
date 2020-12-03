import numpy as np
import logging

from imlib.radial.misc import phase_unwrap
from imlib.array.math import calculate_gradient_1d
from movement.io.dlc import load_and_clean_dlc
from movement.position.angles import angle_from_points
from movement.movement.velocity import calculate_speed

from opendirection.combine.calculate_general import get_positions


def get_behaviour(
    dlc_files,
    real_length_to_pixel_conversion,
    camera_hz,
    options,
    regex_columns_to_remove=["likelihood", "bodyparts"],
):
    logging.info("Loading behavioural data")
    behaviour_data = load_and_clean_dlc(
        dlc_files, regex_remove_columns=regex_columns_to_remove
    )
    logging.info("Calculating behaviours")
    behaviour_data = calculate_behaviours(
        behaviour_data,
        conversion_factor=real_length_to_pixel_conversion,
        camera_hz=camera_hz,
        hd_filter=options.ang_vel_filter_hd_filt_width,
        velocity_filter=options.velocity_smooth_sigma,
        local_ang_vel_deriv_window=options.ang_vel_local_derivative_window,
        use_head_as_position_velocity=options.velocity_position_head,
    )
    return behaviour_data


def calculate_behaviours(
    df,
    conversion_factor=0.002,
    camera_hz=40,
    hd_filter=None,
    velocity_filter=None,
    local_ang_vel_deriv_window=0.2,
    use_head_as_position_velocity=False,
):

    num_behaviour_measurements = len(df.index)
    df = speed(
        df,
        conversion_factor=conversion_factor,
        camera_hz=camera_hz,
        velocity_filter=velocity_filter,
        use_head_as_position_velocity=use_head_as_position_velocity,
    )
    df = calculate_head_angle_velocity(
        df,
        camera_hz=camera_hz,
        hd_filter=hd_filter,
        local_ang_vel_deriv_window=local_ang_vel_deriv_window,
    )
    df = add_times(df, num_behaviour_measurements, camera_hz)
    return df


def speed(
    df,
    conversion_factor=0.002,
    camera_hz=40,
    velocity_filter=None,
    min_periods=1,
    use_head_as_position_velocity=False,
):
    pixel_per_frame_speed = conversion_factor * camera_hz

    x, y = get_positions(
        df, use_head_as_position=use_head_as_position_velocity
    )
    total_speed = calculate_speed(
        x, y, conversion_factor=pixel_per_frame_speed,
    )

    # total_speed = calculate_speed(
    #     df["Back_x"].to_numpy().astype(float),
    #     df["Back_y"].to_numpy().astype(float),
    #     conversion_factor=pixel_per_frame_speed,
    # )

    df["total_speed"] = total_speed
    if velocity_filter:
        filter_frames = int(np.round(velocity_filter * camera_hz))
        df["total_speed"] = (
            df["total_speed"]
            .rolling(filter_frames, center=True, min_periods=min_periods)
            .median()
        )

    return df


def calculate_head_angle_velocity(
    df,
    camera_hz=40,
    hd_filter=None,
    local_ang_vel_deriv_window=0.2,
    min_periods=1,
):

    ear_positions = np.empty((4, len(df)))
    ear_positions[0, :] = df["Hear_L_x"].to_numpy().astype(float)
    ear_positions[1, :] = df["Hear_L_y"].to_numpy().astype(float)
    ear_positions[2, :] = df["Hear_R_x"].to_numpy().astype(float)
    ear_positions[3, :] = df["Hear_R_y"].to_numpy().astype(float)

    df["absolute_head_angle"] = angle_from_points(ear_positions)
    df["unwrapped_angle"] = phase_unwrap(df["absolute_head_angle"])

    velocity_at_t0 = np.array(0)  # to match length
    angular_head_velocity = np.append(
        velocity_at_t0, np.diff(df["unwrapped_angle"])
    )
    # convert to deg/s
    # instantaneous (x(t) - x(t-1))
    df["angular_head_velocity_instantaneous"] = (
        angular_head_velocity * camera_hz
    )

    if hd_filter:
        filter_frames = int(np.round(hd_filter * camera_hz))
        df["unwrapped_angle"] = (
            df["unwrapped_angle"]
            .rolling(filter_frames, center=True, min_periods=min_periods)
            .median()
        )

    # based on window
    ang_vel_window = int(round(local_ang_vel_deriv_window * camera_hz))
    df["angular_head_velocity"] = (
        df["unwrapped_angle"]
        .rolling(ang_vel_window, min_periods=1)
        .apply(calculate_gradient_1d, raw=True)
        * camera_hz
    )

    return df


def add_times(df, num_behaviour_measurements, camera_hz):
    end_time = num_behaviour_measurements / camera_hz
    time_range = np.arange(0, end_time, (1 / camera_hz))
    df["time"] = time_range
    return df
