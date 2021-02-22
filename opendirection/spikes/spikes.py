import pandas as pd
import numpy as np
import logging

from spikey.io.spikesort import ks_phy_load
from spikey.curate.curate import get_good_spikes
from opendirection.tools.io import load_sync_parameters


def get_spikes(
    spike_time_file,
    spike_clusters_file,
    cluster_groups_file,
    sync_param_file,
    num_behaviour_measurements,
    camera_hz,
    probe_hz,
):
    logging.info("Loading spikes")
    spikes_df, cluster_groups = ks_phy_load(
        spike_time_file, spike_clusters_file, cluster_groups_file
    )
    ratio_mantis_spike, delay_mantis_spike = load_sync_parameters(
        sync_param_file
    )

    logging.info("Cleaning spikes")
    clean_spikes = get_good_spikes(spikes_df, cluster_groups)
    logging.info("Synchronising spikes")
    synchronised_spikes = synchronise(
        clean_spikes,
        ratio_mantis_spike,
        delay_mantis_spike,
        camera_hz,
        probe_hz,
        num_behaviour_measurements,
    )
    return synchronised_spikes


def synchronise(
    df, ratio, delay, camera_hz, probe_hz, num_behaviour_measurements
):
    logging.info("Synchronising spike times")
    # TODO: change to type to support negative numbers
    df = df[df["spike_times"] > delay]
    df.spike_times = df.spike_times - delay
    df.spike_times = df.spike_times * ratio
    max_frames_allowed = (num_behaviour_measurements * probe_hz) / camera_hz
    df = df[df["spike_times"] < max_frames_allowed]
    df = real_time(df, num_behaviour_measurements, camera_hz, probe_hz)
    return df


def real_time(df, num_behaviour_measurements, camera_hz, probe_hz):
    logging.debug("Binning spike times")
    df["time_bin"] = df["spike_times"] / probe_hz
    bin_start = 0.5 * (1 / camera_hz)
    bin_end = bin_start + (num_behaviour_measurements / camera_hz)
    bin_range = np.arange(bin_start, bin_end, (1 / camera_hz))
    # label_end = num_behaviour_measurements / camera_hz
    # label_range = np.arange((1 / camera_hz), label_end, (1 / camera_hz))

    # get midpoints of bins as labels
    label_range = bin_range + bin_start
    label_range = label_range[0:-1]

    df["time_bin"] = pd.cut(df["time_bin"], bin_range, labels=label_range)
    df.time_bin = df.time_bin.to_numpy()
    df = df[np.isfinite(df["time_bin"])]
    df.time_bin = df.time_bin.round(3)
    return df
