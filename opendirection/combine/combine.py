import pandas as pd
import logging


def tidy_behaviour(df):
    df_new = pd.DataFrame()
    df_new["time"] = df.time
    df_new["Back_x"] = df.Back_x
    df_new["Back_y"] = df.Back_y
    df_new["total_speed"] = df.total_speed
    df_new["absolute_head_angle"] = df.absolute_head_angle
    df_new["head_velocity"] = df.Back_x
    df_new = df.set_index("time")
    return df_new


def make_combined_df(spike_df, dlc_data):
    # Make a single df with all relevant behaviour and ephys data
    logging.info("Aligning behaviour and spike data")
    # Make a new dataframe without unecessary dlc info
    df = tidy_behaviour(dlc_data)
    logging.debug("Counting spikes for each cluster")
    # Count spikes per time bin per cell
    clusters = spike_df.spike_clusters.unique()
    num_clusters = len(clusters)
    for idx, cluster in enumerate(clusters):
        logging.debug(
            "Counting spikes for cluster {} (ID: {}) of  {}".format(
                idx + 1, cluster, num_clusters
            )
        )
        time_bins = spike_df["time_bin"][spike_df["spike_clusters"] == cluster]
        binned_times = time_bins.value_counts()
        df = df.join(binned_times)
        new_name = "cell_" + str(cluster)
        df.rename(columns={"time_bin": new_name}, inplace=True)

    df = df.fillna(value=0)
    return df


def make_aligned_spike_df(spike_df, dlc_data):
    # Make a single df with all relevant behaviour and ephys data
    logging.info("Aligning behaviour and spike data")
    # Make a new dataframe without unecessary dlc info
    aligned_behaviour_df = tidy_behaviour(dlc_data)
    df = pd.DataFrame(index=aligned_behaviour_df.index)
    logging.debug("Counting spikes for each cluster")
    # Count spikes per time bin per cell
    clusters = spike_df.spike_clusters.unique()
    num_clusters = len(clusters)

    for idx, cluster in enumerate(clusters):
        logging.debug(
            "Counting spikes for cluster {} (ID: {}) of  {}".format(
                idx + 1, cluster, num_clusters
            )
        )
        time_bins = spike_df["time_bin"][spike_df["spike_clusters"] == cluster]
        binned_times = time_bins.value_counts()
        df = df.join(binned_times)
        new_name = "cell_" + str(cluster)
        df.rename(columns={"time_bin": new_name}, inplace=True)

    aligned_spikes_df = df.fillna(value=0)
    return aligned_spikes_df, aligned_behaviour_df
