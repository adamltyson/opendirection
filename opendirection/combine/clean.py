def clean_df(df, speed_cutoff=None, copy=False):
    # remove areas outside the experimental conditions
    # remove low movement areas
    if speed_cutoff is not None:
        df = remove_low_speeds(df, speed_cutoff)
    if copy:
        return df.copy()
    else:
        return df


def remove_low_speeds(df, speed_cutoff):
    df = df[df["total_speed"] > speed_cutoff]
    return df
