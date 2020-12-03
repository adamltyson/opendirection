import numpy as np
import math

from imlib.general.numerical import get_decimal_places, round_updown_to_x
from imlib.general.parsing import check_none


def get_bins(
    nums, bin_size, deal_with_decimals=True, min_val=None, max_val=None
):
    """
    :param nums: Array of numbers to create bins from
    :param bin_size: How big do you want the bins to be
    :param deal_with_decimals: Default true. If false, will not deal well
    with bin_size with decimal places. Will be faster though
    :param min_val: Specify the minimum value (rather than nums.min())
    :param max_val: Specify the maximum value (rather than nums.max())
    :return: Bin edges
    """
    # FIXME: Clumsy implementation of deal_with_decimals=True

    if min_val is None:
        min_val = nums.min()
    if max_val is None:
        max_val = nums.max()

    if deal_with_decimals:
        original_bin_size = bin_size
        num_decimal_places = get_decimal_places(bin_size)
        # nums = nums * (10 ** num_decimal_places)
        bin_size = bin_size * 10 ** num_decimal_places
        min_val = min_val * (10 ** num_decimal_places)
        max_val = max_val * (10 ** num_decimal_places)

    bin_min = int(round_updown_to_x(min_val, bin_size, direction="down"))
    bin_max = int(round_updown_to_x(max_val, bin_size, direction="up"))

    if deal_with_decimals:
        bin_min = bin_min / (10 ** num_decimal_places)
        bin_max = bin_max / (10 ** num_decimal_places)
        bin_size = original_bin_size

    bin_edges = np.arange(bin_min, bin_max + bin_size, bin_size)
    if deal_with_decimals:
        bin_edges = np.round(bin_edges, decimals=num_decimal_places)
    return bin_edges


def keep_n_central_bins(hist, bin_centers, fraction_keep=0.95):
    """
    Given a histogram, and the central values of the histogram bins, return
    the maximum and minimum bin values to keep a given fraction of the data.
    :param hist: 1D histogram of data
    :param bin_centers:  1D array of hsitogram bin centers
    :param fraction_keep: What fraction of data to keep. Default: 0.95
    :return tuple: Minimum and maximum bin values.
    """
    # TODO: tidy up
    bin_width = bin_centers[1] - bin_centers[0]
    hist_centre_val = None
    if len(hist) % 2 == 1:  # if odd
        center = int(math.ceil(len(hist) / 2))
        hist_centre_val = int(hist[center - 1])
        hist = np.delete(hist, center)
        bin_centers = np.delete(bin_centers, center)

    half = len(hist) // 2
    hist_half_1st = np.flip(hist[:half])
    hist_half_2nd = hist[half:]

    centers_half_1st = np.flip(bin_centers[:half])
    centers_half_2nd = bin_centers[half:]

    mirror_merge = np.add(hist_half_1st, hist_half_2nd)

    if len(hist) % 2 == 1:
        mirror_merge = np.insert(mirror_merge, 0, hist_centre_val)

    cumulative = np.cumsum(mirror_merge)

    total_keep = fraction_keep * hist.sum()

    bins_keep = cumulative < total_keep

    if len(hist) % 2 == 1:
        bins_keep = np.delete(bins_keep, 0)

    centers_keep_1st = centers_half_1st[bins_keep]
    centers_keep_2nd = centers_half_2nd[bins_keep]

    min_keep = centers_keep_1st.min() - (bin_width / 2)
    max_keep = centers_keep_2nd.max() + (bin_width / 2)

    return min_keep, max_keep


def get_cleaned_attrs_from_object(
    object_in, ignore_callable=True, ignore_types=None, ignore_prefixes="__"
):
    """
    Return all the attributes of an object that pass criteria
    :param object_in: Any python object
    the attributes with the "sub_object" they belong to. Default: False
    :param ignore_callable: Don't return attributes than can be called.
    Default: True
    :param ignore_types: Don't return attributes of this type
    (or tuple of types). Default: None.
    :param ignore_prefixes: Don't return attributes starting with prefixes
     (as strings) in this list. Default: "__"
    :return: list of attributes meeting criteria
    """

    attributes = [attr for attr in dir(object_in)]

    if ignore_callable:
        attributes = [
            attr
            for attr in attributes
            if not callable(getattr(object_in, attr))
        ]

    if ignore_types is not None:
        attributes = [
            attr
            for attr in attributes
            if not isinstance(getattr(object_in, attr), ignore_types)
        ]

    if ignore_prefixes is not None:
        for prefix in ignore_prefixes:
            attributes = [
                attr for attr in attributes if not attr.startswith(prefix)
            ]

    return attributes


def get_attrs_multiple_sub_objects(
    main_object,
    sub_objects,
    return_dict_with_sub_objects=False,
    ignore_callable=True,
    ignore_types=None,
    ignore_prefixes="__",
):
    """
    For a "main_object" with given "sub_objects", return all the attributes
    of those "sub_objects" that pass criteria
    :param main_object: Any python object
    :param sub_objects: List of sub_object names as strings
    :param return_dict_with_sub_objects: If True, returns a dict linking
    the attributes with the "sub_object" they belong to. Default: False
    :param ignore_callable: Don't return attributes than can be called.
    Default: True
    :param ignore_types: Don't return attributes of this type
    (or tuple of types). Default: None.
    :param ignore_prefixes: Don't return attributes starting with prefixes
     (as strings) in this list. Default: "__"
    :return: List (optionally a dict with associated sub_objects) of attributes
    meeting criteria.
    """

    if return_dict_with_sub_objects:
        attributes = {}
    else:
        attributes = []

    for sub_object_name in sub_objects:
        full_object = getattr(main_object, sub_object_name)

        attr_list = [attr for attr in dir(full_object)]

        if ignore_callable:
            attr_list = [
                attr
                for attr in attr_list
                if not callable(getattr(full_object, attr))
            ]

        if ignore_types is not None:
            attr_list = [
                attr
                for attr in attr_list
                if not isinstance(getattr(full_object, attr), ignore_types)
            ]

        if ignore_prefixes is not None:
            for prefix in ignore_prefixes:
                attr_list = [
                    attr for attr in attr_list if not attr.startswith(prefix)
                ]

        for attr in attr_list:
            if return_dict_with_sub_objects:
                attributes.update({attr: sub_object_name})
            else:
                attributes.append(attr)

    return attributes


def shuffle_distances_in_units(distance, calibration, to_be_shuffled):
    """
    Based on a distance in "real" units, return how far to shuffle in
    "array" space. If no distance (or no calibration factor) given, returns
    the length of the array to be shuffled.

    :param distance: How far (in real units to shuffle)
    :param calibration: Real units -> array units calibration
    :param to_be_shuffled: Array to be shuffled
    :return: Distance in array units to shuffle
    """
    if check_none(distance, calibration):
        distance = len(to_be_shuffled)
    else:
        distance = int(round(distance * calibration))

    return distance
