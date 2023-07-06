from typing import List, Any
import numpy as np

def get_binned_data(reference_data: List[Any], current_data: List[Any], feel_zeroes: bool = True):
    """Split variable into n buckets based on reference quantiles
    Args:
        reference_data: reference data
        current_data: current data
        feature_type: feature type
        n: number of quantiles
    Returns:
        reference_percents: % of records in each bucket for reference
        current_percents: % of records in each bucket for current
    """
    keys = set(current_data + current_data)
    ref_feature_dict = {key: reference_data.count(key) for key in keys if key not in [np.nan, None]}
    current_feature_dict = {key: current_data.count(key) for key in keys if key not in [np.nan, None]}
    
    reference_percents = np.array([ref_feature_dict[key] / len(reference_data) for key in keys if key not in [np.nan, None]])
    current_percents = np.array([current_feature_dict[key] / len(current_data) for key in keys if key not in [np.nan, None]])

    if feel_zeroes:
        min_ref_percent = min(reference_percents[reference_percents != 0])
        min_current_percent = min(current_percents[current_percents != 0])
        np.place(reference_percents, reference_percents == 0, min_ref_percent / 10**6 if min_ref_percent <= 0.0001 else 0.0001)
        np.place(current_percents, current_percents == 0, min_current_percent / 10**6 if min_current_percent <= 0.0001 else 0.0001)

    return reference_percents, current_percents
