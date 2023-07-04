import numpy as np
import pandas as pd

def get_unique_not_nan_values_list_from_series(current_data: pd.Series, reference_data: pd.Series) -> list:
    """Get unique values from current and reference series, drop NaNs"""
    return list(set(reference_data.dropna().unique()) | set(current_data.dropna().unique()))

def get_binned_data(reference_data: pd.Series, current_data: pd.Series, feature_type: str, n: int, feel_zeroes: bool = True):
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
    if feature_type == "num" and reference_data.nunique() > 20:
        bins = np.percentile(np.concatenate([reference_data.values, current_data.values]), np.linspace(0, 100, n+1))
        reference_counts, _ = np.histogram(reference_data, bins)
        current_counts, _ = np.histogram(current_data, bins)
        reference_percents = reference_counts / len(reference_data)
        current_percents = current_counts / len(current_data)
    else:
        keys = get_unique_not_nan_values_list_from_series(current_data=current_data, reference_data=reference_data)
        ref_feature_dict = {key: 0 for key in keys}
        current_feature_dict = {key: 0 for key in keys}
        ref_feature_dict.update(reference_data.value_counts())
        current_feature_dict.update(current_data.value_counts())
        reference_percents = np.array([ref_feature_dict[key] / len(reference_data) for key in keys])
        current_percents = np.array([current_feature_dict[key] / len(current_data) for key in keys])

    if feel_zeroes:
        min_ref_percent = min(reference_percents[reference_percents != 0])
        min_current_percent = min(current_percents[current_percents != 0])
        np.place(reference_percents, reference_percents == 0, min_ref_percent / 10**6 if min_ref_percent <= 0.0001 else 0.0001)
        np.place(current_percents, current_percents == 0, min_current_percent / 10**6 if min_current_percent <= 0.0001 else 0.0001)

    return reference_percents, current_percents
