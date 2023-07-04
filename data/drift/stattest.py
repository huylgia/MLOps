from typing import Tuple, Dict
import numpy as np

import pandas as pd
from scipy import stats
from scipy.spatial import distance

from .utils import get_binned_data

class StatTest:
    def __init__(self, drift_data_threshold=0.5, drift_column_threshold: float=0.1) -> None:
        self.drift_column_threshold = drift_column_threshold
        self.drift_data_threshold   = drift_data_threshold

    def detect_drift_data(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_config: Dict) -> Tuple[bool,float]:
        keys = ['numeric_columns', 'category_columns']

        total_column = 0
        total_column_drift = 0
        for key in keys:
            feature_type = key[:3]

            for column in feature_config[key]:
                if feature_type == "cat":
                    reference_df[column] = reference_df[column].astype("category") 
                    current_df[column] = current_df[column].astype("category") 
                else:
                    reference_df[column] = pd.to_numeric(reference_df[column])
                    current_df[column] = pd.to_numeric(current_df[column])

                _, is_drift = self.detect_drift_column(
                    reference_data=reference_df[column],
                    current_data=current_df[column],
                    feature_type=feature_type
                )
                
                total_column += 1
                total_column_drift += 1 if is_drift else 0

        drift_data_score = total_column_drift/total_column

        if drift_data_score >= self.drift_data_threshold:
            return True, drift_data_score
        else:
            return False, drift_data_score
               
    def detect_drift_column(self, reference_data: pd.Series, current_data: pd.Series, feature_type: str):
        n_values = pd.concat([reference_data, current_data]).nunique()

        # get statest based on feature type
        if feature_type == "num":
            if n_values <= 5:
                stattest = self._jensenshannon
            else:
                stattest = self._wasserstein_distance_norm
        
        elif feature_type == "cat":
            stattest = self._jensenshannon
        else:
            message = f"Not implement drift detection for {feature_type}"
            raise Exception(message)

        # do detect drift
        drift_score, is_drift = stattest(
            reference_data=reference_data, 
            current_data=current_data,
            feature_type=feature_type,
            threshold=self.drift_column_threshold
        )

        return drift_score, is_drift
    
    @staticmethod
    def _jensenshannon(reference_data: pd.Series, current_data: pd.Series, feature_type: str, threshold: float, n_bins: int = 30) -> Tuple[float, bool]:
        """Compute the Jensen-Shannon distance between two arrays
        Args:
            reference_data: reference data
            current_data: current data
            feature_type: feature type
            threshold: all values above this threshold means data drift
            n_bins: number of bins
        Returns:
            jensenshannon: calculated Jensen-Shannon distance
            test_result: whether the drift is detected
        """
        reference_percents, current_percents = get_binned_data(reference_data, current_data, feature_type, n_bins, False)
        
        jensenshannon_value = distance.jensenshannon(reference_percents, current_percents)
        return jensenshannon_value, jensenshannon_value >= threshold

    @staticmethod
    def _wasserstein_distance_norm(reference_data: pd.Series, current_data: pd.Series, feature_type: str, threshold: float) -> Tuple[float, bool]:
        """Compute the first Wasserstein distance between two arrays normed by std of reference data
        Args:
            reference_data: reference data
            current_data: current data
            feature_type: feature type
            threshold: all values above this threshold means data drift
        Returns:
            wasserstein_distance_norm: normed Wasserstein distance
            test_result: whether the drift is detected
        """
        norm = max(np.std(reference_data), 0.001)

        wd_norm_value = stats.wasserstein_distance(reference_data, current_data) / norm
        return wd_norm_value, wd_norm_value >= threshold