from typing import Tuple, Mapping, List, Any, Dict
import numpy as np

import pandas as pd
from scipy import stats
from scipy.spatial import distance

from .utils import get_binned_data


class StatTest:
    def __init__(self, drift_data_threshold=0.5, drift_column_threshold: float=0.1) -> None:
        self.drift_column_threshold = drift_column_threshold
        self.drift_data_threshold   = drift_data_threshold

    def detect_drift_data(
            self, 
            reference_data: Mapping[str, List[Any]],
            current_data: Mapping[str, List[Any]],
            feature_config: Dict
        ) -> Tuple[bool,float]:
        total_column_drift = 0

        total_column = len(reference_data.keys())

        # numeric column
        for column in feature_config['numeric_columns']:
            if column not in reference_data and column not in current_data:
                continue
            
            _, is_drift = self.detect_drift_column(
                    reference_data=reference_data[column],
                    current_data=current_data[column],
                    feature_type="num"
            )
            
            total_column_drift += 1 if is_drift else 0
            if total_column_drift/total_column >= self.drift_data_threshold:
                return True
        
        # category column
        for column in feature_config['category_columns']:
            if column not in reference_data and column not in current_data:
                continue
           
            _, is_drift = self.detect_drift_column(
                    reference_data=reference_data[column],
                    current_data=current_data[column],
                    feature_type="cat"
            )

            total_column_drift += 1 if is_drift else 0
            if total_column_drift/total_column >= self.drift_data_threshold:
                return True

        return False
 
               
    def detect_drift_column(self, reference_data: List[Any], current_data: List[Any], feature_type: str):
        # get statest based on feature type
        if feature_type == "num":
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
            threshold=self.drift_column_threshold
        )

        return drift_score, is_drift
    
    @staticmethod
    def _jensenshannon(reference_data: List[Any], current_data: List[Any], threshold: float) -> Tuple[float, bool]:
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
        reference_percents, current_percents = get_binned_data(reference_data, current_data, False)
        
        jensenshannon_value = distance.jensenshannon(reference_percents, current_percents)
        return jensenshannon_value, jensenshannon_value >= threshold

    @staticmethod
    def _wasserstein_distance_norm(reference_data: List[Any], current_data: List[Any], threshold: float) -> Tuple[float, bool]:
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