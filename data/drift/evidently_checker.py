import pandas
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from typing import Tuple, Mapping, Any

def report_data_drift(reference: pandas.DataFrame, current: pandas.DataFrame) -> Report:
    report = Report(metrics=[DataDriftTable()])
    report.run(
        reference_data=reference, 
        current_data=current
    )

    return report.as_dict()

def check_data_drift(reference: pandas.DataFrame, current: pandas.DataFrame) -> Tuple[bool, Mapping[str, Any]]:
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(
        reference_data=reference, 
        current_data=current
    )

    # convert to dictionary
    report = report.as_dict()['metrics'][0]
    if report['result']['dataset_drift']:
        return True, report
    return False, report