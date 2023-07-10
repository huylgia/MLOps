import pandas as pd
import numpy as np

def handle_duplicate(data: pd.DataFrame):
    labels = data['label']
    data.drop(columns=['label'], inplace=True)

    # get duplicated value
    duplicated = data[data.duplicated(keep=False)]
    if len(duplicated) == 0:
        data['labels'] = labels
        return data
    
    duplicated['label'] = labels.values[list(duplicated.index)]
    duplicated['index'] = list(duplicated.index)

    # aggerate labels of each duplicated
    agg = duplicated.groupby(list(duplicated.columns[:-2])).agg(list)
    for (lb, index) in agg.values:
        if len(np.unique(lb)) > 1:
            data.drop(index=index, inplace=True)

    # drop rest duplicated rows
    data.drop_duplicates(inplace=True)

    # re-assign label
    data['label'] = labels.values[list(data.index)]

    return data
