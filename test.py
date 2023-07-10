import pandas as pd
from pathlib import Path
import numpy as np

data = pd.read_parquet("/mnt/ai_data/HuyAI/MLOps/samples/phase-2/prob-1/train/raw_train.parquet", engine="pyarrow")
label = data['label']
data.drop(columns=['label'], inplace=True)


duplicated = data[data.duplicated(keep=False)]
duplicated['label'] = label[duplicated.index]
duplicated['index'] = duplicated.index

agg = duplicated.groupby(list(duplicated.columns[:-2])).agg(list)

for (lb, index) in agg.values:
    if len(np.unique(lb)) > 1:
        data.drop(index=index, inplace=True)


data.drop_duplicates(inplace=True)
data['label'] = label.values[list(data.index)]

print(data.shape)

# print(data.shape)
# print(data.drop_duplicates(keep=False).shape)
