import pandas as pd
from pathlib import Path

data_1 = pd.concat(map(pd.read_csv, Path("/home/busmap/MLOps/samples/phase-2/prob-1/test/7_4_12_28").glob("*.csv")))
data_2 = pd.concat(map(pd.read_csv, Path("/home/busmap/MLOps/samples/phase-2/prob-1/test/7_9_4_18").glob("*.csv")))

print(data_2)
print(data_1.drop_duplicates().shape)
print(pd.concat([data_1, data_2]).drop_duplicates().shape)