import pandas as pd

labeled_data = pd.read_parquet("/mnt/ai_data/HuyAI/MLOps/samples/phase-2/prob-2/train/raw_train_2.parquet", engine="pyarrow")

print(labeled_data.shape)
print(labeled_data.drop_duplicates(inplace=True))
print(labeled_data.shape)
