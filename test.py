import pandas
import numpy as np
from pathlib import Path
from data import load_tranformer
from sklearn.cluster import OPTICS
import pickle

def compute_metric(optics_result):
    REPORT = {}

    optics_clusters = np.unique(optics_result)
    for optics_cluster in optics_clusters:
        index = np.where(optics_result == optics_cluster)

        unique, counts = np.unique(Y[index], return_counts=True)
        counts = counts/sum(counts)

        for label, count in zip(unique, counts):
            if label not in REPORT:
                REPORT[label] = []
            
            if count == max(counts):
                REPORT[label].append(count)
    
    for key in REPORT.keys():
        REPORT[key] = {
            '>0.6': len([v for v in REPORT[key] if v >= 0.6])/len(REPORT[key]),
            '>0.7': len([v for v in REPORT[key] if v >= 0.7])/len(REPORT[key]),
            '>0.8': len([v for v in REPORT[key] if v >= 0.8])/len(REPORT[key]),
            '>0.9': len([v for v in REPORT[key] if v >= 0.9])/len(REPORT[key])
        }
    
    print("Total cluster: ", len(optics_clusters))
    print(REPORT)

if __name__ == "__main__":
    work_dir = Path("samples/phase-2/prob-2/")

    data = pandas.read_parquet(str(work_dir/"train"/"raw_train.parquet"), engine="pyarrow")
    data = data.sample(frac=1, random_state=42)

    category_transformer = load_tranformer(work_dir/"transformer", "category_transformer")
    numeric_transformer  = load_tranformer(work_dir/"transformer", "numeric_transformer")

    for col in category_transformer.columns:
        data = category_transformer.transform(data, col)

    for col in numeric_transformer.columns:
        data = numeric_transformer.transform(data, col)

    Y = data['label'].values
    X = data.drop(columns=["label"]).values

    # from sklearn.model_selection import train_test_split
    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(X.shape)
    print("Start clustering")

    for eps in np.arange(0.5,0.9,0.2):
        for min_sample in range(2, 4):
            print(f"===================={eps}-{min_sample}=======================")
            optics_model = OPTICS(eps=eps, min_samples=min_sample, metric="minkowski", n_jobs=8, memory="./model", cluster_method="dbscan")
            optics_result = optics_model.fit_predict(X)

            compute_metric(optics_result)
    
    # optics_result = optics_model.fit_predict(X[10000:20000])
    # compute_metric(optics_result)

    # optics_result = optics_model.fit_predict(X[20000:30000])
    # compute_metric(optics_result)
 



 


    
