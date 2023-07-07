import pandas
import numpy as np
from pathlib import Path
from data import load_tranformer
from sklearn.cluster import OPTICS
import time
from sklearn.model_selection import StratifiedShuffleSplit
import copy

def compute_metric(optics_result, Y):
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

def cluster(X):
    st = time.time()

    print("Start clustering")
    optics_model = OPTICS(eps=0.3, min_samples=3, metric="minkowski", n_jobs=16, cluster_method="dbscan")
    optics_result = optics_model.fit_predict(X)
    print("Take ", time.time()-st)

    return optics_result

def get_label(optics_result, Y):
    optics_clusters = np.unique(optics_result)

    labels = {}
    for optics_cluster in optics_clusters:
        index = np.where(optics_result == optics_cluster)[0]
        
        # get index which labeled or unlabeled
        label_index   = index[index<len(Y)]
        unlabel_index = index[index>=len(Y)]

        # get label
        unique, counts = np.unique(Y[label_index], return_counts=True)
        label = unique[np.argmax(counts)]

        for index in unlabel_index:
            labels[index] = label
    
    keys = list(labels)
    keys.sort()

    labels = [labels[k] for k in keys]
    return labels

def main(phase: str, problem: str):
    work_dir = Path(f"samples/{phase}/{problem}")
    category_transformer = load_tranformer(work_dir/"transformer", "category_transformer")

    # get reference
    refer_data = pandas.read_parquet(str(work_dir/"train"/"raw_train.parquet"), engine="pyarrow")
    refer_data = refer_data.sample(frac=1, random_state=42).head(10)
    
    # get current_data
    cur_data = pandas.concat(map(pandas.read_csv, (work_dir/"test"/"7_4_12_28").glob("*.csv"))).head(10)

    # full data
    X_ = pandas.concat([refer_data.drop(columns=['label']), cur_data])

    Y = refer_data['label']
    X = copy.deepcopy(X_)
    for col in category_transformer.columns:
        X[col] = X[col].astype("category")
        X[col] = X[col].cat.codes

    # clustering
    result = cluster(X.values)
    labels = get_label(result, Y.values)

    # new training data
    final_X = X_.values
    final_Y = np.concatenate([Y, labels])

    final = pandas.DataFrame(final_X, columns=X.columns)
    final['label'] = final_Y

    final.to_parquet(str(work_dir/"train"/"raw_train_2.parquet"))     
 
main("phase-2","prob-2")



 


    
