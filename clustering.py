import pandas
import numpy as np
from pathlib import Path
from sklearn.cluster import OPTICS
import time
import ast
from data.utils import handle_duplicate
from sklearnex import patch_sklearn
patch_sklearn()

def transform_cat(X, columns):
    for col in columns:
        X[col] = X[col].astype("category")
        X[col] = X[col].cat.codes
        X[col] = X[col].fillna(-1)

    return X

def cluster(X):
    st = time.time()

    print("Start clustering")
    optics_model = OPTICS(eps=0.3, min_samples=3, metric="minkowski", n_jobs=24, cluster_method="dbscan")
    optics_result = optics_model.fit_predict(X)
    print("Take ", time.time()-st)

    return optics_result

def get_threshold_label(optics_result, Y):
    '''
        Only use for labeled data
    '''
    REPORT = {}

    # compute apperance rate of each label
    optics_clusters = np.unique(optics_result)
    for optics_cluster in optics_clusters:
        index = np.where(optics_result == optics_cluster)

        # count value for each label
        unique, counts = np.unique(Y[index], return_counts=True)
        counts = counts/sum(counts)

        # get best label and appearance rate
        for label, count in zip(unique, counts):
            if label not in REPORT:
                REPORT[label] = []
            
            if count == max(counts):
                REPORT[label].append(count)

    # get min threshold
    for key in REPORT.keys():
        if len(REPORT[key]) == 0:
            REPORT[key] = 0.0
        else:
            REPORT[key] = min(max(REPORT[key]), 0.6)
    
    return REPORT

def label_for_unseen(optics_result, Y, REPORT):
    '''
        Label for unseen label from labeld dataset
    '''
    optics_clusters = np.unique(optics_result)

    unseen_label_indexs = []
    unseen_labels = []
    for optics_cluster in optics_clusters:
        indexs = np.where(optics_result == optics_cluster)[0]
        
        # get label based on labeled data
        labeled_indexs = indexs[indexs<len(Y)]
        if len(labeled_indexs) == 0:
            continue

        ## compute apperance rate of each label
        unique, counts = np.unique(Y[labeled_indexs], return_counts=True)
        counts = counts/sum(counts)

        ## get label and apperance rate
        label = unique[np.argmax(counts)]
        rate  = max(counts)

        if rate >= REPORT[label]:
            unlabel_index = indexs[indexs>=len(Y)]

            # label for unseen
            unseen_labels.extend([label for _ in unlabel_index])
            unseen_label_indexs.extend(unlabel_index)

    return unseen_label_indexs, unseen_labels
    
def main(phase: str, problem: str):
    work_dir = Path(f"samples/{phase}/{problem}")
    feature_config = ast.literal_eval(
        (work_dir/"train"/"features_config.json").read_text()
    )

    # load labeled data
    labeled_data = pandas.read_parquet(str(work_dir/"train"/"raw_train.parquet"), engine="pyarrow")
    labeled_data = labeled_data.sample(frac=1, random_state=42)
    labeled_data = handle_duplicate(labeled_data)

    Y = labeled_data['label']
    # ============================= TRAIN LABELED DATA ===========================================    
    X = labeled_data.drop(columns=['label'])
    X = transform_cat(X, feature_config['category_columns']) 
    
    result = cluster(X.values)
    report = get_threshold_label(result, Y.values)
    print(f"{phase}-{problem}: ", report)

    # ============================= TRAIN MIX LABLED and UNSEEN DATA ===========================================
    unseen_data = pandas.concat(map(pandas.read_csv, (work_dir/"test"/"7_4_12_28").glob("*.csv")))
    unseen_data.drop_duplicates(inplace=True)

    unseen_data.reset_index(inplace=True)
    unseen_data.drop(columns=["index"], inplace=True)

    X = pandas.concat([labeled_data.drop(columns=['label']), unseen_data])
    X = transform_cat(X, feature_config['category_columns'])

    # clustering
    result = cluster(X.values)
    indexs, labels = label_for_unseen(result, Y.values, report)

    # setup labeled 
    labeled_unseen_data_index    = [i-len(Y) for i in indexs]
    labeled_unseen_data          = unseen_data.loc[labeled_unseen_data_index, :]
    labeled_unseen_data["label"] = labels

    final = pandas.concat([labeled_data, labeled_unseen_data])
    final.reset_index(inplace=True)
    final.drop(columns=["index"], inplace=True)

    print("Before drop duplicate: ", final.shape)
    final = handle_duplicate(final)
    print("After drop duplicate: ", final.shape)

    final.to_parquet(str(work_dir/"train"/"raw_train_2.parquet"), index=None)     
 
main("phase-2","prob-2")
main("phase-2","prob-1")




 


    
