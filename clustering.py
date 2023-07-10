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
    optics_model = OPTICS(eps=0.3, min_samples=3, metric="minkowski", n_jobs=16, cluster_method="dbscan")
    optics_result = optics_model.fit_predict(X)
    print("Take ", time.time()-st)

    return optics_result

def label_for_unseen(optics_result, Y, threshold=0.9):
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

        if rate >= threshold:
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
    print("Load labeled data")
    labeled_data = pandas.read_parquet(str(work_dir/"train"/"raw_train.parquet"), engine="pyarrow")
    labeled_data = labeled_data.sample(frac=1, random_state=42)
    labeled_data = handle_duplicate(labeled_data)

    # reset index
    labeled_data.reset_index(inplace=True)
    labeled_data.drop(columns=["index"], inplace=True)

    # load unseen data
    print("Load unseen data")
    unseen_data = pandas.concat(map(pandas.read_csv, (work_dir/"test"/"7_4_12_28").glob("*.csv")))
    unseen_data['label'] = np.nan

    # ============== clustered_data ==============
    clustered_data = pandas.concat([labeled_data, unseen_data])
    clustered_data.reset_index(inplace=True)
    clustered_data.drop(columns=["index"], inplace=True)

    # drop duplicates
    clustered_data.drop_duplicates(inplace=True)
    clustered_data.reset_index(inplace=True)
    clustered_data.drop(columns=["index"], inplace=True)

    # ============== Y ==============
    Y = clustered_data['label'].dropna()
    X = clustered_data.drop(columns=['label'])
    print(X.shape)
    print(Y.shape)

    # transform x
    X_tranform = transform_cat(X, feature_config['category_columns'])

    # clustering
    result = cluster(X_tranform.values)
    indexs, labels = label_for_unseen(result, Y.values, threshold=0.9)

    print(len(labels))

    # setup labeled 
    clustered_data['label'] = np.nan
    clustered_data.loc[indexs, 'label'] = labels
    clustered_data.loc[list(Y.index), 'label'] = Y.values

    print(clustered_data.shape)
    # reset index
    clustered_data = clustered_data[~clustered_data['label'].isna()]
    print(clustered_data.shape)

    clustered_data.reset_index(inplace=True)
    clustered_data.drop(columns=["index"], inplace=True)

    clustered_data.to_parquet(str(work_dir/"train"/"raw_train_2.parquet"), index=None)     
 
main("phase-2","prob-2")
main("phase-2","prob-1")




 


    
