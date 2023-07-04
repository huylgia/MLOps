import pandas, ast
from typing import Tuple, Mapping, Any
from sklearn.model_selection import train_test_split
from data import StatTest
from pathlib import Path
from utils import measure_execute_time
from model import load_weight
import numpy as np
if __name__ == "__main__":
    work_dir = Path("samples/phase-2/prob-1/model/CatBoostClassifier")

    model = load_weight(work_dir, "model")

    importance = model.feature_importances_
    importance = importance[importance>=1]

    mean = np.mean(importance)
    for i, v in enumerate(model.feature_importances_):
        if v > mean:
            print(i, v)
