CONFIG = {
    "CatBoostClassifier":
        {   
            # "use_best_model": True,
            "iterations": 1000, 
            "grow_policy": "Depthwise",
            "eval_metric": "AUC",
            "task_type": "GPU",
            "devices": '3'
        },
    "grid": {
        "rsm": [1.0],
        'learning_rate': [0.75, 0.25, 0.075],
        'depth': [9, 11, 13],
        'l2_leaf_reg': [3, 5, 7],
    }
}