CONFIG = {
    "CatBoostClassifier":
        {   
            # "use_best_model": True,
            "iterations": 1000, 
            "grow_policy": "Depthwise",
            "eval_metric": "AUC",
            "task_type": "GPU",
            "devices": '0:1'
        },
    "grid": {
        "rsm": [1.0, 1.5],
        'learning_rate': [0.025, 0.015],
        'depth': [9, 11],
        'l2_leaf_reg': [3, 5, 7],
    }
}