CONFIG = {
    "CatBoostClassifier":
        {   
            # "use_best_model": True,
            "iterations": 1000, 
            "grow_policy": "Depthwise",
            "eval_metric": "AUC",
            "task_type": "GPU",
            "devices": '0:3'
        },
    "grid": {
        "rsm": [1.0],
        'learning_rate': [0.025, 0.015],
        'depth': [11],
        'l2_leaf_reg': [3, 7],
    }
}