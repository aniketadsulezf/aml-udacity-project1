# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The data is of direct marketing campaigns (phone calls) of a Portuguese banking institution. We are trying to predict if client will subscribe a term deposit (binary classification).

We have compared 2 models **Hyder-Drive assisted LogisticRegression** model which got accuracy of **91.39%** and **VotingEnsemble** model produced by AutoML which got accuracy of **92.07%**

## Scikit-learn Pipeline
Pipeline consiste of running train.py using Azure HyderDrive for hyper parameter tuning running from Jupyter Notebook.
train.py does data download, followed by data cleaning, splitting it in train and test data, model traininga and model validation.
Here Logistic Regression was used as ML model for classification.
**Hyder-Drive assisted LogisticRegression** model which got best accuracy of **91.39%**
Following are the best parameters for LogisticRegression: **C: 1.9456524612308663 & max_iter: 200**

I choose Bayesian sampling which intelligently selects next sample of hyperparameter based on model performance on previous hyperparameter, which leads to improvement with each model training.

I choose BanditPolicy which automatically terminated poorly performing run to improve computational efficiency. Based on slack factor and evaluation interval it terminates runs where the primary metric is not within the specified slack factor compared to the best performing run.

## AutoML
Best model produced by AutoML was **VotingEnsemble** with accuracy of **92.07%**
VotingEnsemble consisted of 7 voting classifers with specific weights. Predictions of the 7 voting classifiers are combined based weights.
Following are classifiers with respective weights:
ensembled_algorithms': "['XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'LogisticRegression']"
'ensemble_weights': '[0.2857142857142857, 0.07142857142857142, 0.14285714285714285, 0.21428571428571427, 0.14285714285714285, 0.07142857142857142, 0.07142857142857142]'

details of each classifier is as follows:
1. {
    "class_name": "LogisticRegression",
    "module": "sklearn.linear_model",
    "param_args": [],
    "param_kwargs": {
        "C": 51.79474679231202,
        "class_weight": null,
        "multi_class": "ovr",
        "penalty": "l2",
        "solver": "lbfgs"
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}  
2. {
    "class_name": "XGBoostClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "booster": "gbtree",
        "colsample_bytree": 0.6,
        "eta": 0.3,
        "gamma": 0,
        "max_depth": 6,
        "max_leaves": 0,
        "n_estimators": 10,
        "objective": "reg:logistic",
        "reg_alpha": 0.3125,
        "reg_lambda": 2.3958333333333335,
        "subsample": 1,
        "tree_method": "auto"
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
3. {
    "class_name": "XGBoostClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "booster": "gbtree",
        "colsample_bytree": 0.5,
        "eta": 0.5,
        "gamma": 0,
        "max_depth": 6,
        "max_leaves": 3,
        "n_estimators": 10,
        "objective": "reg:logistic",
        "reg_alpha": 0.7291666666666667,
        "reg_lambda": 2.3958333333333335,
        "subsample": 0.8,
        "tree_method": "auto"
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
4. {
    "class_name": "LightGBMClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "boosting_type": "gbdt",
        "colsample_bytree": 0.6933333333333332,
        "learning_rate": 0.09473736842105263,
        "max_bin": 110,
        "max_depth": 8,
        "min_child_weight": 6,
        "min_data_in_leaf": 0.003457931034482759,
        "min_split_gain": 1,
        "n_estimators": 25,
        "num_leaves": 227,
        "reg_alpha": 0.9473684210526315,
        "reg_lambda": 0.42105263157894735,
        "subsample": 0.49526315789473685
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
5. {
    "class_name": "XGBoostClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "booster": "gbtree",
        "colsample_bytree": 0.7,
        "eta": 0.01,
        "gamma": 0.01,
        "max_depth": 7,
        "max_leaves": 31,
        "n_estimators": 10,
        "objective": "reg:logistic",
        "reg_alpha": 2.1875,
        "reg_lambda": 1.0416666666666667,
        "subsample": 1,
        "tree_method": "auto"
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
6. {
    "spec_class": "sklearn",
    "class_name": "XGBoostClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "tree_method": "auto"
    },
    "prepared_kwargs": {}
}
7. {
    "spec_class": "sklearn",
    "class_name": "LightGBMClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "min_data_in_leaf": 20
    },
    "prepared_kwargs": {}
}

## Pipeline comparison
We have compared 2 models **Hyder-Drive assisted LogisticRegression** model which got accuracy of **91.39%** and **VotingEnsemble** model produced by AutoML which got accuracy of **92.07%**. Accuracy of model produced by AutoML is slightly better than model produced by HyperDrive.
For HyperDrive you need custom model with defined hyperparameter and their ranges. While for AutoML very few parameters are required to set in configuration and it find both best model and hyperparameters bu comparing all models. This features make life of ML Engg easy.

## Future work
1. Experiment with different hyperparameters and sampling methods like gird sampling or random sampling on the Scikit-learn LogicRegression model or other custom-coded machine learning models
2. Better data preprocessing - handling class imbalance, utilize model interpretability to get insights about features which will help in better feature engineering

## Proof of cluster clean up
I have used delete() method to delete the compute cluster at the end of Jupyter Notebook.
