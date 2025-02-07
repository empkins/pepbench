import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import biopsykit as bp
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

#Feature Selection
from sklearn.feature_selection import SelectKBest, RFE, f_regression, mutual_info_regression

#Classification
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

# Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Cross-Validation
from sklearn.model_selection import GroupKFold

from biopsykit.classification.model_selection import SklearnPipelinePermuter

# Set the working directory to the script's directory to ensure expected behavior of the relative paths
job_id = sys.argv[1] if len(sys.argv) > 1 else "0"

data_path = Path("../../results/data")
models_path = Path("../../results/models")

# Train Regression model for B-Point detection

input_data_b_point = pd.read_csv(data_path.joinpath("train_data_rr_interval_no_outlier_correction.csv"), index_col=[0,1,2,3,4])

X_b_point, y_b_point, groups_b_point, group_keys_b_point = bp.classification.utils.prepare_df_sklearn(data=input_data_b_point, label_col="b_point_sample_reference", subject_col="participant", print_summary=False)

model_dict_b_point = {
    "scaler": {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()},
    "reduce_dim": {"SelectKBest": SelectKBest(), "RFE": RFE(SVR(kernel="linear", C=1))},
    "clf": {
        "RandomForestRegressor": RandomForestRegressor(),
        "LinearSVR": LinearSVR(),
        "SVR": SVR(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        #"AdaBoostRegressor": AdaBoostRegressor(),
    },
}

params_dict_b_point = {
    "StandardScaler": None,
    "MinMaxScaler": None,
    "SelectKBest": {
        "score_func": [f_regression, mutual_info_regression],
        "k": [2, 4, 6, 8, 10, "all"],
    },
    "RFE": {
        "n_features_to_select": list(np.arange(2, 10, 2)),
        "step": [1, 2, 3],
    },
    "RandomForestRegressor": {
        "bootstrap": [True, False],
        "criterion": ["squared_error", "friedman_mse"],
        "max_depth": [*list(np.arange(4, 50, 2)), None],
        "max_features": [*list(np.arange(0.1, 1.0, 0.1)), "sqrt"],
        "min_samples_leaf": np.arange(0.1, 0.5, 0.1),
        "min_samples_split": np.arange(0.1, 0.8, 0.1),
        "min_weight_fraction_leaf": np.arange(0.0, 0.5, 0.1),
        "max_leaf_nodes": np.arange(2, 20, 2),
        "min_impurity_decrease": np.arange(0, 0.1, 0.01),
        "n_estimators": np.arange(10, 500, 10),
        "ccp_alpha": np.arange(0, 1, 0.1),
    },
    "LinearSVR": {
        "C": np.logspace(-2, 4, 7),  # 0.01 - 10000
        "epsilon": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "tol": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    },
    "SVR": [
        {
            "kernel": ["rbf"],
            "C": np.logspace(-2, 4, 7),  # 0.01 - 10000
            "gamma": np.logspace(-4, 3, 8),  # 0.0001 - 1000
        },
        {
            "kernel": ["poly"],
            "C": np.logspace(-2, 4, 7),  # 0.01 - 10000
            "degree": np.arange(2, 6),  # 2 - 6
        },
    ],
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "splitter": ["best", "random"],
        "max_depth": np.arange(1, 20, 2),
        "min_samples_leaf": np.arange(0.1, 0.5, 0.1),
        "min_samples_split": np.arange(0.1, 0.8, 0.1),
        "max_features": [*list(np.arange(0.1, 0.6, 0.1)), "log2", None],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [8, 9, 10, 11, 12, 13, 14],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    #"AdaBoostRegressor": {
    #    "estimator": [DecisionTreeRegressor(max_depth=1, criterion="friedman_mse"), SVR(kernel="linear", C=1)],
    #    "n_estimators": np.arange(10, 500, 20),
    #    "learning_rate": list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1.6, 0.1)),
    #},
}

hyper_search_dict = {"RandomForestRegressor": {"search_method": "random", "n_iter": 1000}}

input_file_path_b_point = models_path.joinpath("b_point_regression.pkl")
if input_file_path_b_point.exists():
    print(f"Loading pre-fitted pipeline permuter from {input_file_path_b_point}")
    pipeline_permuter_b_point = SklearnPipelinePermuter.from_pickle(input_file_path_b_point)
else:
    pipeline_permuter_b_point = SklearnPipelinePermuter(model_dict_b_point, params_dict_b_point, hyper_search_dict, random_state=0)

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pipeline_permuter_b_point.fit_and_save_intermediate(X=X_b_point, y=y_b_point, file_path=models_path.joinpath(f"b_point_regression.pkl"), outer_cv=outer_cv, inner_cv=inner_cv, scoring="neg_mean_absolute_error", groups=groups_b_point)


pipeline_permuter_b_point.to_pickle(models_path.joinpath("b_point_regression.pkl"))