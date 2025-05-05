import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import biopsykit as bp
from sklearn.linear_model._cd_fast import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

#Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import SelectFromModel


# Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso

# Cross-Validation
from sklearn.model_selection import GroupKFold

from biopsykit.classification.model_selection import SklearnPipelinePermuter


# Set the working directory to the script's directory to ensure expected behavior of the relative paths
job_id = sys.argv[1] if len(sys.argv) > 1 else "0"
file_name = Path(__file__).with_suffix("").name

data_path = Path("../../results/data")
models_path = Path("../../results/models")

# Train Regression model for B-Point detection

input_data_b_point = pd.read_csv(data_path.joinpath("b-point/rr-interval/train_data_b_point_rr_interval.csv"), index_col=[0,1,2,3,4,5])

X_b_point, y_b_point, groups_b_point, group_keys_b_point = bp.classification.utils.prepare_df_sklearn(data=input_data_b_point, label_col="b_point_sample_reference", subject_col="participant", print_summary=False)

model_dict_b_point = {
    "scaler": {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()},
    "reduce_dim": {
        "SelectFromModel": SelectFromModel(estimator=RandomForestRegressor(n_estimators=100, random_state=0)),
        "SelectKBest": SelectKBest(),
    },
    "clf": {
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR(),
        "KNeighborsRegressor": KNeighborsRegressor(),
    },
}

params_dict_b_point = {
    "StandardScaler": None,
    "MinMaxScaler": None,
    "SelectFromModel": {
        "estimator": [
            RandomForestRegressor(n_estimators=100, random_state=0),
        ],
        "threshold": ["mean", "median", "0.5*mean", "0.25*mean", "0.75*mean", "1.25*mean"],
    },
    "SelectKBest": {
        "score_func": [f_regression, mutual_info_regression],
        "k": [2, 4, 6, 8, 10, "all"],
    },
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "splitter": ["best", "random"],
        "max_depth": [4, 8, 16, 32, None],
        "min_samples_leaf": [2, *list(np.arange(10, 100, 10))],
        "min_samples_split": [2, *list(np.arange(10, 100, 10))],
        "max_features": [*list(np.arange(0.1, 1.0, 0.2)), "log2", "sqrt", None],
    },
    "RandomForestRegressor": {
        "bootstrap": [True],  # Disabling bootstrap often doesn't help much in regression
        "criterion": ["squared_error", "friedman_mse"],
        "max_depth": [*list(np.arange(10, 100, 10)), None],  # Reducing unnecessary fine granularity
        "max_features": ["sqrt", "log2", 0.2, 0.5, 0.8],  # Balanced choices
        "min_samples_leaf": [1, 5, 10, 20, 50],  # More practical ranges
        "min_samples_split": [2, 5, 10, 20, 50],  # Logarithmic scaling
        "min_weight_fraction_leaf": [0.0, 0.01, 0.05],  # Only necessary values
        "max_leaf_nodes": [None, 10, 50, 100, 150, 200],  # Avoiding overly fine search
        "min_impurity_decrease": [0.0, 0.001, 0.01, 0.05],  # Prioritizing smaller values
        "n_estimators": [50, 100, 150, 200, 250, 400],  # Avoiding excessive values
        "ccp_alpha": [0.0, 0.001, 0.01, 0.05, 0.1],  # Small regularization values
    },
    "SVR": [
        {
            "kernel": ["linear"],
            "C": np.logspace(-1, 2, 4),  # [0.1, 1, 10, 100]
            "epsilon": np.logspace(-2, 0, 3),  # [0.01, 0.1, 1]
        },
        {
            "kernel": ["rbf"],
            "C": np.logspace(-1, 2, 4),  # [0.1, 1, 10, 100]
            "gamma": np.logspace(-3, 0, 4),  # [0.001, 0.01, 0.1, 1] (avoid extreme values)
            "epsilon": np.logspace(-2, 0, 3),
        },
    ],
    "KNeighborsRegressor": {
        "n_neighbors": np.arange(2, 40, 2),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}

hyper_search_dict = {
    "DecisionTreeRegressor": {"search_method": "random", "n_iter": 4000},
    "RandomForestRegressor": {"search_method": "random", "n_iter": 4000},
}

input_file_path_b_point = models_path.joinpath(f"b-point/rr-interval/b_point_{file_name}_hpc_{job_id}.pkl")
if input_file_path_b_point.exists():
    print(f"Loading pre-fitted pipeline permuter from {input_file_path_b_point}")
    pipeline_permuter_b_point = SklearnPipelinePermuter.from_pickle(input_file_path_b_point)
else:
    pipeline_permuter_b_point = SklearnPipelinePermuter(model_dict_b_point, params_dict_b_point, hyper_search_dict, random_state=0)

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pipeline_permuter_b_point.fit_and_save_intermediate(X=X_b_point, y=y_b_point, file_path=models_path.joinpath(f"b-point/rr-interval/b_point_{file_name}_hpc_{job_id}_baseline_result_rr.pkl"), outer_cv=outer_cv, inner_cv=inner_cv, scoring="neg_mean_absolute_error", groups=groups_b_point)


pipeline_permuter_b_point.to_pickle(models_path.joinpath(f"b-point/rr-interval/b_point_{file_name}_hpc_{job_id}_baseline_result_rr.pkl"))
print("Generated pickle file!")