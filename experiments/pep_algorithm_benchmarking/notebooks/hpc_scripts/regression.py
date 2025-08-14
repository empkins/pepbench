import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import biopsykit as bp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.linear_model._cd_fast import ConvergenceWarning

# Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Cross-Validation
from sklearn.model_selection import GroupKFold

from biopsykit.classification.model_selection import SklearnPipelinePermuter
import sklearn


# Set the working directory to the script's directory to ensure expected behavior of the relative paths
job_id = sys.argv[1] if len(sys.argv) > 1 else "0"
file_name = Path(__file__).with_suffix("").name

data_path = Path("../../results/data")
models_path = Path("../../results/models")
rater = "rater_01"

print(sklearn.__version__)
# Train Regression model for B-Point detection

input_data_b_point = pd.read_csv(data_path.joinpath(f"b-point/rr-interval/{rater}/train_data_b_point_rr_interval_include_nan.csv"), index_col=[0,1,2,3,4,5]).astype(float)

X_b_point, y_b_point, groups_b_point, group_keys_b_point = bp.classification.utils.prepare_df_sklearn(data=input_data_b_point, label_col="b_point_sample_reference", subject_col="participant", print_summary=False)
print(f"Input data dtype: {X_b_point.dtype}")
print(f"y_b_point isnan: {np.isnan(y_b_point).any()}")

model_dict_b_point = {
    "scaler": {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()},
    "clf": {
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
    },
}

params_dict_b_point = {
    "StandardScaler": None,
    "MinMaxScaler": None,
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "friedman_mse"],
        "splitter": ["best"],
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
}

hyper_search_dict = {
    "DecisionTreeRegressor": {"search_method": "random", "n_iter": 4000},
    "RandomForestRegressor": {"search_method": "random", "n_iter": 4000},
}

input_file_path_b_point = models_path.joinpath(f"b-point/rr-interval/{rater}/b_point_{file_name}_hpc_{job_id}.pkl")
if input_file_path_b_point.exists():
    print(f"Loading pre-fitted pipeline permuter from {input_file_path_b_point}")
    pipeline_permuter_b_point = SklearnPipelinePermuter.from_pickle(input_file_path_b_point)
else:
    pipeline_permuter_b_point = SklearnPipelinePermuter(model_dict_b_point, params_dict_b_point, hyper_search_dict, random_state=0)

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pipeline_permuter_b_point.fit_and_save_intermediate(X=X_b_point, y=y_b_point, file_path=models_path.joinpath(f"b-point/rr-interval/{rater}/b_point_{file_name}_hpc_{job_id}_baseline_result_rr_include_nan_{rater}.pkl"), outer_cv=outer_cv, inner_cv=inner_cv, scoring="neg_mean_absolute_error", groups=groups_b_point)


pipeline_permuter_b_point.to_pickle(models_path.joinpath(f"b-point/rr-interval/{rater}/b_point_{file_name}_hpc_{job_id}_baseline_result_rr_include_nan_{rater}.pkl"))
print("Generated pickle file!")