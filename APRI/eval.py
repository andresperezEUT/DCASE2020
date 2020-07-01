"""
eval.py

Given a target preset, the script computes and displays the evaluation metrics
for all csv files computed with this preset.
"""

from baseline import parameter
import os
from APRI.compute_metrics import compute_metrics


# %% PARAMS

preset = '4EVALUATION_Q!'
params = parameter.get_params(preset)
gt_folder = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations
this_file_path = os.path.dirname(os.path.abspath(__file__))
result_folder_path = os.path.join(this_file_path, params['results_dir'], preset)


# %% RUN

compute_metrics(gt_folder, result_folder_path, params)
