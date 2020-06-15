"""
TODO COMPLETE

evaluate precomputed output files
"""

from baseline import parameter
import os
from APRI.compute_metrics import compute_metrics
from APRI.utils import plot_results


# %% PARAMS

preset = '4EVALUATION'
params = parameter.get_params(preset)
gt_folder = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations
this_file_path = os.path.dirname(os.path.abspath(__file__))
result_folder_path = os.path.join(this_file_path, params['results_dir'], preset)


# %% RUN

compute_metrics(gt_folder, result_folder_path, params)
