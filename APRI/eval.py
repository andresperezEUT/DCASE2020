"""
TODO COMPLETE

evaluate precomputed output files
"""

from baseline import parameter
import os
from APRI.compute_metrics import compute_metrics

# %% PARAMS

params = parameter.get_params()
gt_folder = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations





# %% RUN

compute_metrics(gt_folder, result_folder_path)