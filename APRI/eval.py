"""
TODO COMPLETE

evaluate precomputed output files
"""

from baseline import parameter
import os
from APRI.compute_metrics import compute_metrics
from APRI.utils import plot_results


# %% PARAMS

plot = True

preset = 'particle'
params = parameter.get_params(preset)
gt_folder = os.path.join(params['dataset_dir'], 'metadata_dev')  # path to annotations
this_file_path = os.path.dirname(os.path.abspath(__file__))
result_folder_path = os.path.join(this_file_path, params['results_dir'], preset)


# %% RUN

compute_metrics(gt_folder, result_folder_path, params)


# %% PLOT

# Achtung! will plot *all* metadata result files
res_files = [f for f in os.listdir(result_folder_path) if f != '.DS_Store']
res_files = ['fold1_room1_mix007_ov1.csv',
             'fold2_room1_mix007_ov1.csv',
             'fold3_room1_mix007_ov1.csv',
             'fold4_room1_mix007_ov1.csv',
             'fold5_room1_mix007_ov1.csv',
             'fold6_room1_mix007_ov1.csv',
             'fold2_room1_mix008_ov1.csv',
             'fold2_room1_mix009_ov1.csv',
             'fold2_room1_mix010_ov1.csv',
             'fold2_room1_mix047_ov2.csv',
             'fold2_room1_mix048_ov2.csv',
             'fold2_room1_mix049_ov2.csv'
             ]

if plot:
    for res_file in res_files:
        plot_results(os.path.join(result_folder_path,res_file), params)
