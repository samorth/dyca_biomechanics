#%%
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from dynamic_components.Dyca import Dyca 
from dynamic_components.DycaResult import DycaResult
import importlib
from dynamic_components import utils
importlib.reload(utils)
from dynamic_components.Interpolator import Interpolator
from dynamic_components.MeanCenterer import MeanCenterer
from dynamic_components.PcaTransformer import PcaTransformer
from dynamic_components.Pipeline import Pipeline
from dynamic_components.PositionCenterer import PositionCenterer
from dynamic_components.Reader import Reader
from dynamic_components.Smoother import Smoother
from dynamic_components.Subject import Subject
from dynamic_components.DycaReconstructor import DycaReconstructor
from dynamic_components.config import setup_logging, current_subject

BASE_DIR        = Path(__file__).resolve().parents[1]
DATA_DIR        = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR   = BASE_DIR / 'data' / 'processed'
FILENAMES       = utils.get_filenames(DATA_DIR)

print('percent criterion, steepest slope criterion')
for filename in FILENAMES:
    if "T18" in filename or "T23" in filename or "T28" in filename:
        continue
    load_path = PROCESSED_DIR / 'preprocessed_data' /(Path(filename).stem + '.pkl')
    dyca_result = DycaResult.load(PROCESSED_DIR / 'dyca_result_objects' / (Path(filename).stem + '.pkl'))
    print(f"{dyca_result.get_iteration_by_key('percent_criterion').m}, {dyca_result.get_iteration_by_key('steepest_slope').m + 1}")

id = 'Sub8_Kinematics_T3'
dyca_result = DycaResult.load(PROCESSED_DIR / 'dyca_result_objects' / (id + '.pkl'))    
subject = Subject.load_subject(PROCESSED_DIR / 'preprocessed_data' / (id + '.pkl'))

iteration = dyca_result.get_iteration_by_key('percent_criterion_svd')
singular_values = iteration.dyca_output['singular_values']
eigenvalues = iteration.dyca_output['generalized_eigenvalues']

highlights = {'steepest slope': 1}
utils.plot_bars(singular_values, title='', xlabel='Index', ylabel=r'$\sigma_i$', annotate=False, highlights=highlights)
highlights = {'95% criterion': 8, 'steepest slope': 13}
utils.plot_bars(eigenvalues[:25], title='', xlabel='Index', ylabel=r'$\lambda_i$', xtick_step=5, annotate=False, highlights=highlights, special_ytick=0.95)
# %%
