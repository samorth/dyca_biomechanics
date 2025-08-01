import logging
import numpy as np
import pandas as pd
from pathlib import Path

from dynamic_components.Dyca import Dyca 
from dynamic_components.DycaResult import DycaResult
from dynamic_components import utils
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

logger = logging.getLogger(__name__)

def process_subjects():
    for filename in FILENAMES:   
        dataset_id = Path(filename).stem
        current_subject['id'] = dataset_id
        
        logger.info(f"Start processing {filename}")
        
        preproc = Pipeline([('reader', Reader(DATA_DIR / filename)),
                            ('interp', Interpolator()),
                            ('smooth', Smoother()),
                            ('pos_center', PositionCenterer()),
                            ('mean_center', MeanCenterer()),
                            ('pca', PcaTransformer(n_components = None))
                            ])
        
        subject = Subject(origin = filename, pipeline=preproc, dataset_id=dataset_id)
        subject.save(PROCESSED_DIR / 'preprocessed_data' /filename)
        logger.info("Finished subject processing")
        

def generate_dyca_results():
    save_path = Path(PROCESSED_DIR / 'dyca_result_objects')
    for filename in FILENAMES:
        dataset_id = Path(filename).stem
        current_subject['id'] = dataset_id
        
        logger.info(f"Start Dyca processing for {filename}")
        
        load_path = PROCESSED_DIR / 'preprocessed_data' /(Path(filename).stem + '.pkl')
        subject = Subject.load_subject(load_path)
        dyca = Dyca(subject)
        dyca_result = dyca.run()
        dyca_result.dataset_id = dataset_id
        dyca_result.save(save_path / filename)
        
    logger.info("Finished Dyca processing")
    
    
def reconstruct_dyca_signal(dataset_id):
    save_path = Path(PROCESSED_DIR / 'reconstruction_objects')
    
    for id in dataset_id:
        current_subject['id'] = id
        
        logger.info(f"Start reconstruction for {id}")
        
        dyca_result = DycaResult.load(PROCESSED_DIR / 'dyca_result_objects' / (id + '.pkl'))    
        subject = Subject.load_subject(PROCESSED_DIR / 'preprocessed_data' / (id + '.pkl'))
        
        reconstructor = DycaReconstructor(subject = subject,
                                        iteration=dyca_result.get_iteration_by_key('percent_criterion')
                                        )
        
        result = reconstructor.run()
        result.save(save_path / (id + '_reconstructions' +'.pkl'))
        
        
BASE_DIR        = Path(__file__).resolve().parents[1]
DATA_DIR        = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR   = BASE_DIR / 'data' / 'processed'
FILENAMES       = utils.get_filenames(DATA_DIR)
    
if __name__ == "__main__":
    setup_logging()
    #process_subjects()
    generate_dyca_results()
    #reconstruct_dyca_signal(dataset_id=["Sub8_Kinematics_T3"])
    
