import pickle
from pathlib import Path
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

class ReconstructionResult:
    def __init__(self, 
                dataset_id, 
                reconstructions,
                m, 
                criterion, 
                save_path="Object hasnt been saved yet",
                modes=None,
                selected_amplitudes=None):
                
        self.dataset_id = dataset_id
        self.reconstructions = reconstructions
        self.modes = modes or {k: [] for k in reconstructions.keys()}
        self.selected_amplitudes = selected_amplitudes or {k: [] for k in reconstructions.keys()}
        self.m = m
        self.criterion = criterion
        self.save_path = save_path
        
    
    def select(self, kinds=None, indices=None):
        """
        Gibt ein neues ReconstructionResult-Objekt zurück, das nur die ausgewählten
        Arten (‘position’, ‘velocity’, ‘combined’) und Indizes enthält.
        
        kinds: List[str] mit {'position','velocity','combined'} oder None für alle
        indices: Dict[str, list[int]] mit keys als kinds und values als Liste von Indizes
        """
        kinds = kinds or self.reconstructions.keys()
        sel_recs = {}
        sel_modes = {}
        sel_amps = {}

        for k in kinds:
            recs = self.reconstructions[k]
            modes = self.modes[k]
            amps = self.selected_amplitudes[k]
            idxs = indices.get(k, range(len(recs))) if indices else range(len(recs))

            sel_recs[k] = [recs[i] for i in idxs]
            sel_modes[k] = [modes[i] for i in idxs]
            sel_amps[k] = [amps[i] for i in idxs]
            
        return ReconstructionResult(
            dataset_id=self.dataset_id,
            reconstructions=sel_recs,
            m=self.m,
            criterion=self.criterion,
            modes=sel_modes,
            selected_amplitudes=sel_amps,
            save_path=self.save_path
        )
        
    def save(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.save_path = str(save_path)
        logger.info(f"Reconstruction result for dataset {self.dataset_id} saved to {self.save_path}")
        
    
    @classmethod
    def load(cls, load_path):
        with open(load_path, 'rb') as f:
            result = pickle.load(f)
        return result

    def __repr__(self):
        return (
            f"<Subject={self.dataset_id}, ReconstructionResult m={self.m}, criterion={self.criterion!r}, "
            f"saved_to={self.save_path!r}>"
        )
        
    
class DycaReconstructor:
    def __init__(self, subject, iteration):
        self.subject = subject
        self.iteration = iteration
        self.criterion = iteration.cutoff_method
        self.m = iteration.m
        
        logger.debug(f"Initialized Reconstructor: m={self.m}, criterion={self.criterion!r}")
        
        
    def _do_reconstruction(self, selected_amplitudes):
        result = self.iteration.do_reconstruction(selected_amplitudes)
        rec = result['reconstruction'].transpose().real
        rec_df = pd.DataFrame(data=rec, columns = self.subject.pipeline.steps[-1][1]._column_list)
        rec_df.insert(0, 'Frame', self.subject.processed_data['Frame'])
        Q_det = self.subject.undo_processing(upto='mean_center', dyca_reconstruction=rec_df)
        modes = result.get('modes', None)
        return Q_det, modes
    
    def run(self):
        amps_all = np.atleast_2d(self.iteration.dyca_output['amplitudes'])
        reconstructions = {"position": [], "velocity": [], "combined": []}
        modes_dict      = {"position": [], "velocity": [], "combined": []}
        amps_dict       = {"position": [], "velocity": [], "combined": []}
        
        for i in range(self.m):
            selected_amplitudes = [i]
            Q_det, modes = self._do_reconstruction(selected_amplitudes)
            amp_vals = np.atleast_2d(amps_all[selected_amplitudes])
            reconstructions["position"].append(Q_det)
            modes_dict["position"].append(modes)
            amps_dict["position"].append(amp_vals)
        
        for i in range(self.m, self.m * 2):
            selected_amplitudes = [i]
            Q_det, modes = self._do_reconstruction(selected_amplitudes)
            amp_vals = np.atleast_2d(amps_all[selected_amplitudes])
            reconstructions["velocity"].append(Q_det)
            modes_dict["position"].append(modes)
            amps_dict["position"].append(amp_vals)
            
        for i in range(self.m):
            selected_amplitudes = [i, i + self.m]
            Q_det = self._do_reconstruction(selected_amplitudes)
            amp_vals = np.atleast_2d(amps_all[selected_amplitudes])
            reconstructions["combined"].append(Q_det)
            modes_dict["position"].append(modes)
            amps_dict["position"].append(amp_vals)
            
        logger.info(
            "Completed reconstructions: position=%d, velocity=%d, combined=%d",
            len(reconstructions["position"]),
            len(reconstructions["velocity"]),
            len(reconstructions["combined"])
        )
        
        result = ReconstructionResult(
            dataset_id=self.subject.dataset_id,
            reconstructions=reconstructions,
            modes=modes_dict,
            selected_amplitudes=amps_dict,
            m=self.m,
            criterion=self.criterion
            )
        
        return result