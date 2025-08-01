from typing import Any
import numpy as np
import pandas as pd
from sourcecode_modified.dyca_modified import reconstruction

class DycaIterationResult:
    def __init__(self, m, dyca_output, signal, cutoff_method=None, report=None, n=None, with_svd=False):
        self.m = m
        self.dyca_output = dyca_output
        self.signal = signal
        self.cutoff_method = cutoff_method
        self.report = report
        self.n = n
        with_svd = with_svd
        
    def do_reconstruction(self, subject, selected_amplitudes):
        if self.n is not None:   
            amplitudes = np.atleast_2d(self.dyca_output['amplitudes_svd'][selected_amplitudes])
            print("Chose to reconstruct using SVD amplitudes.")
        else:
            amplitudes = np.atleast_2d(self.dyca_output['amplitudes'][selected_amplitudes])
        #Anmerkung: Fehler im Github - signal muss shape = (channels, time) haben, nicht (time, channels)!
        dyca_reconstruction = reconstruction(self.signal.transpose(), amplitudes)
        #Hier eventuell checken ob Realteil != 0
        recon = dyca_reconstruction['reconstruction'].transpose().real
        rec_df = pd.DataFrame(data=recon, columns = subject.pipeline.steps[-1][1]._column_list)
        rec_df.insert(0, 'Frame', subject.processed_data['Frame'])
        Q_det = subject.undo_processing(upto='mean_center', dyca_reconstruction=rec_df)
        modes = dyca_reconstruction.get('modes', None)
        
        result = {
            'Q_det': Q_det,
            'modes': modes,
            'amplitudes': amplitudes,
            'cost' : dyca_reconstruction['cost'],
            'selected_amplitudes': selected_amplitudes,
            'subject': subject,
            'iteration': self
        }
        return result