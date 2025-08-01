from dynamic_components.DycaResultBuilder import DycaResultBuilder 
from dynamic_components.ReportGenerator import ReportGenerator
from dynamic_components import utils
from sourcecode_modified.dyca_modified import dyca, reconstruction
import numpy as np
import logging
logger = logging.getLogger(__name__)

class Dyca:
    def __init__(self, subject):
        self.subject = subject
        
        self.signal = self.subject.processed_data.drop('Frame', axis=1).to_numpy()
        self.t = self.subject.processed_data['Frame'].to_numpy()
        self.signal_derivative = utils.get_signal_derivative(self.signal, self.t)
        
    def _validate_input(self, m, n):
        if self.signal is None:
            raise ValueError('No signal provided')
        if not isinstance(self.signal, np.ndarray):
            raise ValueError('Signal has to be a numpy array')
        if self.signal.ndim != 2:
            raise ValueError('Signal has to be a 2D array')
        if self.signal.shape[0] < self.signal.shape[1]:
            raise ValueError('Signal has to have more timepoints than channels')

        if not isinstance(self.t, np.ndarray):
            raise ValueError('Time signal has to be a numpy array')
        if self.t.ndim != 1:
            raise ValueError('Time signal has to be a 1D array')
        if self.t.shape[0] != self.signal.shape[0]:
            raise ValueError('Time signal has to have the same length as the signal')

        if not isinstance(self.signal_derivative, np.ndarray):
            raise ValueError('Derivative signal has to be a numpy array')
        if self.signal_derivative.shape != self.signal.shape:
            raise ValueError('Derivative signal has to have the same shape as the signal')

        if (n is not None):

            if not isinstance(n, int) or n < -1 or n > self.signal.shape[1] or n == 0:
                raise ValueError('n has to be an integer greater than 0, or -1 for no limit')

        if (m is not None):
            if not isinstance(m, int) or m < -1 or m > self.signal.shape[1] or m == 0:
                raise ValueError('m has to be an integer greater than 0, or -1 for no limit')

        if (m is not None and n is not None):
            # dyca conditions m >= n - m
            if m < n - m:
                raise ValueError('m has to be greater than or equal to n - m')
            if m > self.signal.shape[1]:
                raise ValueError('m has to be smaller than the number of channels')
            if n > self.signal.shape[1]:
                raise ValueError('n has to be smaller than the number of channels')
            if n < m:
                raise ValueError('n has to be greater than or equal to m')
            
    
    def _get_m(self):
        eigenvalues = self._do_dyca(m=2)['generalized_eigenvalues']
        m_dict = {}
        m_dict['percent_criterion'] = int(utils.get_95_criterion(eigenvalues))
        m_dict['steepest_slope']    = int(utils.get_steepest_slope(eigenvalues))
        
        logger.debug("Calculated m values: %s", m_dict)
        
        for cutoff_method, m in  m_dict.items():
            if m <= 1:
                logging.warning("Calculated m value %s at cutoff_method %s is not valid. Please consider setting it to 1", m, cutoff_method)
                
        return m_dict
    
    
    def _do_dyca(self, m, n=None):
        self._validate_input(m, n)
        dyca_output = dyca(
                signal=self.signal,
                m=m,
                n=n,
                time_index=self.t,
                derivative_signal=self.signal_derivative
            )
        
        #Vielleicht später hinzufügen:
        #self._validate_dyca_output(dyca_output)
        
        logger.debug("DyCA returned output keys=%s, with amplitudes shape=%s "
                    "and generalized_eigenvalues shape=%s",
                    list(dyca_output.keys()),
                    dyca_output['amplitudes'].shape,
                    dyca_output['generalized_eigenvalues'].shape
                    )
        
        return dyca_output
        
    
    def _create_dyca_output_report(self, dyca_output):
        if not isinstance(dyca_output, dict):
            raise ValueError('DyCA output must be a dictionary')
        report = ReportGenerator(dyca_output)
        return report
    
    def _get_n(self, m):
        singular_values = self._do_dyca(m)['singular_values']
        n = int(utils.get_steepest_slope(singular_values)) + m
        
        logger.debug("Calculated n value via steepest_slope: %s", )
        
        return n
        
    
    def run(self, m=None, n=None, custom_key='custom'):
        logger.info("Starting to run DyCA")
        print(n)
        builder = DycaResultBuilder()
        m_dict = self._get_m() if m is None else {custom_key: m}
        for cutoff_method, m in m_dict.items():
            dyca_output = self._do_dyca(m)
            report = self._create_dyca_output_report(dyca_output)
            builder.add_iteration(m, dyca_output, self.signal, cutoff_method=cutoff_method, report=report)
            
            n=self._get_n(m) if n is None else n
            dyca_output_svd = self._do_dyca(m, n)
            builder.add_iteration(m, dyca_output_svd, self.signal, cutoff_method=f"{cutoff_method}_svd", n=n, report=None, with_svd=True)

        result = builder.build()
        logger.info(f"Completed DyCA run with iterations {result.list_m_values()}")
        return result       
    
    
    
        
    