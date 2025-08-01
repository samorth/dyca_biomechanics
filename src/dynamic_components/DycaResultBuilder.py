from dynamic_components.DycaResult import DycaResult
from dynamic_components.DycaIterationResult import DycaIterationResult

class DycaResultBuilder:
    def __init__(self):
        self._iters = {}
        
        
    def add_iteration(self, m, dyca_output, signal, cutoff_method='UNKNOWN', report=None, n=None, with_svd=False):
        iteration_result = DycaIterationResult(
            m=m,
            dyca_output=dyca_output,
            signal = signal,
            cutoff_method = cutoff_method,
            report=report,
            with_svd=with_svd,
            n=n 
        )
        key = f"{cutoff_method}_svd" if with_svd else cutoff_method
        self._iters[key] = iteration_result
        
        
    def build(self):
        return DycaResult(
            iterations=self._iters
        )