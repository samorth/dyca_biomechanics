class Reconstructor:
    def __init__(self):
        pass
    
    @classmethod
    def select(cls, dataset_id, dyca_iteration_result, selected_amplitudes=None):
        if selected_amplitudes is None:
            selected_amplitudes = list(range(dyca_iteration_result.m))
        elif [selected_amplitudes][-1] > dyca_iteration_result.m:
            raise ValueError(
                f"Selected amplitudes {selected_amplitudes} exceed the number of modes {dyca_iteration_result.m}."
            )
        return ReconstructionResult(
            dataset_id=dataset_id,
            cutoff_method=dyca_iteration_result.cutoff_method,
            m=dyca_iteration_result.m,
            dyca_iteration_result=dyca_iteration_result,
            selected_amplitudes = selected_amplitudes
        )

class ReconstructionResult:
    def __init__(self, 
                dataset_id,
                cutoff_method,
                m,
                selected_amplitudes,
                dyca_iteration_result, 
                ):
        
        self.dataset_id = dataset_id
        self.cutoff_method = cutoff_method
        self.m = m
        self.selected_amplitudes = selected_amplitudes
        dyca_iteration_result = dyca_iteration_result
        self.Q_det, self.amplitude = dyca_iteration_result.do_reconstruction(self.selected_amplitudes)
        
        