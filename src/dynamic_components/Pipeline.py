from dynamic_components import Reader
import logging

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, steps):
        self.steps = steps
        
    
    def get_params(self):
        params = {}
        for name, transformer in self.steps:
            if hasattr(transformer, 'get_params'):
                params[name] = transformer.get_params()
            else:
                params[name] = {}
        return params
    
    
    def fit_transform(self, data=None, dataset_id=None):
        for _, transformer in self.steps:
            try:
                if hasattr(transformer, 'fit_transform'):
                    data = transformer.fit_transform(data)
                else:
                    transformer.fit(data)
                    data = transformer.transform(data)
            except Exception as e:
                logger.error(f"Error in step {transformer.__class__.__name__}: {e} for subject {dataset_id}")
                return None
            
        logger.info("Pipeline.fit_transform(): Finished; returning %s", 
            'None' if data is None else f"DataFrame{data.shape}")
        
        return data

    
    def fit(self, data=None):
        for _, transformer in self.steps:
                data = transformer.fit(data).transform(data)
        return self
    
            
    def transform(self, data,dataset_id=None):
        try:
            for _, transformer in self.steps:
                data = transformer.transform(data)
            return data
        except Exception as e:
            logger.error(f"Error in step {transformer.__class__.__name__}: {e} for subject {dataset_id}")
            return None
    
    
    def inverse_transform(self, data, upto=None, dataset_id=None):
        steps = self.steps
        
        try:
            if upto is None:
                to_apply = reversed(steps)
            else:
                if isinstance(upto, str):
                    names = [name for name, _ in steps]
                    if upto not in names:
                        raise ValueError(f"No such step name: {upto}")
                    idx = names.index(upto)
                elif isinstance(upto, int):
                    idx = upto
                    if idx < 0 or idx >= len(steps):
                        raise IndexError(f"Step index out of range: {upto}")
                else:
                    raise TypeError("`upto` must be a step name (str), an index (int), or None.")
                to_apply = reversed(steps[idx:])
                
            for _, transformer in to_apply:
                if hasattr(transformer, 'inverse_transform'):
                    data = transformer.inverse_transform(data)
        except Exception as e:
            logger.error(f"Error in step {transformer.__class__.__name__}: {e} for subject {dataset_id}")
            return None
        
        logger.info("Pipeline.inverse_transform(): Finished; returning %s", 
            'None' if data is None else f"DataFrame{data.shape}")
        
        return data