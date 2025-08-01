import pickle
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

class Subject:
    def __init__(self, pipeline, dataset_id, origin = None):
        self.dataset_id = dataset_id
        if origin is not None:
            self.origin = origin
        self.pipeline = pipeline
        
        self.processed_data = self.pipeline.fit_transform(dataset_id=dataset_id)
        self.steps = [step for step, _ in self.pipeline.steps]
        
        self.last_save_path = None
        
        
    def undo_processing(self, upto=None, dyca_reconstruction=None):
        if dyca_reconstruction is not None:
            retransformed_signal = self.pipeline.inverse_transform(data=dyca_reconstruction, upto=upto, dataset_id=self.dataset_id)
        else:    
            retransformed_signal = self.pipeline.inverse_transform(data=self.processed_data, upto=upto, dataset_id=self.dataset_id)
        return retransformed_signal
        
    
    def save(self, save_path: str = None):
        if save_path is not None:
            orig_path = Path(save_path)
            path = orig_path
            
            while path.suffix:
                path = path.with_suffix('')
            path = path.with_suffix('.pkl')
            self.last_save_path = path
            
        elif self.last_save_path is not None:
            path = Path(self.last_save_path)
        else:
            raise ValueError("No save path provided and no last save path available.")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)  
        
        
    def get_last_save_path(self):
        return self.last_save_path
    
    @classmethod
    def load_subject(cls, load_path):
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"File {load_path} does not exist.")
        with open(path, 'rb') as f:
            subject = pickle.load(f)
        if not isinstance(subject, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}, got {type(subject).__name__} instead.")
        
        return subject
    
    
    
        
        

