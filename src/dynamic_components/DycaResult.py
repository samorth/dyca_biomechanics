import os
import pickle as pkl
from pathlib import Path

class DycaResult:
    def __init__(self, iterations):
        self._iterations = iterations
        self.last_save_path = None
        self.dataset_id = 'UNKNOWN'  
        
    
    def list_keys(self):
        keys = self._iterations.keys()
        key_strs = (str(k) for k in keys)
        return ", ".join(key_strs)
    
    
    def get_iteration_by_key(self, key):
        if key in self._iterations:
            return self._iterations[key]
        else:
            raise KeyError(f"Key '{key}' not found in DycaResult iterations.")
        
        
    def get_iteration_by_index(self, index: int):
        """Gibt die i-te Iteration zurück (0-basiert)."""
        keys = list(self._iterations.keys())
        try:
            key = keys[index]
            return self._iterations[key]
        except IndexError:
            raise IndexError(f"Iteration index {index} out of range. There only exist {len(keys)} iterations.")
        
        
    def list_m_values(self):    
        return [iteration.m for iteration in self._iterations.values()]
    
    
    def get_last_save_path(self):
        return self.last_save_path
    

    def save(self, save_path):
        save_path_str = str(save_path)
        orig_path = Path(save_path_str + '_dyca_result')
        if orig_path.suffix.lower() != '.pkl':
            orig_path = orig_path.with_suffix('.pkl')
        directory = orig_path.parent
        if directory and not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        with orig_path.open('wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
        self.last_save_path = str(orig_path.resolve())
        
        
    @classmethod
    def load(cls, path: str):
        import os
        import pickle as pkl
        
        with open(path, 'rb') as f:
            obj = pkl.load(f)

        if isinstance(obj, cls):
            obj.last_save_path = os.path.abspath(path)
            return obj
        else:
            raise TypeError(f"The file at {path} does not contain a DycaResult instance.")
