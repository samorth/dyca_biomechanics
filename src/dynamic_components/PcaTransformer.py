import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from typing import Union

import logging
logger = logging.getLogger(__name__)

class PcaTransformer:
    def __init__(self, n_components=None, **pca_kwargs):
        self.n_components = n_components
        self.pca_kwargs = pca_kwargs
        self._pca_model : PCA = None
        self._column_list : list[str] = None 
        
    def fit(self, data):
        if self._pca_model is not None:
            raise RuntimeError("PCA ist already fitted. Please create another Instance to fit again.")
        
        logger.debug("PCA.fit(): Starting fit with n_components=%s, data shape=%s",
            self.n_components, data.shape)
        
        if not isinstance(data, (pd.DataFrame)):
            raise ValueError("Data should be a DataFrame")
        if 'Frame' not in data.columns:
            raise ValueError("DataFrame must contain a 'Frame' column")
        
        self._column_list = data.columns.drop('Frame').tolist()
        
        X = data[self._column_list]
        
        max_comp = X.shape[1]
        if self.n_components is not None:
            if not (1 <= self.n_components <= max_comp):
                raise ValueError(f"`n_components` must be between 1 and {max_comp}, got {self.n_components}.")

        self._pca_model= PCA(n_components=self.n_components, **self.pca_kwargs)
        self._pca_model.fit(X)
        
        logger.info("PCA.fit(): Model fitted with %d components; explained variance ratios: %s",
            self._pca_model.n_components_,
            np.round(self._pca_model.explained_variance_ratio_, 4).tolist())
        
        return self
    
    
    def transform(self, data):
        logger.debug("PCA.transform(): Called with data shape %s; model fitted? %s",
                    data.shape,
                    self._pca_model is not None)
        
        if self._pca_model is None:
            raise ValueError("Transformer has not been fitted. Call `fit()` first.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a pandas DataFrame.")
        if 'Frame' not in data.columns:
            raise ValueError("DataFrame must contain a 'Frame' column.")
        
        frame_vals = data['Frame'].values
        X = data[self._column_list]
        
        X_pca = self._pca_model.transform(X)
        
        n_pc = X_pca.shape[1]
        pc_names = [f"PC{i+1}" for i in range(n_pc)]
        df_pca = pd.DataFrame(X_pca, columns=pc_names, index=data.index)
        df_pca.insert(0, 'Frame', frame_vals)
        
        logger.info("PCA.transform(): Transformed to %d principal components", n_pc)
        
        return df_pca
    
        
    def inverse_transform(self, data_transformed: Union[pd.DataFrame, np.ndarray]):
        logger.debug("PCA.inverse_transform(): Called with data_transformed shape %s", data_transformed.shape)
        
        if self._pca_model is None:
            raise ValueError("Transformer has not been fitted. Call `fit()` first.")
        
        meta, index = None, None
        if isinstance(data_transformed, pd.DataFrame):
            if 'Frame' not in data_transformed.columns:
                raise ValueError("Transformed DataFrame muss eine 'Frame'-Spalte enthalten.")
            meta = data_transformed['Frame']
            pc_data = data_transformed.drop(columns='Frame')
            index = data_transformed.index
            X_pca = pc_data.values
        else:
            X_pca = data_transformed
            
        if not self._check_compatibility(X_pca):
            raise ValueError("Inkompatible Dimensionen für Inverse-Transform.")
        
        X_orig = self._pca_model.inverse_transform(X_pca)
        
        if meta is not None:
            df = pd.DataFrame(X_orig, columns=self._column_list, index=index)
            df.insert(0, 'Frame', meta.values)
            logger.info("PCA.inverse_transform(): zurück in %d Original-Features", X_orig.shape[1])
            return df
        else:
            logger.info("PCA.inverse_transform(): zurück in %d Original-Features", X_orig.shape[1])
            return X_orig
        
            
    def _check_compatibility(self, data):
        return True      

        
    