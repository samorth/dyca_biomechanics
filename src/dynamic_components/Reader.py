import pandas as pd
from dynamic_components import utils
import logging
from dynamic_components.config import settings

logger = logging.getLogger(__name__)

class Reader:
    def __init__(self, file_path, columns_to_drop=None):
        self.file_path = file_path
        self.columns_to_drop = columns_to_drop or settings.columns_to_drop
        self._data = None

    def fit(self, data=None):
        cols = utils.get_raw_data_columns(self.file_path)
        df = pd.read_csv(self.file_path, sep=',', names=cols, skiprows=5)
        if df.empty:
            logger.warning(f"No data found in {self.file_path}")
            raise ValueError(f"No data found in {self.file_path}")
        logger.info(f"Data loaded successfully from {self.file_path}")
        self._data = df.loc[:, ~df.columns.isin(self.columns_to_drop)]
        logger.info(f"Dropped columns: {self.columns_to_drop}")
        return self

    def transform(self, data=None):
        if self._data is None:
            raise RuntimeError("DataReader muss zuerst mit .fit() aufgerufen werden.")
        return self._data.copy()
    