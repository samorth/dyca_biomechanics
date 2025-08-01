import pandas as pd

class MeanCenterer():
    def __init__(self):
        self.means_ = {}

    def fit(self, data):
        for col in data.select_dtypes('number').columns:
            if col == 'Frame':
                continue
            self.means_[col] = data[col].mean()
        return self

    def transform(self, data):
        if not self.means_:
            raise RuntimeError("Fit must be called before transform.")
        data = data.copy()
        for col, mean in self.means_.items():
            if col in data.columns:
                data[col] -= mean
            else:
                raise ValueError(f"Column {col} not found in data for transformation.")
        return data

    def inverse_transform(self, data):
        if not self.means_:
            raise RuntimeError("Fit must be called before inverse_transform.")
        data = data.copy()
        for col, mean in self.means_.items():
            if col in data.columns:
                data[col] += mean
            else:
                raise ValueError(f"Column {col} not found in data for inverse transformation.")
        return data