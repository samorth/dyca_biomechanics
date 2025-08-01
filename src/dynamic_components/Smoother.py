from scipy.signal import butter, filtfilt
import pandas as pd

class Smoother:
    def __init__(self, fs=100.0, cutoff=6.0, order=4):
        self.fs = fs
        self.cutoff = cutoff
        self.order = order

    def fit(self, data):
        return self

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        result = data.copy()
        for col in result.select_dtypes('number').columns:
            if col == 'Frame':
                continue
            result[col] = filtfilt(b, a, result[col].values)
        return result

