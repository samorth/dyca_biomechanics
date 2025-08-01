from dynamic_components import utils
import pandas as pd

class Interpolator():
    def __init__(self, method = 'spline', order=3):
        self.method = method
        self.order = order
    
    def fit(self, data):
        utils.check_for_faulty_channels(data)
        return self
    
    def transform(self, data):
        frames = data['Frame']
        df = data.drop(columns='Frame').copy()
        df = df.interpolate(method=self.method, order=self.order, axis=0)
        df = df.ffill().bfill()
        if df.isnull().values.any():
            df = df.fillna(df.mean())
        df.insert(0, 'Frame', frames)
        return df