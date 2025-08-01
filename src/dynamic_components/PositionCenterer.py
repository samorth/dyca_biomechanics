import pandas as pd

class PositionCenterer():
    def __init__(self, marker_cols=None):
        self.marker_cols = marker_cols or [
            'LASI_X','LASI_Y','LASI_Z','RASI_X','RASI_Y','RASI_Z',
            'LPSI_X','LPSI_Y','LPSI_Z','RPSI_X','RPSI_Y','RPSI_Z'
        ]
        self.offset_ = None

    def fit(self, data):
        hip = data[self.marker_cols]
        self.offset_ = pd.DataFrame({
            'x': hip[['LASI_X','RASI_X','LPSI_X','RPSI_X']].mean(axis=1),
            'y': hip[['LASI_Y','RASI_Y','LPSI_Y','RPSI_Y']].mean(axis=1),
            'z': hip[['LASI_Z','RASI_Z','LPSI_Z','RPSI_Z']].mean(axis=1),
        }, index=data.index)
        return self

    def transform(self, data):
        off = self.offset_
        df = data.copy()
        for col in df:
            if col.endswith('_X'): df[col] -= off['x']
            elif col.endswith('_Y'): df[col] -= off['y']
            elif col.endswith('_Z'): df[col] -= off['z']
        return df

    def inverse_transform(self, data):
        df = data.copy()
        off = self.offset_
        for col in df:
            if col.endswith('_X'): df[col] += off['x']
            elif col.endswith('_Y'): df[col] += off['y']
            elif col.endswith('_Z'): df[col] += off['z']
        return df