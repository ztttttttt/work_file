from sklearn.base import BaseEstimator, TransformerMixin


class Preprocess(BaseEstimator, TransformerMixin):

    def __init__(self, MX=1, XY=1, JXL=1):
        self.MX = MX
        self.XY = XY
        self.JXL = JXL


