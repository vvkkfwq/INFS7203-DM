from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
