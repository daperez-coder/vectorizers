import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer


class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(
        self: BaseEstimator, b: float = 0.75, k1: float = 1.6, delta: float = 0.0
    ) -> None:
        self.b = b
        self.k1 = k1
        self.delta = delta
        self.tfidf_transformer = TfidfTransformer(norm=None, smooth_idf=False)

    def fit(
        self: BaseEstimator,
        X: csr_matrix | npt.NDArray[np.float64] | pd.DataFrame,
        y: npt.NDArray[np.float64] | None = None,
    ) -> BaseEstimator:
        self.tfidf_transformer.fit(X)
        self.avdl_ = X.sum(1).mean()
        self.idf_ = self.tfidf_transformer.idf_ - 1
        return self

    def transform(
        self: BaseEstimator,
        X: csr_matrix | npt.NDArray[np.float64] | pd.DataFrame,
        y: npt.NDArray[np.float64] | None = None,
    ) -> csr_matrix:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = csr_matrix(X)
        doc_len = X.sum(axis=1).A1
        b, k1, avdl, delta = self.b, self.k1, self.avdl_, self.delta

        # IDF-adjusted matrix
        idf = self.idf_
        X_idf = X.multiply(idf)

        # Original BM25 components
        numerator = X_idf.multiply(k1 + 1)
        one_matrix = csr_matrix(
            (np.ones_like(X.data), X.indices, X.indptr), shape=X.shape
        )
        denominator = X + one_matrix.multiply(
            (k1 * (1 - b + b * doc_len / avdl)).reshape(-1, 1)
        )

        # Delta component (sparse matrix with delta*idf values)
        delta_data = delta * idf[X.indices]  # Sparse delta calculation
        delta_matrix = csr_matrix((delta_data, X.indices, X.indptr), shape=X.shape)

        # Combine components using sparse operations
        bm25 = numerator.multiply(denominator.power(-1)) + delta_matrix

        return bm25.tocsr()

    def fit_transform(
        self: BaseEstimator,
        X: csr_matrix | npt.NDArray[np.float64] | pd.DataFrame,
        y: npt.NDArray[np.float64] | None = None,
    ) -> csr_matrix | npt.NDArray[np.float64] | pd.DataFrame:
        return self.fit(X).transform(X)

    def get_feature_names_out(self: BaseEstimator) -> npt.NDArray[np.str_]:
        return self.tfidf_transformer.get_feature_names_out()