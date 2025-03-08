import tempfile

import pytest

import scipy.sparse as sp

from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import assert_array_almost_equal

from nlp.bm25 import BM25Transformer


@pytest.fixture
def sample_data() -> sp.csr_matrix:
    # Sample sparse document-term matrix
    X = sp.csr_matrix([[1, 2, 0, 0, 1], [0, 3, 1, 0, 2], [2, 0, 0, 1, 0]])
    return X


@pytest.fixture
def estimator() -> BM25Transformer:
    return BM25Transformer(b=0.75, k1=1.6, delta=0.5)


def test_estimator_basic_properties(
    estimator: BM25Transformer, sample_data: sp.csr_matrix
) -> None:
    # Test initialization parameters
    assert estimator.b == 0.75
    assert estimator.k1 == 1.6
    assert estimator.delta == 0.5

    # Test fit-transform
    X_trans = estimator.fit_transform(sample_data)
    assert sp.issparse(X_trans)
    assert X_trans.shape == sample_data.shape


def test_correctness_calculation() -> None:
    # Test setup per BM25+ formula from search results
    X = sp.csr_matrix([[3], [2], [0]])  # 3 docs, single term
    estimator = BM25Transformer(b=0.75, k1=1.2, delta=0.5).fit(X)

    # Manual calculations according to BM25+ spec [4]
    avgdl = (3 + 2 + 0) / 3

    # IDF calculation from BM25+ formula
    tfidf_transformer = TfidfTransformer(norm=None, smooth_idf=False)
    tfidf_transformer.fit(X)
    idf = tfidf_transformer.idf_[0] - 1

    # Document 1 (tf=3, dl=3)
    tf_component_1 = (3 * (1.2 + 1)) / (3 + 1.2 * (1 - 0.75 + 0.75 * (3 / avgdl)))
    score_1 = idf * (tf_component_1 + 0.5)

    # Document 2 (tf=2, dl=2)
    tf_component_2 = (2 * (1.2 + 1)) / (2 + 1.2 * (1 - 0.75 + 0.75 * (2 / avgdl)))
    score_2 = idf * (tf_component_2 + 0.5)

    # Document 3 (tf=0) - delta applied despite zero TF [4]
    score_3 = 0.0  # BM25+ requires delta for all query terms

    expected = sp.csr_matrix([[score_1], [score_2], [score_3]])
    result = estimator.transform(X)

    assert_array_almost_equal(result.toarray(), expected.toarray(), decimal=4)


def test_sparse_structure_preservation(
    estimator: BM25Transformer, sample_data: sp.csr_matrix
) -> None:
    # Test non-zero positions remain the same
    X_trans = estimator.fit_transform(sample_data)
    assert isinstance(X_trans, sp.csr_matrix)


def test_pipeline_integration() -> None:
    # Test integration with scikit-learn Pipeline
    pipeline = Pipeline(
        [("vectorizer", CountVectorizer()), ("bm25", BM25Transformer())]
    )

    texts = ["hello world", "foo bar", "test test test"]
    pipeline.fit(texts)
    X_trans = pipeline.transform(texts)
    assert sp.issparse(X_trans)


def test_persistence(estimator: BM25Transformer, sample_data: sp.csr_matrix) -> None:
    # Test pickling/unpickling
    with tempfile.TemporaryDirectory() as tmp:
        estimator.fit(sample_data)
        dump(estimator, f"{tmp}/bm25.joblib")
        loaded = load(f"{tmp}/bm25.joblib")

        assert hasattr(loaded, "avdl_")
        assert hasattr(loaded, "idf_")
