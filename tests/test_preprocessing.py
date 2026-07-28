import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import snapatac2 as snap
from snapatac2.preprocessing._harmony import _harmony


def mock_harmony(monkeypatch, corrected):
    module = SimpleNamespace(
        run_harmony=lambda *args, **kwargs: SimpleNamespace(Z_corr=corrected)
    )
    monkeypatch.setitem(sys.modules, "harmonypy", module)


def test_harmony_preserves_observation_major_output(monkeypatch):
    embedding = np.arange(6, dtype=float).reshape(3, 2)
    corrected = embedding + 0.5
    adata = ad.AnnData(
        X=np.zeros((3, 1)),
        obs={"batch": ["first", "second", "first"]},
    )
    adata.obsm["X_spectral"] = embedding
    mock_harmony(monkeypatch, corrected)

    snap.pp.harmony(adata, batch="batch", inplace=True)

    np.testing.assert_array_equal(adata.obsm["X_spectral_harmony"], corrected)
    assert adata.obsm["X_spectral_harmony"].shape == embedding.shape


def test_harmony_transposes_component_major_output(monkeypatch):
    embedding = np.arange(6, dtype=float).reshape(3, 2)
    corrected = embedding + 0.5
    batch = pd.DataFrame({"batch": ["first", "second", "first"]})
    mock_harmony(monkeypatch, corrected.T)

    result = snap.pp.harmony(embedding, batch=batch, inplace=False)

    np.testing.assert_array_equal(result, corrected)
    assert result.shape == embedding.shape


def test_harmony_rejects_unexpected_output_shape(monkeypatch):
    embedding = np.arange(6, dtype=float).reshape(3, 2)
    batch = pd.DataFrame({"batch": ["first", "second", "first"]})
    mock_harmony(monkeypatch, np.zeros((4, 2)))

    with pytest.raises(
        ValueError,
        match=r"Unexpected Harmony output shape: expected \(3, 2\), got \(4, 2\)",
    ):
        _harmony(embedding, batch)


@pytest.mark.parametrize(
    ("embedding", "batch"),
    [
        (np.array([[1.0, 2.0]]), pd.DataFrame({"batch": ["first"]})),
        (
            np.arange(6, dtype=float).reshape(3, 2),
            pd.DataFrame({"batch": ["first", "first", "first"]}),
        ),
    ],
)
def test_harmony_short_circuits(monkeypatch, embedding, batch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("run_harmony should not be called")

    monkeypatch.setitem(
        sys.modules,
        "harmonypy",
        SimpleNamespace(run_harmony=fail_if_called),
    )

    np.testing.assert_array_equal(_harmony(embedding, batch), embedding)
