import os

import numpy as np
import pytest

from app.core.embedder import HashingEmbedder, load_embedder


def test_hashing_embedder_is_normalized():
    vecs = HashingEmbedder().encode(["오늘 발표를 했다", ""])
    assert vecs.shape == (2, HashingEmbedder.dimension)
    assert vecs.dtype == np.float32
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0)


def test_load_embedder_selects_hashing_and_rejects_unknown():
    assert isinstance(load_embedder("hashing"), HashingEmbedder)
    with pytest.raises(ValueError):
        load_embedder("nope")


@pytest.mark.skipif(os.environ.get("RUN_REAL_EMBEDDER") != "1", reason="ko-sroberta 모델 다운로드가 필요함 (RUN_REAL_EMBEDDER=1)")
def test_ko_sroberta_embedder_shape_and_similarity():
    emb = load_embedder("sroberta")
    vecs = emb.encode(["멘탈이 터졌다", "하루 종일 멍하니 천장만 바라봤다", "친구와 저녁을 먹었다"])
    assert vecs.shape == (3, emb.dimension)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    query, close, far = vecs
    assert float(query @ close) > float(query @ far)
