import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(autouse=True)
def _stub_emotion_model(monkeypatch):
    # 테스트는 모델 가중치 없이 돌도록 스텁 분류기를 쓴다. 실모델 테스트는 test_emotion_model.py에서 따로 고른다.
    monkeypatch.setenv("EMOTION_MODEL", "stub")
    # 임베딩도 해시 임베더로 돌려 테스트에서 모델을 내려받지 않는다.
    monkeypatch.setenv("EMBEDDER", "hashing")


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "test.db")
    with TestClient(app) as c:
        yield c
