import pytest


@pytest.fixture(autouse=True)
def _stub_emotion_model(monkeypatch):
    # 테스트는 모델 가중치 없이 돌도록 스텁 분류기를 쓴다. 실모델 테스트는 test_emotion_model.py에서 따로 고른다.
    monkeypatch.setenv("EMOTION_MODEL", "stub")
