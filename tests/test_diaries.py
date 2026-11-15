import pytest
from fastapi.testclient import TestClient

from app import config
from app.core import emotion
from app.core.sentence_splitter import split_sentences
from app.main import create_app




def test_split_sentences_uses_kiwi():
    sentences = split_sentences("오늘 발표를 했다. 너무 피곤하다. 그래도 끝냈다.")
    assert len(sentences) == 3


def test_suggest_tags_rules():
    # 신뢰도 낮은 문장 제외, 중립 제외, 많이 나온 순 최대 3개
    labels = [("불안", 0.9), ("불안", 0.8), ("기쁨", 0.9), ("슬픔", 0.9), ("분노", 0.9), ("기쁨", 0.1)]
    assert emotion.suggest_tags(labels) == ["불안", "기쁨", "슬픔"]
    # 남은 감정이 없으면 중립
    assert emotion.suggest_tags([("중립", 1.0)]) == [config.NEUTRAL_LABEL]


def test_text_draft_is_listed_and_not_indexed(client):
    r = client.post("/api/diaries/text", json={"date": "2026-11-04", "content": "오늘 프로젝트 때문에 밤을 새웠다."})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "draft"
    assert body["is_indexed"] is False
    assert body["suggested_tags"] == [config.NEUTRAL_LABEL]

    listed = client.get("/api/diaries", params={"status": "draft"}).json()
    assert [d["diary_id"] for d in listed] == [body["diary_id"]]


def test_text_requires_date(client):
    r = client.post("/api/diaries/text", json={"content": "날짜 없음"})
    assert r.status_code == 422


def test_confirm_indexes_and_recall_finds_it(client):
    diary = client.post("/api/diaries/text", json={"date": "2026-11-04", "content": "오늘 발표를 무사히 끝냈다."}).json()
    r = client.patch(
        f"/api/diaries/{diary['diary_id']}",
        json={"status": "confirmed", "emotion_tags": ["기쁨"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "confirmed"
    assert body["is_indexed"] is True
    assert body["emotion_tags"] == ["기쁨"]

    recall = client.get("/api/diaries/recall", params={"query": "오늘 발표를 무사히 끝냈다", "top_k": 3})
    assert recall.status_code == 200
    results = recall.json()["results"]
    assert results and results[0]["diary_id"] == diary["diary_id"]


def test_confirmed_edit_stays_confirmed_and_reindexes(client):
    diary = client.post("/api/diaries/text", json={"date": "2026-11-04", "content": "처음 본문이다."}).json()
    did = diary["diary_id"]
    client.patch(f"/api/diaries/{did}", json={"status": "confirmed"})

    r = client.patch(f"/api/diaries/{did}", json={"content": "교정된 새 본문이다."})
    body = r.json()
    assert body["status"] == "confirmed"
    assert body["is_indexed"] is True
    assert [s["text"] for s in body["sentences"]] == ["교정된 새 본문이다."]


def test_delete_removes_diary_and_index(client):
    did = client.post("/api/diaries/text", json={"date": "2026-11-04", "content": "지울 일기다."}).json()["diary_id"]
    client.patch(f"/api/diaries/{did}", json={"status": "confirmed"})
    assert client.delete(f"/api/diaries/{did}").status_code == 204
    assert client.get(f"/api/diaries/{did}").status_code == 404
    assert client.get("/api/diaries/recall", params={"query": "지울 일기"}).json()["results"] == []


def test_confirm_without_tags_uses_suggested(client):
    did = client.post("/api/diaries/text", json={"date": "2026-11-04", "content": "확정만 한다."}).json()["diary_id"]
    body = client.patch(f"/api/diaries/{did}", json={"status": "confirmed"}).json()
    assert body["suggested_tags"] == [config.NEUTRAL_LABEL]
    assert body["emotion_tags"] == body["suggested_tags"]


def test_similar_on_missing_diary_is_404(client):
    assert client.get("/api/diaries/999/similar").status_code == 404
