from app.core.keywords import extract_nouns, top_keywords


def _confirmed(client, text: str, date: str, tags: list[str]) -> int:
    did = client.post("/api/diaries/text", json={"date": date, "content": text}).json()["diary_id"]
    client.patch(f"/api/diaries/{did}", json={"status": "confirmed", "emotion_tags": tags})
    return did


def test_extract_nouns_keeps_nouns_only():
    nouns = extract_nouns("오늘 발표 준비를 열심히 했다")
    assert "발표" in nouns
    assert "준비" in nouns
    assert "했다" not in nouns


def test_stopwords_are_not_keywords():
    assert "오늘" not in extract_nouns("오늘 발표를 했다")
    assert "하루" not in top_keywords(["하루 종일 발표 준비", "하루가 길었다 발표"], top_k=5)


def test_top_keywords_prefers_repeated_terms_across_diaries():
    texts = ["발표 준비가 힘들었다", "발표가 끝나서 홀가분하다", "친구와 저녁을 먹었다"]
    kws = top_keywords(texts, top_k=2)
    assert kws and kws[0] == "발표"


def test_analytics_averages_tags_per_diary(client):
    _confirmed(client, "발표를 끝냈다.", "2026-10-05", ["기쁨", "불안"])  # 각 0.5
    _confirmed(client, "친구와 저녁을 먹었다.", "2026-10-06", ["기쁨"])  # 기쁨 1.0
    body = client.get("/api/diaries/analytics", params={"start_date": "2026-10-01", "end_date": "2026-10-31"}).json()
    assert body["diary_count"] == 2
    assert body["average"]["기쁨"] == 0.75
    assert body["average"]["불안"] == 0.25
    assert body["average"]["슬픔"] == 0.0
    assert round(sum(body["average"].values()), 4) == 1.0
    assert isinstance(body["top_keywords"], list)


def test_analytics_respects_period_and_handles_empty(client):
    _confirmed(client, "지난달 일기다.", "2026-09-01", ["슬픔"])
    body = client.get("/api/diaries/analytics", params={"start_date": "2026-10-01", "end_date": "2026-10-31"}).json()
    assert body["diary_count"] == 0
    assert all(v == 0.0 for v in body["average"].values())
    assert body["top_keywords"] == []


def test_analytics_ignores_drafts(client):
    client.post("/api/diaries/text", json={"date": "2026-10-03", "content": "아직 draft다."})
    body = client.get("/api/diaries/analytics").json()
    assert body["diary_count"] == 0


def test_similar_finds_other_diary_with_same_sentence_and_excludes_self(client):
    a = _confirmed(client, "오늘 발표를 무사히 끝냈다.", "2026-10-05", ["기쁨"])
    b = _confirmed(client, "오늘 발표를 무사히 끝냈다. 기분이 좋았다.", "2026-10-06", ["기쁨"])
    _confirmed(client, "친구와 저녁을 먹었다.", "2026-10-07", ["기쁨"])

    body = client.get(f"/api/diaries/{a}/similar").json()
    ids = [r["diary_id"] for r in body["results"]]
    assert b in ids
    assert a not in ids
    top = body["results"][0]
    assert top["diary_id"] == b
    assert top["score"] > 0.9
    assert top["matched_text"] == "오늘 발표를 무사히 끝냈다."


def test_similar_requires_confirmed_diary(client):
    draft = client.post("/api/diaries/text", json={"date": "2026-10-05", "content": "확정 전이다."}).json()["diary_id"]
    r = client.get(f"/api/diaries/{draft}/similar")
    assert r.status_code == 422


def test_similar_empty_when_nothing_close(client):
    a = _confirmed(client, "오늘 발표를 무사히 끝냈다.", "2026-10-05", ["기쁨"])
    body = client.get(f"/api/diaries/{a}/similar").json()
    assert body["results"] == []
    assert body["message"] == "비슷한 기록을 찾지 못했습니다."
