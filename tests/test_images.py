from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import config
from app.core.date_extractor import extract_date
from app.core.ocr_client import OcrError, parse_text
from app.main import create_app

JPEG = b"\xff\xd8\xff" + b"0" * 200  # 시그니처만 맞춘 가짜 jpg


class FakeOcr:
    """실제 Clova를 호출하지 않는 테스트용 OCR 클라이언트."""

    def __init__(self, text: str = "2026년 11월 4일\n오늘 발표를 무사히 끝냈다.", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls = 0

    def recognize(self, image: bytes) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.text


@pytest.fixture
def setup(tmp_path):
    ocr = FakeOcr()
    app = create_app(tmp_path / "test.db", upload_dir=tmp_path / "uploads", ocr_client=ocr)
    with TestClient(app) as c:
        yield c, ocr, tmp_path / "uploads"


def _upload(client, data: bytes = JPEG, name: str = "diary.jpg"):
    return client.post("/api/diaries/image", files={"file": (name, data, "image/jpeg")})


def test_upload_creates_draft_with_extracted_date_and_image(setup):
    client, ocr, uploads = setup
    r = _upload(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "draft"
    assert body["date"] == "2026-11-04"
    assert "오늘 발표를 무사히 끝냈다." in body["content"]
    assert body["image"] == f"/api/diaries/{body['diary_id']}/image"
    assert body["is_indexed"] is False
    assert (uploads / f"{body['diary_id']}.jpg").read_bytes() == JPEG

    img = client.get(body["image"])
    assert img.status_code == 200
    assert img.headers["content-type"].startswith("image/jpeg")
    assert img.content == JPEG


def test_upload_without_year_leaves_date_empty(setup):
    client, ocr, _ = setup
    ocr.text = "11월 4일\n그냥 메모"
    body = _upload(client).json()
    assert body["date"] is None


def test_rejects_non_jpg_name(setup):
    client, ocr, _ = setup
    r = _upload(client, name="diary.png")
    assert r.status_code == 422
    assert ocr.calls == 0


def test_rejects_wrong_signature_even_with_jpg_name(setup):
    client, ocr, _ = setup
    r = _upload(client, data=b"PNG-not-jpeg" * 10)
    assert r.status_code == 422
    assert ocr.calls == 0


def test_rejects_over_5mb(setup):
    client, ocr, _ = setup
    too_big = b"\xff\xd8\xff" + b"0" * config.OCR_MAX_BYTES
    r = _upload(client, data=too_big)
    assert r.status_code == 422
    assert ocr.calls == 0


def test_returns_503_when_ocr_not_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CLOVA_OCR_URL", "")
    monkeypatch.setattr(config, "CLOVA_OCR_SECRET", "")
    app = create_app(tmp_path / "test.db", upload_dir=tmp_path / "uploads")
    with TestClient(app) as client:
        assert _upload(client).status_code == 503


def test_daily_quota_returns_429(setup, monkeypatch):
    client, ocr, _ = setup
    monkeypatch.setattr(config, "OCR_DAILY_LIMIT", 2)
    assert _upload(client).status_code == 200
    assert _upload(client).status_code == 200
    r = _upload(client)
    assert r.status_code == 429
    assert ocr.calls == 2  # 한도를 넘으면 OCR을 호출하지 않는다


def test_ocr_error_returns_502_and_creates_nothing(setup):
    client, ocr, _ = setup
    ocr.error = OcrError("OCR 서비스 호출에 실패했습니다.")
    r = _upload(client)
    assert r.status_code == 502
    assert client.get("/api/diaries").json() == []


def test_empty_ocr_text_returns_422(setup):
    client, ocr, _ = setup
    ocr.text = "   "
    r = _upload(client)
    assert r.status_code == 422
    assert client.get("/api/diaries").json() == []


def test_text_diary_has_no_image(setup):
    client, _, _ = setup
    did = client.post("/api/diaries/text", json={"date": "2026-11-04", "content": "글만 쓴 일기"}).json()["diary_id"]
    assert client.get(f"/api/diaries/{did}/image").status_code == 404
    assert client.get(f"/api/diaries/9999/image").status_code == 404


def test_delete_removes_image_file(setup):
    client, _, uploads = setup
    did = _upload(client).json()["diary_id"]
    assert (uploads / f"{did}.jpg").exists()
    assert client.delete(f"/api/diaries/{did}").status_code == 204
    assert not (uploads / f"{did}.jpg").exists()
    assert client.get(f"/api/diaries/{did}/image").status_code == 404


def test_recall_result_carries_image_url(setup):
    client, _, _ = setup
    body = _upload(client).json()
    client.patch(f"/api/diaries/{body['diary_id']}", json={"status": "confirmed"})
    results = client.get("/api/diaries/recall", params={"query": "오늘 발표를 무사히 끝냈다", "top_k": 3}).json()["results"]
    hit = [r for r in results if r["diary_id"] == body["diary_id"]]
    assert hit and hit[0]["image"] == body["image"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026년 11월 4일 날씨 맑음", date(2026, 11, 4)),
        ("2026.11.04", date(2026, 11, 4)),
        ("2026-1-5 메모", date(2026, 1, 5)),
        ("11월 4일", None),  # 연도 없음
        ("2026년 2월 30일", None),  # 존재하지 않는 날짜
        ("날짜 없는 본문", None),
    ],
)
def test_extract_date(text, expected):
    assert extract_date(text) == expected


def test_root_serves_demo_page_with_notices(setup):
    client, _, _ = setup
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Re:mind" in r.text
    assert "109" in r.text  # 위기 안내 문구
    assert "실제 일기를 올리지" in r.text


def test_parse_text_joins_lines_and_rejects_failure():
    data = {"images": [{"inferResult": "SUCCESS", "fields": [
        {"inferText": "오늘", "lineBreak": False},
        {"inferText": "발표", "lineBreak": True},
        {"inferText": "끝", "lineBreak": True},
    ]}]}
    assert parse_text(data) == "오늘발표\n끝"
    with pytest.raises(OcrError):
        parse_text({"images": [{"inferResult": "ERROR"}]})
    with pytest.raises(OcrError):
        parse_text({})
