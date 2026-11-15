import json
import os
import subprocess
import sys
from pathlib import Path

from app import config

ROOT = Path(__file__).resolve().parent.parent


def _demo_file(tmp_path: Path) -> Path:
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"diaries": [
        {"id": 1, "date": "2026-10-02", "content": "시험 결과가 나빠서 하루 종일 자책했다."},
        {"id": 2, "date": "2026-10-03", "content": "친구와 만나 저녁을 먹었다."},
    ]}, ensure_ascii=False), encoding="utf-8")
    return path


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "EMBEDDER": "hashing", "EMOTION_MODEL": "stub", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reset_demo.py"), *args],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )


def test_reset_requires_yes(tmp_path):
    db = tmp_path / "app.db"
    db.write_text("keep")
    r = _run(["--db", str(db), "--uploads", str(tmp_path / "up"), "--demo", str(_demo_file(tmp_path))])
    assert r.returncode == 1
    assert db.read_text() == "keep"  # 아무것도 지우지 않는다


def test_reset_wipes_and_seeds_confirmed_demo(tmp_path):
    import sqlite3

    db = tmp_path / "app.db"
    db.write_text("old junk")  # 기존 파일은 지우고 새 DB로 바꾼다
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "1.jpg").write_bytes(b"x")

    r = _run(["--yes", "--db", str(db), "--uploads", str(uploads), "--demo", str(_demo_file(tmp_path))])
    assert r.returncode == 0, r.stderr
    assert not (uploads / "1.jpg").exists()

    conn = sqlite3.connect(db)
    statuses = [row[0] for row in conn.execute("SELECT status FROM diaries ORDER BY id")]
    tags = [row[0] for row in conn.execute("SELECT label FROM diary_tags WHERE source='confirmed'")]
    indexed = conn.execute("SELECT COUNT(*) FROM sentence_embeddings").fetchone()[0]
    conn.close()
    assert statuses == ["confirmed", "confirmed"]
    assert tags  # 추천 태그가 확정 태그로 복사되었는지
    assert indexed >= 2  # 확정 시 문장 벡터가 만들어졌는지


def test_demo_file_is_in_repo():
    assert config.DEMO_DATA_PATH.exists()
