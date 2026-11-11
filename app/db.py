import sqlite3
from pathlib import Path

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS diaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT,                       -- YYYY-MM-DD. draft에서는 NULL 허용, 확정 시 필수
    content     TEXT NOT NULL,
    image_path  TEXT,
    status      TEXT NOT NULL CHECK (status IN ('draft', 'confirmed')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diary_tags (
    diary_id    INTEGER NOT NULL REFERENCES diaries(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('suggested', 'confirmed')),
    PRIMARY KEY (diary_id, label, source)
);

CREATE TABLE IF NOT EXISTS diary_sentences (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    diary_id        INTEGER NOT NULL REFERENCES diaries(id) ON DELETE CASCADE,
    sentence_index  INTEGER NOT NULL,
    text            TEXT NOT NULL,
    emotion_label   TEXT NOT NULL,
    confidence      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sentence_embeddings (
    sentence_id  INTEGER PRIMARY KEY REFERENCES diary_sentences(id) ON DELETE CASCADE,
    chunk_type   TEXT NOT NULL DEFAULT 'sentence',
    model_name   TEXT NOT NULL,
    dimension    INTEGER NOT NULL,
    embedding    BLOB NOT NULL              -- float32 바이트
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
