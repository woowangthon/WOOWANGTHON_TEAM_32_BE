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
    path = db_path or config.DB_PATH # 인자로 받은 db_path가 비어 있으면 config.DB_PATH로 대체
    path.parent.mkdir(parents=True, exist_ok=True) # path의 상위 폴더가 존재하지 않으면 생성. 기준은 현재 작업 디렉터리(실행한 위치)
    conn = sqlite3.connect(path, check_same_thread=False) # db 연결. 요청마다 별개의 스레드를 사용하는 fastapi 특성상, 어떤 스레드가 접근해도 conn 객체를 쓸 수 있도록 허용.
    conn.row_factory = sqlite3.Row # 컬럼 이름으로 접근할 수 있도록 함.
    conn.execute("PRAGMA foreign_keys = ON") # db의 외래 키 제약 조건 활성화(sqlite는 기본적으로 비활성화 상태)
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA) # 한번에 여러 sql 문 일괄 실행
        conn.commit() # 메모리 상에서 db에 가한 변경을 실제 파일에 반영
    finally:
        conn.close() # db 연결 종료. 관련 리소스/파일 잠금 해제.
