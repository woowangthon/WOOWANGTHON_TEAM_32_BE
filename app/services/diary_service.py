"""일기 등록·확정·수정·삭제·회상 검색의 비즈니스 로직 (기획서 3.1, 3.3)."""
import sqlite3
import threading
from datetime import date as Date
from datetime import datetime

import numpy as np

from app import config
from app.core import emotion
from app.core.embedder import Embedder
from app.core.search_engine import Entry, VectorIndex
from app.core.sentence_splitter import split_sentences
from app.schemas import (
    DiaryOut,
    DiaryUpdate,
    RecallResponse,
    RecallResult,
    SentenceOut,
)


class DiaryNotFound(Exception):
    """일기를 찾지 못했을 때"""
    pass


class InvalidDiaryState(Exception):
    """요청이 현재 상태와 맞지 않을 때 (예: 날짜 없이 확정)."""
    pass


class DiaryService:
    def __init__(self, conn: sqlite3.Connection, embedder: Embedder): # db connection과 embedder 받아옴
        self.conn = conn
        self.embedder = embedder
        self.index = VectorIndex() # 일기 담고 관리하는 클래스(삭제, 수정, 검색 기능)
        self._lock = threading.Lock()  # SQLite 쓰기와 인덱스 갱신을 직렬화

    # ---------- 내부 헬퍼 ----------

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds") # 현재 날짜, 시간을 초 단위까지 담은 문자열('YYYY-MM-DDTHH:MM:SS')

    def _analyze(self, content: str) -> tuple[list[tuple[str, str, float]], list[str]]:
        """본문 → 문장별 (text, label, confidence)와 감정 라벨 후보."""
        sentences = split_sentences(content)
        if not sentences: # 문장이 없다면
            return [], [config.NEUTRAL_LABEL]
        predictions = emotion.get_classifier().predict(sentences) # 문장마다 (예측 감정 라벨, 신뢰도) 반환
        rows = [(text, label, conf) for text, (label, conf) in zip(sentences, predictions)] # (문장, 라벨, 신뢰도) 튜플들의 리스트
        suggested = emotion.suggest_tags(predictions) # 추천 감정 라벨 반환
        return rows, suggested

    def _store_analysis(self, diary_id: int, content: str) -> list[int]:
        """문장과 추천 감정 라벨을 다시 저장한다. 새로 생긴 문장 id 목록을 반환."""
        rows, suggested = self._analyze(content) # (문장, 예측 라벨, 신뢰도), 추천 감정 라벨 반환

        # 기존 내용 DB에서 삭제 [질문]
        self.conn.execute("DELETE FROM diary_sentences WHERE diary_id = ?", (diary_id,))
        self.conn.execute(
            "DELETE FROM diary_tags WHERE diary_id = ? AND source = 'suggested'", (diary_id,)
        )
        sentence_ids = []
        for idx, (text, label, conf) in enumerate(rows):
            cur = self.conn.execute( # 쿼리 실행하고 커서 객체 반환
                "INSERT INTO diary_sentences (diary_id, sentence_index, text, emotion_label, confidence)"
                " VALUES (?, ?, ?, ?, ?)",
                (diary_id, idx, text, label, conf),
            )
            sentence_ids.append(cur.lastrowid) # 마지막 insert된 행의 기본키를 리스트에 추가
        for label in suggested:
            self.conn.execute( # 추천 감정 라벨 DB에 저장
                "INSERT INTO diary_tags (diary_id, label, source) VALUES (?, ?, 'suggested')",
                (diary_id, label),
            )
        return sentence_ids

    def _replace_confirmed_tags(self, diary_id: int, tags: list[str]) -> None: # 확정된 감정 라벨 받아와, DB에 저장
        self.conn.execute(
            "DELETE FROM diary_tags WHERE diary_id = ? AND source = 'confirmed'", (diary_id,)
        )
        for label in dict.fromkeys(tags):  # 중복 제거, 순서 유지
            self.conn.execute(
                "INSERT INTO diary_tags (diary_id, label, source) VALUES (?, ?, 'confirmed')",
                (diary_id, label),
            )

    def _write_embeddings(self, diary_id: int) -> tuple[Date, list[str], np.ndarray]:
        """확정 일기의 문장 벡터를 DB에 다시 쓴다 (재벡터화). 인덱스 갱신은 커밋 후에 한다."""
        diary = self._row(diary_id) # 일기 가져옴
        sentences = self.conn.execute( # 일기 내용(+id) 전부 가져옴
            "SELECT id, text FROM diary_sentences WHERE diary_id = ? ORDER BY sentence_index",
            (diary_id,),
        ).fetchall()
        self.conn.execute( # 해당 일기의 문장은 sentence_embeddings에서 삭제
            "DELETE FROM sentence_embeddings WHERE sentence_id IN "
            "(SELECT id FROM diary_sentences WHERE diary_id = ?)",
            (diary_id,),
        )
        texts = [s["text"] for s in sentences]
        if texts:
            vectors = self.embedder.encode(texts).astype(np.float32) # 각 문장을 단위 벡터로 변환해 반환
            for s, vec in zip(sentences, vectors):
                self.conn.execute(
                    "INSERT INTO sentence_embeddings (sentence_id, chunk_type, model_name, dimension, embedding)"
                    " VALUES (?, 'sentence', ?, ?, ?)",
                    (s["id"], self.embedder.model_name, self.embedder.dimension, vec.tobytes()),
                )
        else: # 문장이 없다면 영벡터로 설정
            vectors = np.zeros((0, self.embedder.dimension), dtype=np.float32)
        return Date.fromisoformat(diary["date"]), texts, vectors

    def _row(self, diary_id: int) -> sqlite3.Row: # 해당하는 번호의 일기 가져옴
        row = self.conn.execute("SELECT * FROM diaries WHERE id = ?", (diary_id,)).fetchone()
        if row is None:
            raise DiaryNotFound(diary_id)
        return row

    def _tags(self, diary_id: int, source: str) -> list[str]: # 해당하는 일기의 감정 라벨 반환
        rows = self.conn.execute(
            "SELECT label FROM diary_tags WHERE diary_id = ? AND source = ? ORDER BY rowid", # 행 고유번호(rowid) 기준 오름차순 정렬
            (diary_id, source),
        ).fetchall()
        return [r["label"] for r in rows]

    def _is_indexed(self, diary_id: int) -> bool: # 일기 관리 클래스에 일기가 등록되어 있는지 확인
        return any(e.diary_id == diary_id for e in self.index.entries)

    def _to_out(self, diary_id: int, with_sentences: bool = False) -> DiaryOut: # 일기 조회 응답 결과 반환
        row = self._row(diary_id) # 일기 가져옴
        sentences: list[SentenceOut] = []
        if with_sentences:
            sentences = [ # 각 문장별 SentenceOut의 리스트
                SentenceOut(
                    sentence_index=s["sentence_index"],
                    text=s["text"],
                    emotion_label=s["emotion_label"],
                    confidence=s["confidence"],
                )
                for s in self.conn.execute( # 일기의 문장들 가져옴
                    "SELECT * FROM diary_sentences WHERE diary_id = ? ORDER BY sentence_index",
                    (diary_id,),
                ).fetchall()
            ]
        return DiaryOut( # 응답 결과 반환
            diary_id=row["id"],
            status=row["status"],
            date=Date.fromisoformat(row["date"]) if row["date"] else None,
            content=row["content"],
            image=None,  # 이미지 업로드·OCR 단계에서 연결 (현재 범위 밖)
            suggested_tags=self._tags(diary_id, "suggested"),
            emotion_tags=self._tags(diary_id, "confirmed"),
            is_indexed=self._is_indexed(diary_id),
            sentences=sentences,
        )

    # ---------- 공개 API ----------

    def load_index(self) -> None:
        """서버 시작 시 확정 일기의 벡터를 DB에서 읽어 인덱스를 구성한다."""
        with self._lock:
            self.index.entries.clear() # 기존 항목을 비우고 DB 기준으로 다시 채운다 (중복 방지)
            rows = self.conn.execute( # 감정이 확정된 일기의 id, date, 내용, 문장 벡터(byte 형태) 가져옴
                """
                SELECT d.id AS diary_id, d.date AS diary_date, s.text AS text, e.embedding AS embedding
                FROM sentence_embeddings e
                JOIN diary_sentences s ON s.id = e.sentence_id
                JOIN diaries d ON d.id = s.diary_id
                WHERE d.status = 'confirmed'
                ORDER BY d.id, s.sentence_index
                """
            ).fetchall()
            for r in rows:
                vec = np.frombuffer(r["embedding"], dtype=np.float32) # 기존 데이터 담긴 메모리를 np.float32 형태로 읽음(데이터 복사 대신 메모리 참조)
                self.index.entries.append( # 일기 관리 클래스에 일기(+인덱스) 추가
                    Entry(r["diary_id"], Date.fromisoformat(r["diary_date"]), r["text"], vec)
                )

    def create_text_draft(self, diary_date: Date, content: str) -> DiaryOut:
        with self._lock, self.conn: # 스레드 동기화(한번에 하나의 스레드만 사용). 트랜잭션 관리.
            now = self._now() # 현재 시간 문자열
            cur = self.conn.execute( # draft 일기 DB에 저장
                "INSERT INTO diaries (date, content, status, created_at, updated_at)"
                " VALUES (?, ?, 'draft', ?, ?)",
                (diary_date.isoformat(), content, now, now),
            )
            diary_id = cur.lastrowid # 가장 최근에 넣은 일기의 행 번호
            self._store_analysis(diary_id, content) # 문장과 추천 감정 라벨 저장
            return self._to_out(diary_id, with_sentences=True) # 일기 저장 결과 반환

    def get(self, diary_id: int) -> DiaryOut: # 일기 조회 결과 반환
        return self._to_out(diary_id, with_sentences=True)

    def list(self, status: str | None = None) -> list[DiaryOut]:
        sql = "SELECT id FROM diaries"
        params: tuple = ()
        if status: # status 명시되어 있으면 조건 추가
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY date DESC, id DESC" # 최신순으로 조회
        ids = [r["id"] for r in self.conn.execute(sql, params).fetchall()]
        return [self._to_out(i) for i in ids] # 일기들 조회 결과 리스트로 반환

    def update(self, diary_id: int, payload: DiaryUpdate) -> DiaryOut:
        with self._lock, self.conn: # 스레드 동기화. 트랜잭션 관리.
            row = self._row(diary_id) # 해당 일기 가져옴
            new_date = payload.date.isoformat() if payload.date else row["date"] # 페이로드에 날짜 있으면 그거로, 없으면 기존 날짜로 설정
            new_content = payload.content if payload.content is not None else row["content"] # 페이로드 내용이 명시적으로 None만 아니면 그거로, 아니면 기존 내용으로 설정

            # 확정된 일기는 계속 confirmed로 유지한다 (기획서 답변 9번).
            # draft는 payload.status가 confirmed일 때만 확정된다.
            target_status = (
                "confirmed"
                if row["status"] == "confirmed" or payload.status == "confirmed"
                else "draft"
            )
            if target_status == "confirmed" and not new_date: # DB date가 비어있었고 새로 날짜가 안 들어온 경우
                raise InvalidDiaryState("확정하려면 날짜(date)가 필요합니다.")

            content_changed = new_content != row["content"]
            self.conn.execute( # 일기 테이블 DB 업데이트
                "UPDATE diaries SET date = ?, content = ?, status = ?, updated_at = ? WHERE id = ?",
                (new_date, new_content, target_status, self._now(), diary_id),
            )

            if content_changed or not self._has_sentences(diary_id): # 내용이 바뀌었거나, 문장 테이블이 비어있는 경우 [질문]
                self._store_analysis(diary_id, new_content) # 문장과 추천 감정 라벨 저장

            if payload.emotion_tags is not None: # 페이로드에 확정 감정 라벨이 들어있으면
                self._replace_confirmed_tags(diary_id, payload.emotion_tags) # 확정 감정 라벨 DB 저장
            elif target_status == "confirmed" and row["status"] == "draft":
                # 기본 동작: draft를 확정할 때 감정 라벨을 보내지 않았으면 추천 라벨을 확정 라벨로 복사한다
                self._replace_confirmed_tags(diary_id, self._tags(diary_id, "suggested"))

            # 확정 또는 확정 일기 수정: 문장 벡터를 현재 본문 기준으로 다시 만든다 (재벡터화, DB 쓰기만)
            embedded = self._write_embeddings(diary_id) if target_status == "confirmed" else None

            self.conn.commit()  # DB 변경을 먼저 확정한 뒤에 인덱스를 갱신한다

            if embedded is not None:
                self.index.replace_diary(diary_id, *embedded) # 일기 관리 클래스에 반영
            else:
                self.index.remove_diary(diary_id) # 확정 안 된 일기는 인덱스에서 제거

            return self._to_out(diary_id, with_sentences=True)

    def delete(self, diary_id: int) -> None:
        with self._lock, self.conn: # 스레드 동기화. 트랜잭션 관리.
            row = self._row(diary_id) # 일기 가져옴
            self.conn.execute("DELETE FROM diaries WHERE id = ?", (diary_id,)) # DB에서 제거
            self.conn.commit()  # DB 커밋 후에 인덱스를 갱신한다
            self.index.remove_diary(diary_id) # 일기 관리 클래스에서도 제거
            # TODO: row["image_path"]가 있으면 업로드 이미지 파일도 삭제 (이미지 단계에서 구현)

    def recall(
        self,
        query: str,
        start_date: Date | None,
        end_date: Date | None,
        top_k: int,
    ) -> RecallResponse:
        query_vec = self.embedder.encode([query])[0]
        hits = self.index.search( # 일기 클래스 상에서 검색해서 top_k개 반환
            query_vec,
            start_date,
            end_date,
            top_k,
            threshold=config.RECALL_SCORE_THRESHOLD,
        )
        if not hits:
            return RecallResponse(results=[], message="비슷한 기록을 찾지 못했습니다.")

        results = []
        for h in hits:
            row = self._row(h.diary_id) # 일기 가져옴
            results.append(
                RecallResult(
                    diary_id=h.diary_id,
                    date=h.diary_date,
                    content=row["content"],
                    matched_text=h.matched_text,
                    image=None,
                    emotion_tags=self._tags(h.diary_id, "confirmed"), # 확정된 감정 라벨
                    score=round(h.score, 4),
                )
            )
        return RecallResponse(results=results, message=None)

    def _has_sentences(self, diary_id: int) -> bool:
        row = self.conn.execute( # 문장 개수 계산
            "SELECT COUNT(*) AS n FROM diary_sentences WHERE diary_id = ?", (diary_id,)
        ).fetchone()
        return row["n"] > 0 # 문장이 있는지 여부 반환
