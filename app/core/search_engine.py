"""NumPy 인메모리 벡터 인덱스 (기획서 2, 5).

- 확정된 일기의 문장 벡터만 담는다. draft는 인덱스에 들어가지 않는다.
- 검색은 날짜 필터 → 문장 코사인 유사도 → 일기별 최고 점수 → 임계값 → Top-K 순서.
"""
from dataclasses import dataclass, field
from datetime import date as Date

import numpy as np


@dataclass # 특수 메서드를 자동 생성하기 위한 데코레이터(__init__, __repr__, __eq__ 등)
class Entry:
    diary_id: int
    diary_date: Date
    sentence_text: str
    vector: np.ndarray  # L2 정규화된 float32


@dataclass
class SearchHit:
    diary_id: int
    diary_date: Date
    matched_text: str
    score: float


@dataclass
class VectorIndex:
    entries: list[Entry] = field(default_factory=list) # field의 default_factory=list를 통해 내부적으로 list()를 호출해 새 객체를 지정하도록 함

    def remove_diary(self, diary_id: int) -> None:
        self.entries = [e for e in self.entries if e.diary_id != diary_id]

    def replace_diary(
        self,
        diary_id: int,
        diary_date: Date,
        sentence_texts: list[str],
        vectors: np.ndarray,
    ) -> None:
        self.remove_diary(diary_id) # 해당 일기 제거하고
        for text, vec in zip(sentence_texts, vectors):
            self.entries.append(Entry(diary_id, diary_date, text, vec.astype(np.float32))) # 각 백터의 내부 타입을 float32로 설정해 entries에 추가

    def search(
        self,
        query_vector: np.ndarray, # 검색키워드(감정관련) 벡터
        start_date: Date | None,
        end_date: Date | None,
        top_k: int,
        threshold: float,
    ) -> list[SearchHit]:
        candidates = [
            e
            for e in self.entries
            if (start_date is None or e.diary_date >= start_date)
            and (end_date is None or e.diary_date <= end_date)
        ]
        if not candidates:
            return []

        matrix = np.stack([e.vector for e in candidates]) # (조건 맞는 벡터 수, 256) 크기의 행렬(np.stack의 기본 축: 행)
        scores = matrix @ query_vector.astype(np.float32)  # 정규화 벡터끼리의 내적 = 코사인 유사도

        # 일기별 최고 점수 문장만 남긴다. 한 일기의 여러 문장이 결과를 채우지 않게 한다.
        best: dict[int, SearchHit] = {}
        for entry, score in zip(candidates, scores):
            score = float(score)
            current = best.get(entry.diary_id) # 일기별 현재 최고 점수 문장 가져옴
            if current is None or score > current.score: # 없거나, 새 문장의 점수가 더 높으면
                best[entry.diary_id] = SearchHit( # 새 일기로 교체
                    entry.diary_id, entry.diary_date, entry.sentence_text, score
                )

        hits = [h for h in best.values() if h.score >= threshold] # 점수가 임계값 이상인 문장만 남김
        hits.sort(key=lambda h: h.score, reverse=True) # 점수 기준 내림차순 정렬
        return hits[:top_k] # top_k개 반환
