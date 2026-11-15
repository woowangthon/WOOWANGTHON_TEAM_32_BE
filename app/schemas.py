from datetime import date as Date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DiaryTextCreate(BaseModel):
    """텍스트 등록 요청. 날짜는 검색을 위해 필수 (기획서 확정)."""
    date: Date
    content: str = Field(min_length=1)


class DiaryUpdate(BaseModel):
    """수정·확정 요청. status=confirmed면 확정 처리하며 벡터를 (재)생성한다."""
    date: Optional[Date] = None
    content: Optional[str] = Field(default=None, min_length=1)
    emotion_tags: Optional[list[str]] = None
    status: Optional[Literal["draft", "confirmed"]] = None


class SentenceOut(BaseModel):
    sentence_index: int
    text: str
    emotion_label: str
    confidence: float


class DiaryOut(BaseModel):
    diary_id: int
    status: Literal["draft", "confirmed"]
    date: Optional[Date]
    content: str
    image: Optional[str]
    suggested_tags: list[str]
    emotion_tags: list[str]
    is_indexed: bool
    sentences: list[SentenceOut] = []


class RecallResult(BaseModel):
    diary_id: int
    date: Date
    content: str
    matched_text: str
    image: Optional[str]
    emotion_tags: list[str]
    score: float


class RecallResponse(BaseModel):
    results: list[RecallResult]
    message: Optional[str] = None


class AnalyticsOut(BaseModel):
    """감정 통계 (기획서 4.4). average는 일기별 태그를 1/n씩 나눠 기간 평균을 낸 값이다."""
    start_date: Optional[Date] = None
    end_date: Optional[Date] = None
    diary_count: int
    average: dict[str, float]
    top_keywords: list[str]
