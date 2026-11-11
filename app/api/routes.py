from datetime import date as Date

from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas import DiaryOut, DiaryTextCreate, DiaryUpdate, RecallResponse
from app.services.diary_service import DiaryNotFound, DiaryService, InvalidDiaryState

router = APIRouter(prefix="/api/diaries", tags=["diaries"])


def get_service(request: Request) -> DiaryService:
    return request.app.state.service


def _not_implemented(detail: str) -> HTTPException:
    return HTTPException(status_code=501, detail=detail)


# 주의: /recall은 /{diary_id}보다 먼저 선언해야 경로가 겹치지 않는다.

@router.get("/recall", response_model=RecallResponse)
def recall(
    request: Request,
    query: str = Query(min_length=1),
    start_date: Date | None = None,
    end_date: Date | None = None,
    top_k: int = Query(default=3, ge=1, le=20),
) -> RecallResponse:
    return get_service(request).recall(query, start_date, end_date, top_k)


@router.get("/analytics")
def analytics() -> None:
    raise _not_implemented("감정 통계는 다음 단계에서 구현합니다.")


@router.post("/image")
def create_from_image() -> None:
    raise _not_implemented("이미지 업로드·OCR은 OCR 연동 단계에서 구현합니다.")


@router.post("/text", response_model=DiaryOut)
def create_from_text(request: Request, payload: DiaryTextCreate) -> DiaryOut:
    return get_service(request).create_text_draft(payload.date, payload.content)


@router.get("", response_model=list[DiaryOut])
def list_diaries(request: Request, status: str | None = Query(default=None, pattern="^(draft|confirmed)$")) -> list[DiaryOut]:
    return get_service(request).list(status)


@router.get("/{diary_id}", response_model=DiaryOut)
def get_diary(request: Request, diary_id: int) -> DiaryOut:
    return _call(lambda: get_service(request).get(diary_id))


@router.patch("/{diary_id}", response_model=DiaryOut)
def update_diary(request: Request, diary_id: int, payload: DiaryUpdate) -> DiaryOut:
    return _call(lambda: get_service(request).update(diary_id, payload))


@router.delete("/{diary_id}", status_code=204)
def delete_diary(request: Request, diary_id: int) -> None:
    _call(lambda: get_service(request).delete(diary_id))


@router.get("/{diary_id}/similar")
def similar(diary_id: int) -> None:
    raise _not_implemented("비슷한 날 보기는 다음 단계에서 구현합니다.")


def _call(fn):
    """서비스 예외를 기획서 13번 규칙에 맞는 HTTP 응답으로 바꾼다."""
    try:
        return fn()
    except DiaryNotFound:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")
    except InvalidDiaryState as e:
        raise HTTPException(status_code=422, detail=str(e))
