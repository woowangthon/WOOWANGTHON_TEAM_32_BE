from datetime import date as Date

from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas import DiaryOut, DiaryTextCreate, DiaryUpdate, RecallResponse
from app.services.diary_service import DiaryNotFound, DiaryService, InvalidDiaryState

router = APIRouter(prefix="/api/diaries", tags=["diaries"]) # 각 경로들을 모으고, 이후 main.py에서 include_router로 애플리케이션과 연결.


def get_service(request: Request) -> DiaryService:
    return request.app.state.service # 현재 요청에서 애플리케이션의 service에 접근해 반환


def _not_implemented(detail: str) -> HTTPException:
    return HTTPException(status_code=501, detail=detail)


# /recall 등 정적 경로는 /{diary_id} 등 동적 경로보다 먼저 선언해야 의도한대로 실행할 수 있다.

@router.get("/recall", response_model=RecallResponse)
def recall(
    request: Request,
    query: str = Query(min_length=1), # 검색용 키워드 파라미터.
    start_date: Date | None = None,
    end_date: Date | None = None,
    top_k: int = Query(default=3, ge=1, le=20), # 뽑을 개수 파라미터. 1에서 20 사이. default는 3.
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
def list_diaries(request: Request, status: str | None = Query(default=None, pattern="^(draft|confirmed)$")) -> list[DiaryOut]: # status 파라미터. draft나 confirmed로 제한(다른 건 422 에러)
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
