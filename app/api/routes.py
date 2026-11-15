from datetime import date as Date
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from app import config
from app.core.ocr_client import OcrError
from app.schemas import AnalyticsOut, DiaryOut, DiaryTextCreate, DiaryUpdate, RecallResponse
from app.services.diary_service import (
    DiaryNotFound,
    DiaryService,
    InvalidDiaryState,
    OcrQuotaExceeded,
)

router = APIRouter(prefix="/api/diaries", tags=["diaries"]) # 각 경로들을 모으고, 이후 main.py에서 include_router로 애플리케이션과 연결.

JPEG_MAGIC = bytes([0xFF, 0xD8, 0xFF])  # jpg 파일 시작 바이트


def get_service(request: Request) -> DiaryService:
    return request.app.state.service # 현재 요청에서 애플리케이션의 service에 접근해 반환


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


@router.get("/analytics", response_model=AnalyticsOut)
def analytics(
    request: Request,
    start_date: Date | None = None,
    end_date: Date | None = None,
) -> AnalyticsOut:
    return get_service(request).analytics(start_date, end_date)


@router.post("/image", response_model=DiaryOut)
def create_from_image(request: Request, file: UploadFile = File(...)) -> DiaryOut:
    ocr = request.app.state.ocr
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR 설정이 없어 이미지 등록을 할 수 없습니다.")

    name = (file.filename or "").lower()
    if not name.endswith((".jpg", ".jpeg")):
        raise HTTPException(status_code=422, detail="jpg 파일만 올릴 수 있습니다.")
    data = file.file.read(config.OCR_MAX_BYTES + 1) # 한도보다 1바이트 더 읽어서 초과 여부를 확인한다
    if len(data) > config.OCR_MAX_BYTES:
        raise HTTPException(status_code=422, detail="5MB 이하 파일만 올릴 수 있습니다.")
    if not data.startswith(JPEG_MAGIC): # JPEG 시그니처 확인 (확장자만 바꾼 파일 거르기)
        raise HTTPException(status_code=422, detail="올바른 jpg 파일이 아닙니다.")

    return _call(lambda: get_service(request).create_from_image(
        data, ocr.recognize, config.OCR_DAILY_LIMIT, _today_kst(),
    ))


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


@router.get("/{diary_id}/image")
def get_image(request: Request, diary_id: int) -> FileResponse:
    path = _call(lambda: get_service(request).get_image_path(diary_id))
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="이미지가 없습니다.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{diary_id}/similar", response_model=RecallResponse)
def similar(
    request: Request,
    diary_id: int,
    top_k: int = Query(default=3, ge=1, le=20),
) -> RecallResponse:
    return _call(lambda: get_service(request).similar(diary_id, top_k))


def _call(fn):
    """서비스 예외를 기획서 13번 규칙에 맞는 HTTP 응답으로 바꾼다."""
    try:
        return fn()
    except DiaryNotFound:
        raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")
    except InvalidDiaryState as e:
        raise HTTPException(status_code=422, detail=str(e))
    except OcrQuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except OcrError as e:
        raise HTTPException(status_code=502, detail=str(e))


def _today_kst() -> Date:
    """OCR 한도는 한국 날짜 기준으로 센다 (서버 시간대와 무관하게 고정)."""
    return datetime.now(timezone(timedelta(hours=9))).date()
