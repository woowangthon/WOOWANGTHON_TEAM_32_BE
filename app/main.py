from contextlib import asynccontextmanager # app의 lifespan 관리
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import config
from app.api.routes import router
from app.core import emotion
from app.core.embedder import load_embedder
from app.core.model_loader import load_emotion_classifier
from app.core.ocr_client import ClovaOcrClient
from app.db import get_connection, init_db
from app.services.diary_service import DiaryService


def create_app(
    db_path: Path | None = None,
    upload_dir: Path | None = None,
    ocr_client: ClovaOcrClient | None = None,
) -> FastAPI:
    @asynccontextmanager # app의 lifespan 관리. yield 이전은 시작 시, yield 이후는 종료 시 실행.
    async def lifespan(app: FastAPI):
        emotion.set_classifier(load_emotion_classifier()) # EMOTION_MODEL에 맞는 감정 분류기 설정
        init_db(db_path) # db 테이블 생성(이미 있으면 유지)
        conn = get_connection(db_path) # db 연결
        service = DiaryService(conn, load_embedder(), upload_dir) # DiaryService 객체 생성
        # OCR 클라이언트: 테스트에서는 주입하고, 실행 환경에서는 환경 변수로 만든다 (설정 없으면 None)
        app.state.ocr = ocr_client if ocr_client is not None else ClovaOcrClient.from_env()
        service.load_index() # 일기 벡터를 읽어, 검색할 수 있는 인덱스 구성.
        app.state.service = service # 애플리케이션의 lifespan 동안 공통으로 사용할 수 있게 app.state에 저장
        try:
            yield
        finally:
            conn.close()

    app = FastAPI(title="Re:mind API", lifespan=lifespan) # API 담당하는 FastAPI 객체 생성
    app.include_router(router) # 라우터를 FastAPI 애플리케이션에 등록

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return FileResponse(config.BASE_DIR / "static" / "index.html", media_type="text/html; charset=utf-8")

    return app


app = create_app()
