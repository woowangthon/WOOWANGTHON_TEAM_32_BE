from contextlib import asynccontextmanager # app의 lifespan 관리
from pathlib import Path

from fastapi import FastAPI

from app import config
from app.api.routes import router
from app.core.embedder import get_embedder
from app.db import get_connection, init_db
from app.services.diary_service import DiaryService


def create_app(db_path: Path | None = None) -> FastAPI:
    @asynccontextmanager # app의 lifespan 관리. yield 이전은 시작 시, yield 이후는 종료 시 실행.
    async def lifespan(app: FastAPI):
        init_db(db_path) # db 테이블 생성(이미 있으면 유지)
        conn = get_connection(db_path) # db 연결
        service = DiaryService(conn, get_embedder()) # DiaryService 객체 생성
        service.load_index() # 일기 벡터를 읽어, 검색할 수 있는 인덱스 구성.
        app.state.service = service # 애플리케이션의 lifespan 동안 공통으로 사용할 수 있게 app.state에 저장
        try:
            yield
        finally:
            conn.close()

    app = FastAPI(title="Re:mind API", lifespan=lifespan) # API 담당하는 FastAPI 객체 생성
    app.include_router(router) # 라우터를 FastAPI 애플리케이션에 등록

    @app.get("/")
    def root() -> dict:
        return {"message": "Re:mind API", "docs": "/docs"}

    return app


app = create_app()
