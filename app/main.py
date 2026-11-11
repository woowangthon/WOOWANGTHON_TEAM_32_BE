from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app import config
from app.api.routes import router
from app.core.embedder import get_embedder
from app.db import get_connection, init_db
from app.services.diary_service import DiaryService


def create_app(db_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(db_path)
        conn = get_connection(db_path)
        service = DiaryService(conn, get_embedder())
        service.load_index()
        app.state.service = service
        try:
            yield
        finally:
            conn.close()

    app = FastAPI(title="Re:mind API", lifespan=lifespan)
    app.include_router(router)

    @app.get("/")
    def root() -> dict:
        return {"message": "Re:mind API", "docs": "/docs"}

    return app


app = create_app()
