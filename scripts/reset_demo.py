"""데모 초기화 스크립트 (기획서 3.6, 6.2).

공개 데모를 깨끗한 시연 상태로 되돌린다.
1. 데이터베이스 파일(app.db, -wal, -shm)을 지운다.
2. 업로드된 이미지(data/uploads)를 지운다.
3. data/demo/synthetic_examples.json의 합성 일기를 확정 상태로 다시 넣는다.

사용법 (서버 API가 실행 중이면 먼저 멈춘다. DB 파일을 쓰는 중이면 안 된다):
    uv run python scripts/reset_demo.py --yes

--yes를 주지 않으면 아무것도 지우지 않는다.
"""
import argparse
import json
import sys
from datetime import date as Date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 저장소 루트를 import 경로에 추가

from app import config  # noqa: E402
from app.core.embedder import load_embedder  # noqa: E402
from app.core.model_loader import load_emotion_classifier  # noqa: E402
from app.core import emotion  # noqa: E402
from app.db import get_connection, init_db  # noqa: E402
from app.schemas import DiaryUpdate  # noqa: E402
from app.services.diary_service import DiaryService  # noqa: E402


def wipe(db_path: Path, upload_dir: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(db_path) + suffix)
        if target.exists():
            target.unlink()
            print(f"삭제: {target}")
    if upload_dir.exists():
        for f in upload_dir.iterdir():
            if f.is_file():
                f.unlink()
                print(f"삭제: {f}")


def seed(db_path: Path, demo_path: Path) -> int:
    demo = json.loads(demo_path.read_text(encoding="utf-8"))["diaries"]
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        emotion.set_classifier(load_emotion_classifier())
        service = DiaryService(conn, load_embedder())
        for item in demo:
            draft = service.create_text_draft(Date.fromisoformat(item["date"]), item["content"])
            service.update(draft.diary_id, DiaryUpdate(status="confirmed"))  # 태그 미지정 → 추천 태그 복사
        return len(demo)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="데모 DB와 업로드 이미지를 지우고 합성 일기를 다시 넣는다.")
    parser.add_argument("--yes", action="store_true", help="실제로 지운다. 없으면 실행하지 않는다.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--uploads", type=Path, default=config.UPLOAD_DIR)
    parser.add_argument("--demo", type=Path, default=config.DEMO_DATA_PATH)
    args = parser.parse_args()

    if not args.yes:
        print("--yes 옵션이 없어 아무것도 바꾸지 않았습니다.")
        sys.exit(1)
    if not args.demo.exists():
        print(f"데모 데이터가 없습니다: {args.demo}")
        sys.exit(1)

    wipe(args.db, args.uploads)
    count = seed(args.db, args.demo)
    print(f"완료: 합성 일기 {count}편을 확정 상태로 넣었습니다. ({args.db})")


if __name__ == "__main__":
    main()
