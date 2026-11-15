import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # config.py 기준으로 2단계 위의 상위 폴더(루트 디렉터리)

# 테스트 등에서 DATA_DIR 환경 변수로 경로를 바꿀 수 있게 한다.
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "app.db"
UPLOAD_DIR = DATA_DIR / "uploads"
# 리셋 스크립트가 넣는 데모 일기 (합성 데이터, Git 포함)
DEMO_DATA_PATH = DATA_DIR / "demo" / "synthetic_examples.json"

# 감정 모델 가중치 위치 (Git 제외, 서버에 직접 업로드). 환경 변수 MODEL_DIR로 바꿀 수 있다.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))

# 감정 라벨 (기획서 3.2, 확정)
EMOTION_LABELS = ["기쁨", "슬픔", "분노", "불안", "상처", "당황", "중립"]
NEUTRAL_LABEL = "중립"

# 임계값 (기획서 3.2 / 3.3). AI Hub 검증 데이터로 조정 예정이며 현재는 잠정값.
SENTENCE_CONFIDENCE_THRESHOLD = 0.5
RECALL_SCORE_THRESHOLD = 0.5
MAX_TAG_CANDIDATES = 3

# 임베딩 모델 (기획서 3.3, 확정). 실제 로딩은 core/embedder.py에서 담당.
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# 이미지 업로드와 Clova OCR (기획서 3.1, 8)
# 키와 호출 주소는 환경 변수로만 받는다 (코드·Git에 넣지 않음). 없으면 이미지 등록은 503.
CLOVA_OCR_URL = os.environ.get("CLOVA_OCR_URL", "")
CLOVA_OCR_SECRET = os.environ.get("CLOVA_OCR_SECRET", "")
OCR_DAILY_LIMIT = 100  # 하루 OCR 호출 한도
OCR_MAX_BYTES = 5 * 1024 * 1024  # 업로드 최대 5MB
