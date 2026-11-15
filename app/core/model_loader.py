"""감정 분류 모델 선택과 로딩 (기획서 3.2).

환경 변수 EMOTION_MODEL로 고른다.
- tfidf (기본): TF-IDF + LogisticRegression, 가중치는 MODEL_DIR에서 읽는다.
- stub: 항상 '중립'을 반환한다. 테스트와 모델 파일이 없는 환경에서 쓴다.
- roberta: 학습이 끝나 평가에서 더 나을 때만 연결한다. 지금은 미구현.
"""
import os
from pathlib import Path

from app import config
from app.core.emotion import EmotionClassifier, StubEmotionClassifier
from app.core.emotion_tfidf import TfidfEmotionClassifier


def load_emotion_classifier(name: str | None = None, model_dir: Path | None = None) -> EmotionClassifier:
    name = (name or os.environ.get("EMOTION_MODEL", "tfidf")).lower()
    model_dir = model_dir or config.MODEL_DIR

    if name == "stub":
        return StubEmotionClassifier()
    if name == "tfidf":
        if not (model_dir / "emotion_tfidf.npz").exists():
            raise RuntimeError(
                f"감정 모델 가중치가 없습니다: {model_dir}. "
                "AI 저장소의 models/에서 복사하거나 EMOTION_MODEL=stub으로 실행하세요."
            )
        # 학습 때 sklearn의 classes_는 정렬된 라벨 순서다. 같은 순서로 맞춘다.
        return TfidfEmotionClassifier.load(model_dir, classes=sorted(config.EMOTION_LABELS))
    if name == "roberta":
        raise NotImplementedError("RoBERTa 감정 모델은 아직 연결되지 않았습니다.")
    raise ValueError(f"알 수 없는 EMOTION_MODEL 값입니다: {name}")
