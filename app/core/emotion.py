"""감정 분류와 일기 태그 후보 규칙 (기획서 3.2).

모델 연결 전까지는 StubEmotionClassifier가 모든 문장을 '중립'으로 분류한다.
TF-IDF + LR 모델이 준비되면 같은 인터페이스(predict)를 구현해 교체한다.
"""
from collections import Counter
from typing import Protocol

from app import config


class EmotionClassifier(Protocol):
    def predict(self, sentences: list[str]) -> list[tuple[str, float]]:
        """문장마다 (라벨, 신뢰도)를 반환한다."""
        ...


class StubEmotionClassifier:
    def predict(self, sentences: list[str]) -> list[tuple[str, float]]:
        # TODO: 학습된 TF-IDF + LR 모델로 교체
        return [(config.NEUTRAL_LABEL, 1.0) for _ in sentences]


_classifier: EmotionClassifier = StubEmotionClassifier()


def get_classifier() -> EmotionClassifier:
    return _classifier


def set_classifier(classifier: EmotionClassifier) -> None:
    global _classifier
    _classifier = classifier


def suggest_tags(sentence_labels: list[tuple[str, float]]) -> list[str]:
    """기획서 3.2 일기 태그 규칙.

    1. 문장별 감정은 이미 분류된 상태로 받는다.
    2. 신뢰도가 임계값보다 낮은 문장은 제외한다.
    3. 중립을 제외하고 많이 나온 감정 순으로 최대 3개를 후보로 한다.
    4. 남은 감정이 없으면 '중립'을 후보로 한다.
    """
    kept = [
        label
        for label, conf in sentence_labels
        if conf >= config.SENTENCE_CONFIDENCE_THRESHOLD
    ]
    counts = Counter(label for label in kept if label != config.NEUTRAL_LABEL)
    if not counts:
        return [config.NEUTRAL_LABEL]

    # 동점은 기획 라벨 순서를 우선한다 (Counter는 삽입 순서를 유지하므로 정렬 키로 고정)
    order = {label: i for i, label in enumerate(config.EMOTION_LABELS)}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], order.get(kv[0], 99)))
    return [label for label, _ in ranked[: config.MAX_TAG_CANDIDATES]]
