import numpy as np
import pytest

from app import config
from app.core.emotion import StubEmotionClassifier
from app.core.emotion_tfidf import TfidfEmotionClassifier
from app.core.model_loader import load_emotion_classifier
from app.core.tokenizer import extract_tokens, ngrams


def test_ngrams_do_not_cross_nothing_and_keep_order():
    assert ngrams(["a", "b", "c"], (1, 2)) == ["a", "b", "c", "a b", "b c"]
    assert ngrams([], (1, 2)) == []


def test_negation_marks_content_word():
    # 부정 마커는 버리고, 범위에 든 내용어에 NEG_를 붙인다
    tokens = extract_tokens("행복하지 않다")
    assert "NEG_행복" in tokens
    assert "행복" not in tokens
    assert not any(t in ("않", "않다") for t in tokens)


def test_plain_content_words_are_kept():
    tokens = extract_tokens("오늘 발표를 무사히 끝냈다")
    assert "발표" in tokens
    assert not any(t.startswith("NEG_") for t in tokens)


def test_tfidf_predict_returns_label_and_probability(tmp_path):
    # 가중치를 0으로 두고 절편만 '슬픔'에 크게 준 모델: 어떤 문장이든 '슬픔'이 나와야 한다
    labels = sorted(config.EMOTION_LABELS)
    k = labels.index("슬픔")
    intercept = np.zeros(len(labels))
    intercept[k] = 5.0
    clf = TfidfEmotionClassifier(
        idf=np.ones(2), coef=np.zeros((len(labels), 2)), intercept=intercept,
        vocab=["x", "y"], classes=labels,
    )
    out = clf.predict(["오늘 발표를 했다.", "너무 피곤하다."])
    assert [label for label, _ in out] == ["슬픔", "슬픔"]
    assert all(0.0 < conf <= 1.0 for _, conf in out)
    assert clf.predict([]) == []


def test_tfidf_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        TfidfEmotionClassifier(
            idf=np.ones(3), coef=np.zeros((7, 2)), intercept=np.zeros(7),
            vocab=["x", "y"], classes=sorted(config.EMOTION_LABELS),
        )


def test_loader_selects_stub_and_rejects_unknown(tmp_path):
    assert isinstance(load_emotion_classifier("stub"), StubEmotionClassifier)
    with pytest.raises(ValueError):
        load_emotion_classifier("nope")
    with pytest.raises(NotImplementedError):
        load_emotion_classifier("roberta")


def test_loader_tfidf_without_weights_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="가중치"):
        load_emotion_classifier("tfidf", model_dir=tmp_path)


@pytest.mark.skipif(not (config.MODEL_DIR / "emotion_tfidf.npz").exists(), reason="가중치 파일 없음")
def test_real_weights_load_and_predict():
    clf = load_emotion_classifier("tfidf")
    out = clf.predict(["오늘 발표를 무사히 끝냈다.", "면접에서 떨어져서 하루 종일 울었다.", "그냥 평범한 하루였다."])
    assert len(out) == 3
    assert all(label in config.EMOTION_LABELS for label, _ in out)
    assert all(0.0 < conf <= 1.0 for _, conf in out)
