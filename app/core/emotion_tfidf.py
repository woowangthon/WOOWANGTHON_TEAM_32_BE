"""TF-IDF + LogisticRegression 감정 분류기의 numpy 추론 (기획서 3.2).

학습은 AI 저장소 notebooks/02에서 scikit-learn으로 한다. 서버에는 scikit-learn을 깔지 않고,
내보낸 idf, coef, intercept와 어휘 목록으로 같은 계산을 numpy로 한다.
계산 순서는 노트북의 numpy_predict_proba와 같다.
"""
import json
from pathlib import Path

import numpy as np

from app.core.tokenizer import extract_tokens, ngrams

NGRAM_RANGE = (1, 2)  # 학습 때 확정한 설정 (bigram + NEG_)


class TfidfEmotionClassifier:
    def __init__(self, idf: np.ndarray, coef: np.ndarray, intercept: np.ndarray,
                 vocab: list[str], classes: list[str]) -> None:
        if coef.shape != (len(classes), len(vocab)) or idf.shape != (len(vocab),) or intercept.shape != (len(classes),):
            raise ValueError("모델 가중치 shape가 어휘 또는 라벨 수와 맞지 않습니다.")
        self.idf = idf.astype(np.float64)
        self.coef = coef.astype(np.float64)
        self.intercept = intercept.astype(np.float64)
        self.classes = list(classes)
        self._index = {token: i for i, token in enumerate(vocab)}
        self.vocab_size = len(vocab)

    @classmethod
    def load(cls, model_dir: Path, classes: list[str]) -> "TfidfEmotionClassifier":
        # allow_pickle=False: 숫자 배열만 읽는다. classes는 파일에서 읽지 않고 config의 라벨로 받는다.
        with np.load(model_dir / "emotion_tfidf.npz", allow_pickle=False) as z:
            idf, coef, intercept = z["idf"], z["coef"], z["intercept"]
        vocab = json.loads((model_dir / "emotion_tfidf_vocab.json").read_text(encoding="utf-8"))
        return cls(idf, coef, intercept, vocab, classes)

    def _proba(self, token_lists: list[list[str]]) -> np.ndarray:
        X = np.zeros((len(token_lists), self.vocab_size), dtype=np.float64)
        for row, tokens in enumerate(token_lists):
            for gram in ngrams(tokens, NGRAM_RANGE):
                col = self._index.get(gram)
                if col is not None:
                    X[row, col] += 1.0
        X = X * self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        z = X @ self.coef.T + self.intercept
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, sentences: list[str]) -> list[tuple[str, float]]:
        if not sentences:
            return []
        proba = self._proba([extract_tokens(s, use_negation=True) for s in sentences])
        best = proba.argmax(axis=1)
        return [(self.classes[i], float(proba[row, i])) for row, i in enumerate(best)]
