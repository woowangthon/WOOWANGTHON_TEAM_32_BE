"""문장 임베딩 (기획서 3.3: jhgan/ko-sroberta-multitask).

- sroberta: sentence-transformers로 ko-sroberta-multitask를 로드한다. 기본값이다.
- hashing: 글자 2-gram 해시 임베더. 모델 없이 파이프라인을 테스트할 때만 쓴다. 점수 해석에 쓰지 않는다.
환경 변수 EMBEDDER로 고른다. 모델은 처음 실행할 때 Hugging Face에서 내려받아 캐시에 둔다.
"""
import hashlib
import os
from typing import Protocol  # 상속 대신, 최소한의 구조만 검증하기 위함.

import numpy as np

from app import config


class Embedder(Protocol):  # Embedder라면 해당 멤버변수와 멤버함수를 갖춰야 함
    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:  # SentenceTransformer.encode()와 동일한 형태
        """(n, dimension) float32 배열을 반환한다. 행은 L2 정규화되어 있다."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME) -> None:
        # torch가 무거우므로 실제로 쓸 때만 import한다. 테스트의 hashing 경로는 torch를 읽지 않는다.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        # sentence-transformers 5.x 이후 이름이 바뀌었으므로 새 이름을 먼저 쓰고, 없으면 이전 이름을 쓴다.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """글자 2-gram 해시 기반 임시 임베더. 실제 모델이 아니므로 점수 해석에 쓰지 않는다."""

    model_name = "hashing-placeholder"
    dimension = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)  # 문장 개수 N에 맞춰 (N, 256) float32 배열 할당
        for row, text in enumerate(texts):  # 각 문장에 대해서
            for i in range(len(text) - 1):
                bigram = text[i : i + 2].encode("utf-8")  # bigram 만들고
                idx = int(hashlib.md5(bigram).hexdigest(), 16) % self.dimension  # 같은 bigram은 항상 같은 인덱스로 간다
                vectors[row, idx] += 1.0  # 같은 bigram이면 같은 칸의 빈도가 올라간다
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # 행 단위로 L2 Norm 계산
        norms[norms == 0] = 1.0  # 빈 행의 0으로 나누기를 막는다
        return vectors / norms  # 단위 벡터로 변환해 반환


def load_embedder(name: str | None = None) -> Embedder:
    name = (name or os.environ.get("EMBEDDER", "sroberta")).lower()
    if name == "sroberta":
        return SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"알 수 없는 EMBEDDER 값입니다: {name}")
