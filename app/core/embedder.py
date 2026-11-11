"""문장 임베딩 (기획서 3.3: jhgan/ko-sroberta-multitask).

실제 모델 로딩은 sentence-transformers 설치 후 구현한다.
지금은 같은 인터페이스를 가진 HashingEmbedder로 파이프라인 전체를 검증한다.
"""
import hashlib
from typing import Protocol

import numpy as np

from app import config


class Embedder(Protocol):
    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """(n, dimension) float32 배열을 반환한다. 행은 L2 정규화되어 있다."""
        ...


class HashingEmbedder:
    """글자 2-gram 해시 기반 임시 임베더. 실제 모델이 아니므로 점수 해석에 쓰지 않는다."""

    model_name = "hashing-placeholder"
    dimension = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for i in range(len(text) - 1):
                bigram = text[i : i + 2].encode("utf-8")
                idx = int(hashlib.md5(bigram).hexdigest(), 16) % self.dimension
                vectors[row, idx] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


_embedder: Embedder = HashingEmbedder()


def get_embedder() -> Embedder:
    return _embedder


def set_embedder(embedder: Embedder) -> None:
    global _embedder
    _embedder = embedder
