"""문장 임베딩 (기획서 3.3: jhgan/ko-sroberta-multitask).

실제 모델 로딩은 sentence-transformers 설치 후 구현한다.
지금은 같은 인터페이스를 가진 HashingEmbedder로 파이프라인 전체를 검증한다.
"""
import hashlib
from typing import Protocol # 상속 대신, 최소한의 구조만 검증하기 위함.

import numpy as np

from app import config


class Embedder(Protocol): # Embedder라면 해당 멤버변수와 멤버함수를 갖춰야 함
    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: # SentenceTransformer.encode()와 동일한 형태
        """(n, dimension) float32 배열을 반환한다. 행은 L2 정규화되어 있다."""
        ...


class HashingEmbedder:
    """글자 2-gram 해시 기반 임시 임베더. 실제 모델이 아니므로 점수 해석에 쓰지 않는다."""

    model_name = "hashing-placeholder"
    dimension = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32) # 문장 개수 N에 맞춰 (N, 256) 크기의 float32 배열 할당
        for row, text in enumerate(texts): # 각 문장에 대해서
            for i in range(len(text) - 1):
                bigram = text[i : i + 2].encode("utf-8") # bigram 만들고
                idx = int(hashlib.md5(bigram).hexdigest(), 16) % self.dimension # 사전 구축 없이, 다른 bigram이라면 다른 인덱스에 갈 것으로 가정(일종의 2-gram BoW)
                vectors[row, idx] += 1.0 # 같은 bigram이라면 항상 같은 인덱스로 가서 빈도가 1 추가됨
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) # 행 단위로 L2 Norm 계산
        norms[norms == 0] = 1.0 # 빈 행의 경우, 0으로 나누기를 막기 위해 1로 변경.
        return vectors / norms # 단위 벡터로 변환해 반환


_embedder: Embedder = HashingEmbedder() # default embedder 설정


def get_embedder() -> Embedder:
    return _embedder


def set_embedder(embedder: Embedder) -> None:
    global _embedder
    _embedder = embedder
