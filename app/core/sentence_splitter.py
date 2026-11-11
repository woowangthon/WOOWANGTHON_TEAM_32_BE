from functools import lru_cache

from kiwipiepy import Kiwi


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


def split_sentences(text: str) -> list[str]:
    """Kiwi 문장 분리기로 본문을 문장 단위로 나눈다 (기획서 확정: Kiwi)."""
    cleaned = text.strip()
    if not cleaned:
        return []
    return [s.text.strip() for s in _kiwi().split_into_sents(cleaned) if s.text.strip()]
