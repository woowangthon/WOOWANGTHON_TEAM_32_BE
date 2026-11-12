from functools import lru_cache

from kiwipiepy import Kiwi


@lru_cache(maxsize=1) # Kiwi 객체를 메모리에 캐싱해, 재호출 시 기존 객체 반환(마치 싱글톤)
def _kiwi() -> Kiwi:
    return Kiwi()


def split_sentences(text: str) -> list[str]:
    """Kiwi 문장 분리기로 본문을 문장 단위로 나눈다 (기획서 확정: Kiwi)."""
    cleaned = text.strip()
    if not cleaned:
        return []
    return [s.text.strip() for s in _kiwi().split_into_sents(cleaned) if s.text.strip()] # 문장 단위로 분리해, 빈 문자열이 아닌 경우만 리스트에 담아 반환
