"""OCR 본문에서 날짜를 찾는다 (기획서 3.1).

연도가 있는 표기만 인식한다. '11월 4일'처럼 연도가 없으면 None을 반환하고,
사용자가 직접 입력하도록 둔다.
"""
import re
from datetime import date as Date

_PATTERN = re.compile(r"(\d{4})\s*[년./\-]\s*(\d{1,2})\s*[월./\-]\s*(\d{1,2})\s*일?")


def extract_date(text: str) -> Date | None:
    for m in _PATTERN.finditer(text):
        year, month, day = (int(g) for g in m.groups())
        try:
            return Date(year, month, day)  # 존재하지 않는 날짜(2월 30일 등)는 건너뛴다
        except ValueError:
            continue
    return None
