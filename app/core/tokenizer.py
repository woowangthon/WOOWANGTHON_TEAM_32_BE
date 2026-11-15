"""감정 분류용 토큰 규칙 (기획서 3.2). AI 저장소 nlp/tokenize.py와 같아야 한다.

규칙을 바꾸면 두 저장소를 함께 고친다. 학습 때 쓴 규칙과 다르면 모델 점수가 틀어진다.

- 내용어: NNG, NNP, VV, VA (원형)
- 부정 마커(안, 못, 않, 없, 아니)는 버리고, 범위에 든 내용어만 NEG_ 접두사 토큰으로 바꾼다.
"""
from functools import lru_cache

from kiwipiepy import Kiwi

CONTENT_TAGS = {"NNG", "NNP", "VV", "VA"}
BOUNDARY_TAGS = {"EC", "EF", "ETM", "SF", "SP", "SS", "SE", "SO"}
LIGHT_VERB_FORMS = {"하", "되"}  # 안 하다, 못 되다 등: 건너뛰고 다음 내용어를 본다
MAX_SCOPE = 4  # 범위를 찾을 때 훑는 최대 토큰 수
NEG_PREFIX = "NEG_"


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


def _is_boundary(tag: str) -> bool:
    return tag.startswith("JK") or tag in BOUNDARY_TAGS


def _is_backward_marker(form: str, tag: str, prev_form: str | None, prev_tag: str | None) -> bool:
    if form == "않" and tag == "VX":
        return True
    if form == "못" and tag == "VX":
        return True
    if form in ("안", "못") and prev_form == "지" and prev_tag == "EC":
        return True  # 'V지 못하다', 'V지 안다'
    if form == "없" and tag == "VA":
        return True
    if form == "아니" and tag == "VCN":
        return True
    return False


def _is_forward_marker(form: str, tag: str) -> bool:
    return form in ("안", "못") and tag == "MAG"


def _mark_backward(tokens, i: int, role: list, negated: list[bool]) -> None:
    form = tokens[i].form
    j = i - 1
    if j >= 0:
        prev = tokens[j]
        if form in ("없", "아니") and (prev.tag.startswith("JK") or (prev.form == "이" and prev.tag == "VCP")):
            j -= 1  # '돈이 없다', '행복이 아니다'
        elif form in ("않", "못", "안") and prev.form == "지" and prev.tag == "EC":
            j -= 1  # '행복하지 않다'
    steps = 0
    while j >= 0 and steps < MAX_SCOPE:
        t = tokens[j]
        if _is_boundary(t.tag):
            break
        if t.tag in CONTENT_TAGS and role[j] is None:
            negated[j] = True
        j -= 1
        steps += 1


def _mark_forward(tokens, i: int, role: list, negated: list[bool]) -> None:
    j = i + 1
    steps = 0
    while j < len(tokens) and steps < MAX_SCOPE:
        t = tokens[j]
        if _is_boundary(t.tag):
            break
        if t.form in LIGHT_VERB_FORMS and t.tag in ("VV", "XSV", "XSA"):
            j += 1
            steps += 1
            continue
        if t.tag in CONTENT_TAGS and role[j] is None:
            negated[j] = True
            break
        j += 1
        steps += 1


def extract_tokens(text: str, use_negation: bool = True) -> list[str]:
    """문장을 감정 분류용 토큰 목록으로 바꾼다. 학습 때와 같은 기본값(use_negation=True)을 쓴다."""
    tokens = list(_kiwi().tokenize(text.strip()))
    n = len(tokens)
    if n == 0:
        return []

    if not use_negation:
        return [t.form for t in tokens if t.tag in CONTENT_TAGS]

    role: list[str | None] = [None] * n  # None | "marker" | "compound"
    negated = [False] * n
    compound_base: dict[int, str] = {}

    # 1) '재미없다'처럼 한 토큰으로 붙은 어휘 부정은 어근에 NEG_를 붙인다.
    for i, t in enumerate(tokens):
        if t.tag == "VA" and len(t.form) > 1 and t.form.endswith("없"):
            role[i] = "compound"
            compound_base[i] = t.form[:-1]

    # 2) 마커를 찾아 범위를 표시한다.
    for i, t in enumerate(tokens):
        if role[i] is not None:
            continue
        prev = tokens[i - 1] if i > 0 else None
        prev_form = prev.form if prev else None
        prev_tag = prev.tag if prev else None
        if _is_backward_marker(t.form, t.tag, prev_form, prev_tag):
            role[i] = "marker"
            _mark_backward(tokens, i, role, negated)
        elif _is_forward_marker(t.form, t.tag):
            role[i] = "marker"
            _mark_forward(tokens, i, role, negated)

    # 3) 토큰 목록 구성
    out: list[str] = []
    for i, t in enumerate(tokens):
        if role[i] == "marker":
            continue
        if role[i] == "compound":
            out.append(NEG_PREFIX + compound_base[i])
            continue
        if t.tag not in CONTENT_TAGS:
            continue
        out.append(NEG_PREFIX + t.form if negated[i] else t.form)
    return out


def ngrams(tokens: list[str], ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """이미 토큰화된 목록에서 n-gram을 만든다. 문장 경계를 넘지 않는다(학습 때 ngram_analyzer와 같다)."""
    lo, hi = ngram_range
    out: list[str] = []
    for n in range(lo, hi + 1):
        out.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return out
