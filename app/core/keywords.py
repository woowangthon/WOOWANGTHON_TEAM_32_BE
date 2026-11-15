"""일기 키워드 추출 (기획서 3.5): Kiwi 명사 추출 + TF-IDF 점수 상위 키워드."""
import math
from collections import Counter

from app.core.tokenizer import get_kiwi

NOUN_TAGS = {"NNG", "NNP"}
# 일기에 거의 항상 나오지만 주제를 나타내지 않는 명사 (키워드에서 뺀다)
STOPWORDS = {"오늘", "하루", "사람", "지금", "다음", "이번", "정도", "때문", "사이", "요즘", "동안", "이후"}


def extract_nouns(text: str) -> list[str]:
    """본문에서 일반 명사·고유 명사를 뽑는다. 한 글자 명사는 뺀다."""
    return [
        t.form
        for t in get_kiwi().tokenize(text.strip())
        if t.tag in NOUN_TAGS and len(t.form) > 1 and t.form not in STOPWORDS
    ]


def top_keywords(texts: list[str], top_k: int = 5) -> list[str]:
    """여러 일기(문서)에서 TF-IDF 점수가 높은 명사 상위 top_k개를 고른다.

    점수 = 문서별 (단어 빈도 / 문서 명사 수) × idf 의 합. 동점은 단어 순서로 고정한다.
    """
    docs = [Counter(extract_nouns(t)) for t in texts]
    docs = [d for d in docs if d]
    if not docs:
        return []

    n_docs = len(docs)
    df = Counter(word for d in docs for word in d)
    scores: Counter = Counter()
    for d in docs:
        total = sum(d.values())
        for word, count in d.items():
            idf = math.log((n_docs + 1) / (df[word] + 1)) + 1.0
            scores[word] += (count / total) * idf
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:top_k]]
