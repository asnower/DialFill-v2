import math
from collections import Counter

from .utils import Tokenizer13a


def count_ngrams(text, n):
    ngrams = [tuple(text[i:i+n]) for i in range(len(text) - n + 1)]
    return Counter(ngrams)


def repeat_score_at_5(result, tokenizer=Tokenizer13a()):
    result = [tokenizer(p) for p in result]
    cuml_ngrams = 0
    weighted_ngrams = 0
    one_grams = 0

    for text in result:
        one_grams += len(text)

        for i in range(1, 6):
            ngram_counts = count_ngrams(text, i)
            repeated_ngrams = sum(count for ngram, count in ngram_counts.items() if count > 1)

            cuml_ngrams += repeated_ngrams
            weighted_ngrams += (2 ** i) * repeated_ngrams

    if cuml_ngrams == 0:
        return 0

    repeat_score_5 = math.log2(weighted_ngrams / cuml_ngrams) * one_grams
    return repeat_score_5 / len(result)


def eval_repeat(result, tokenizer=Tokenizer13a()):
    """
    応答文の Repeat (Nakamura, AAAI, 2019) を計算して返す
    すべての応答文のうち、同じ単語が二回以上出現する応答文の割合

    Parameters
    ----------
    result : list<list<str>>
        すべての応答文（単語系列のリスト）が格納されたリスト

    Returns
    ----------
    float
        応答文の Repeat
    """
    result = [tokenizer(p) for p in result]
    repeat_sum = 0
    for res in result:
        word_set = set()
        for word in res:
            if word in word_set:
                repeat_sum += 1
                break
            word_set.add(word)
    return 100 * repeat_sum / len(result)