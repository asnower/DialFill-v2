from .utils import Tokenizer13a


def eval_length(result, tokenizer=Tokenizer13a()):
    """
    応答文の平均の単語数を返す

    Parameters
    ----------
    result : list<list<str>>
        すべての応答文（単語系列のリスト）が格納されたリスト

    Returns
    ----------
    float
        応答文の平均の単語数
    """
    result = [tokenizer(p) for p in result]
    length_sum = 0
    for res in result:
        length_sum += len(res)
    return length_sum / len(result)


def eval_distinct(result, n, tokenizer=Tokenizer13a()):
    """
    応答文の Distinct-N (Li, NAACL, 2016) を計算して返す

    Parameters
    ----------
    result : list<list<str>>
        すべての応答文（単語系列のリスト）が格納されたリスト
    n : int
        Distinct-N のNの値
        本研究では1や2を使用

    Returns
    ----------
    float
        応答文の Distinct-N
    """
    result = [tokenizer(p) for p in result]
    ngram_set = set()
    ngram_sum = 0
    for res in result:
        length = len(res) - n + 1
        ngram_sum += length
        for i in range(length):
            ngram_set.add(' '.join(res[i:i+n]))
    return 100 * len(ngram_set) / ngram_sum