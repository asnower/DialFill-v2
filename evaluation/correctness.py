""" 評価用関数群 """
import sys
import re
import math

import nltk
import spacy
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.meteor_score import single_meteor_score
from collections import Counter
from tqdm import tqdm

from .utils import Tokenizer13a

def eval_bleu(answers, result, n):
    """
    応答文の BLEU-N (Papineni, ACL, 2002) を計算して返す

    Parameters
    ----------
    answers : list<list<str>>
        すべての正解文（単語系列のリスト）が格納されたリスト
    result : list<list<str>>
        すべての応答文（単語系列のリスト）が格納されたリスト
        answers と対応関係がある
    n : int
        BLEU-N のNの値
        本研究では1や2を使用

    Returns
    ----------
    float
        応答文の BLEU-N
    """
    if n == 1: weights = (1, 0, 0, 0)
    elif n == 2: weights = (0.5, 0.5, 0, 0)
    elif n == 3: weights = (0.33, 0.33, 0.33, 0)
    elif n == 4: weights = (0.25, 0.25, 0.25, 0.25)
    else:assert False,'range of parameter n : (1,2,3,4)'
    bleu_sum = 0
    for ans, res in zip(answers, result):
        bleu_sum += sentence_bleu([ans], res, weights=weights)
    return 100 * bleu_sum / len(result)


def eval_nist(answers, result, n=5):
    """
    応答文の NIST-N (Doddington, HLT, 2002) を計算して返す

    Parameters
    ----------
    answers : list<list<str>>
        すべての正解文（単語系列のリスト）が格納されたリスト
    result : list<list<str>>
        すべての応答文（単語系列のリスト）が格納されたリスト
        answers と対応関係がある
    n : int
        NIST-N のNの値
        本研究では5を使用している（2などのほうがよいか？）

    Returns
    ----------
    float
        応答文の NIST-N
    """
    answers_ = [[answer] for answer in answers]
    scores = corpus_nist(answers_, result, n)
    return scores


def eval_meteor(answers, result):
    """
    応答文の METEOR (Banerjee, ACL, 2005) を計算して返す

    Parameters
    ----------
    answers : list<list<str>>
        すべての正解文（単語系列のリスト）が格納されたリスト
    result : list<list<str>>
        すべての応答文（単語系列のリスト）が格納されたリスト
        answers と対応関係がある

    Returns
    ----------
    float
        応答文の METEOR
    """
    meteor_sum = 0
    for ans, res in zip(answers, result):
        ans, res = ' '.join(ans), ' '.join(res)
        if ans == res: meteor = 1.0
        else: meteor = single_meteor_score(ans, res)
        meteor_sum += meteor
    return 100 * meteor_sum / len(result)

def eval_acc(predictions, references, knowledge):
    def lower(text):
        if isinstance(text, str):
            text = text.strip().lower()
            text = ' '.join(nltk.word_tokenize(text))
            return text.strip()
        return [lower(item) for item in text]
    nlp = spacy.load("en_core_web_sm")
    ent_f1 = []
    k_f1 = []
    sent_acc = []
    zero_k_f1 = []

    for pred, refe, kg in zip(predictions, references, knowledge):
        label_knowledge = [lower(kg)]
        label_response = lower(refe)
        label_ents = [ent.text for ent in nlp(label_response).ents]

        pred_response = lower(pred)
        pred_ents = [ent.text for ent in nlp(pred_response).ents]
        if len(label_ents) > 0:
            ent_f1.append(f1_score(' '.join(pred_ents), [' '.join(label_ents)]))
        if len(label_knowledge) == 0:
            k_f1.append(0)
        else:
            k_score = f1_score(pred_response, label_knowledge)
            k_f1.append(k_score)
            zero_k_f1.append(1 if k_score == 0 else 0)

        sent_acc.append(f1_score(label_response, [pred_response]))

    return {'KF1': sum(k_f1) / len(k_f1) * 100,
            'ZeroK': sum(zero_k_f1) / len(zero_k_f1) * 100,
            'EntF1': sum(ent_f1) / len(ent_f1) * 100,
            'ACC': sum(sent_acc) / len(sent_acc) * 100}

def eval_kf1(predictions, knowledge):
    def lower(text):
        if isinstance(text, str):
            text = text.strip().lower()
            text = ' '.join(nltk.word_tokenize(text))
            return text.strip()
        return [lower(item) for item in text]
    
    k_f1 = []
    zero_k_f1 = []
    for pred, kg in zip(predictions, knowledge):
        label_knowledge = [lower(kg)]
        pred_response = lower(pred)

        if len(label_knowledge) == 0:
            k_f1.append(0)
        else:
            k_score = f1_score(pred_response, label_knowledge)
            k_f1.append(k_score)
            zero_k_f1.append(1 if k_score == 0 else 0)

    return {'KF1': sum(k_f1) / len(k_f1) * 100,
            'ZeroK': sum(zero_k_f1) / len(zero_k_f1) * 100}


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def rounder(num):
    return round(num, 2)

# ==================================================================================================
# F1 Score
# ==================================================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def f1_score(guess, answers):
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def eval_f1(predicts, answers):
    f1 = 0.
    for predict, answer in zip(predicts, answers):
        answer = answer.split('\t')
        f1 += f1_score(predict, answer)
    return {'F1': rounder(f1 * 100 / len(answers))}