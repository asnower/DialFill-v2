import nltk
import re


def extract_prefix_regex(text: str) -> str:
    """
    基于正则表达式：
    - (?=...) 是零宽度断言，匹配到标记前停止，不消费标记本身。
    - re.DOTALL 使 . 能匹配换行。
    """
    if '###' in text:
        pattern = r'^(.*?)(?=### keywords of response:)'
    else:
        pattern = r'^(.*?)(?= keywords of response:)'
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else text


def extract_keywords(text):
    # 1. 提取 keywords of response 后的关键词
    if '###' in text:
        pattern_keywords = r'### keywords of response:\s*(.*?)### masked response:'
    else:
        pattern_keywords = r' keywords of response:\s*(.*?) masked response:'
    match_keywords = re.search(pattern_keywords, text, flags=re.DOTALL)
    if match_keywords:
        keywords = match_keywords.group(1).strip()
        return keywords
    else:
        return ""


def extract_masked_text(text):
    # 2. 提取 masked response 后的内容，并计算其长度（字符数
    if '###' in text:
        pattern_masked = r'### masked response:\s*(.*?)### response:'
    else:
        pattern_masked = r'masked response:\s*(.*?) response:'
    match_masked = re.search(pattern_masked, text, flags=re.DOTALL)
    if match_masked:
        masked_content = match_masked.group(1).strip()
        return masked_content
    else:
        return ""


def lower(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = ' '.join(nltk.word_tokenize(text))
        return text.strip()
    return [lower(item) for item in text]


def eval_special_acc(inputs, predictions, knowledges):
    kwd_acc, kwd_used, msk_err, msk_err_abs = [], [], [], []

    for inp, pred, kg in zip(inputs, predictions, knowledges):
        pred_response = lower(pred)
        label_knowledge = lower(kg)

        keywords = lower(extract_keywords(inp))
        masked = lower(extract_masked_text(inp))

        kwd_acc.append(int(all([k in label_knowledge for k in keywords.split()])) if keywords else 0)
        kwd_used.append(int(any([k in pred_response for k in keywords.split()])) if keywords else 0)
        msk_err.append(len(masked.split()) - len(pred_response.split()))
        msk_err_abs.append(abs(len(masked.split()) - len(pred_response.split())))

    return {'KWD_ACC': sum(kwd_acc) / len(kwd_acc) * 100,
            'KWD_USD': sum(kwd_used) / len(kwd_used) * 100,
            "MSK_ERR": sum(msk_err_abs) / len(msk_err_abs),
            "MSK_MAX": max(msk_err),
            "MSK_MIN": min(msk_err),
           }