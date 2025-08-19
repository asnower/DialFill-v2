from itertools import groupby


def compress_repeats(seq, repeat_num: int = 2):
    """
    对输入序列进行行程编码：
    - 对于连续出现 >=2 次的相同值，用 "值 * 次数" 表示；
    - 否则直接保留原值。
    """
    compressed = []
    for val, group in groupby(seq):
        count = sum(1 for _ in group)
        if count >= repeat_num:
            compressed.append(f"{val} * {count}")
        else:
            compressed.append(val)
    return compressed