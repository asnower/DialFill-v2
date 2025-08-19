import random
import math
from typing import Any, List, Tuple, Iterator
from collections import Counter, defaultdict
from torch.utils.data import Sampler

from torch.utils.data import Dataset


def allocate_samples(total_length, ratios):
    # 初始计算，不进行四舍五入
    raw_samples = [total_length * ratio for ratio in ratios]

    # 取整数部分
    integer_parts = [int(sample) for sample in raw_samples]

    # 计算剩余的样本数量
    remaining_samples = total_length - sum(integer_parts)

    # 计算小数部分的剩余
    remainders = [sample - int_part for sample, int_part in zip(raw_samples, integer_parts)]

    # 根据小数部分的大小排序索引，降序排列
    indices = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)

    # 分配剩余的样本
    for i in range(int(remaining_samples)):
        integer_parts[indices[i]] += 1

    return integer_parts


class MixedDataset(Dataset):
    def __init__(
            self,
            datasets: List[Tuple[Dataset, float]],
    ):
        total_length = len(datasets[0][0]) if datasets else 0
        dataset_lengths = [len(dataset) for dataset, _ in datasets]
        assert all(x == total_length for x in dataset_lengths)
        ratios = [ratio for _, ratio in datasets]
        assert sum(ratios) <= 1, f'{ratios=}'
        if sum(ratios) > 0 and sum(ratios) < 1:
            scale_factor = 1 / sum(ratios)
            ratios = [x * scale_factor for x in ratios]

        lengths = allocate_samples(total_length, ratios)
        remaining_indices = list(range(total_length))
        self.datasets, self.indices = [], []
        for i, (dataset, _) in enumerate(datasets):
            sample_size = lengths[i]
            sampled_indices = random.sample(remaining_indices, sample_size)
            remaining_indices = [idx for idx in remaining_indices if idx not in sampled_indices]

            self.datasets.append(dataset)
            self.indices.append(sampled_indices)

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.indices)

    def __getitem__(self, index: int):
        for dataset_index, (dataset, sampled_indices) in enumerate(zip(self.datasets, self.indices)):
            dataset_length = len(sampled_indices)
            if index < dataset_length:
                try:
                    actual_index = sampled_indices[index]
                    return dataset[actual_index]
                except:
                    assert 0, f"{actual_index=} {len(dataset)=} {index=}"
            index -= dataset_length
        raise IndexError(f"Index {index} out of range for MixedDataset")


class ConcatenatedDataset(Dataset):
    def __init__(
            self,
            datasets: List[Tuple[Dataset, float]],
    ):
        self.dataset_lengths = [len(dataset) for dataset, _ in datasets]

        self.datasets, self.indices = [], []
        for dataset, ratio in datasets:
            if len(dataset) > 0:
                sampled_indices = random.sample(range(len(dataset)), round(len(dataset) * ratio))
                self.datasets.append(dataset)
                self.indices.append(sampled_indices)
            else:
                self.datasets.append([])
                self.indices.append([])

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.indices)

    def __getitem__(self, index: int):
        for dataset_index, (dataset, sampled_indices) in enumerate(zip(self.datasets, self.indices)):
            dataset_length = len(sampled_indices)
            if index < dataset_length:
                actual_index = sampled_indices[index]
                return dataset[actual_index]
            index -= dataset_length
        raise IndexError(f"Index {index} out of range for MixedDataset")


class ContrastiveDataset(Dataset):
    def __init__(
            self,
            pos_dataset: Tuple[Dataset, float],
            neg_dataset: Tuple[Dataset, float],
    ):

        self.datasets, self.indices = [], []
        for dataset, ratio in pos_dataset:
            if len(dataset) > 0:
                sampled_indices = random.sample(range(len(dataset)), round(len(dataset) * ratio))
                self.datasets.append(dataset)
                self.indices.append(sampled_indices)
            else:
                self.datasets.append([])
                self.indices.append([])

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.indices)

    def __getitem__(self, index: int):
        for dataset_index, (dataset, sampled_indices) in enumerate(zip(self.datasets, self.indices)):
            dataset_length = len(sampled_indices)
            if index < dataset_length:
                actual_index = sampled_indices[index]
                return dataset[actual_index]
            index -= dataset_length
        raise IndexError(f"Index {index} out of range for MixedDataset")



class RoundRobinBatchSampler(Sampler[List[int]]):
    """
    适用于N个子数据集拼接的ConcatDataset的轮询批次采样器。
    - 当只有1个子数据集时，行为等同于标准BatchSampler。
    - 当有多个子数据集时，按轮询顺序从各数据集生成批次。
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        stop_mode: str = "max"
    ):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size应为正整数")
        if stop_mode not in ["max", "min"]:
            raise ValueError("stop_mode应为'max'或'min'")

        self.dataset_sizes = dataset_sizes
        self.num_datasets = len(dataset_sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.stop_mode = stop_mode

        # 计算每个数据集的批次数量
        self.num_batches = []
        for size in dataset_sizes:
            if drop_last:
                nb = size // batch_size
            else:
                nb = math.ceil(size / batch_size)
            self.num_batches.append(nb)

        # 确定总批次数
        if self.num_datasets == 1:
            self.total_batches = self.num_batches[0]
        else:
            if stop_mode == "max":
                self.max_rounds = max(self.num_batches)
            else:
                self.max_rounds = min(self.num_batches)
            self.total_batches = self.max_rounds * self.num_datasets

        # 计算全局索引偏移
        self.offsets = [sum(dataset_sizes[:i]) for i in range(self.num_datasets)]

    def __iter__(self) -> Iterator[List[int]]:
        # 为每个epoch重新生成随机索引
        indices = []
        for size in self.dataset_sizes:
            ds_indices = list(range(size))
            if self.shuffle:
                random.shuffle(ds_indices)
            indices.append(ds_indices)

        # 追踪每个数据集的当前批次
        batch_counters = [0] * self.num_datasets

        for _ in range(self.total_batches):
            # 轮询逻辑：dataset_idx按总批次数循环选择
            dataset_idx = _ % self.num_datasets

            # 若当前数据集已生成足够批次则跳过
            if batch_counters[dataset_idx] >= self.num_batches[dataset_idx]:
                continue

            # 获取当前批次的局部索引
            start = batch_counters[dataset_idx] * self.batch_size
            end = start + self.batch_size
            batch_indices = indices[dataset_idx][start:end]

            # 更新批次计数器
            batch_counters[dataset_idx] += 1

            # 处理不完整批次
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # 转换为全局索引
            global_batch = [self.offsets[dataset_idx] + idx for idx in batch_indices]
            if len(global_batch) > 0 or not self.drop_last:
                yield global_batch

    def __len__(self) -> int:
        return self.total_batches


def to_str(text):
    text_str = str(text)
    if len(text_str) > 1 and text_str[0] in ["'", '"', "#", "$", "~", "+"]:
        text_str = text_str[1:]
    return text_str.strip().lstrip()


def random_index_exclude(n, exclude):
    i = random.randrange(n - 1)
    return i if i < exclude else i + 1


def find_closest_offsets_idx(offset_mapping, start, end):
    start_idx, end_idx = None, None
    min_start_diff, min_end_diff = float('inf'), float('inf')

    # 找到最接近 start 的索引
    for i, (s, e) in enumerate(offset_mapping):
        if abs(s - start) < min_start_diff:
            min_start_diff = abs(s - start)
            start_idx = i
        # 一旦超出 start 的范围就可以停止寻找开始索引
        if s > start:
            break

    # 从 start_idx 开始寻找最接近 end 的索引
    for j in range(start_idx, len(offset_mapping)):
        s, e = offset_mapping[j]
        if abs(e - end) < min_end_diff:
            min_end_diff = abs(e - end)
            end_idx = j
        # 一旦超出 end 的范围就可以停止寻找结束索引
        if e > end:
            break

    # 返回找到的索引范围内的 offset_mapping
    return start_idx, end_idx+1


def response_random_repeat(
    response: List[int],
    min_length: int = 2,
    max_length: int = 6,
) -> Tuple[List[int], List[int], List[int]]:
    assert len(response) > 0

    n = len(response)
    if max_length == 0 or max_length > n:
        max_length = n
    if max_length < min_length or min_length < 1 or n < 3:
        return response, response, []
    else:
        # 确保子序列长度不超过列表长度-1
        subseq_length = random.randint(min_length, min(max_length, n-1))
        start_index = random.randint(0, n - subseq_length - 1)
        end_idx = random.randint(start_index + subseq_length, n)
        prefix_seq = response[:end_idx]
        copied_seq = response[start_index:start_index + subseq_length]
        postfix_seq = response[end_idx:]

    return prefix_seq, copied_seq, postfix_seq


def response_random_extract(
    response: List[int],
    min_length: int = 2,
    max_length: int = 6,
    ensure_middle: bool = False
) -> Tuple[List[int], List[int], List[int]]:
    assert len(response) > 0

    n = len(response)
    if max_length == 0 or max_length > n:
        max_length = n
    if max_length < min_length or n < 3:
        return [], response, []

    # Ensure the extracted sequence is from the middle
    if ensure_middle:
        if n - min_length - 1 <= 1:
            start_index = 1
        else:
            start_index = random.randint(1, n - min_length - 1)
        end_index = min(start_index + random.randint(min_length, max_length), n - 1)
    else:
        subseq_length = random.randint(min_length, min(max_length, n-1))
        start_index = random.randint(0, n - subseq_length)
        end_index = start_index + subseq_length

    prefix_seq = response[:start_index]
    extracted_seq = response[start_index:end_index]
    postfix_seq = response[end_index:]

    return prefix_seq, extracted_seq, postfix_seq


def response_random_uniform_extract(
    n: int,
    min_len: int,
    max_len: int,
    ensure_middle: bool = False,
    p_reverse: float = 0.5,
) -> Tuple[int, int]:
    """
    在长度为 n 的序列中随机抽取一个 [start, end) 片段。
    关键改进：
    1. **参数自洽**：自动交换 `min_len`/`max_len` 的顺序错误。
    2. **窗口裁剪**：在任何情况下都保证 `random.randint(lo, hi)` 的 `lo <= hi`。
    3. **长度先行**：先采样合法长度，再采样起始或终止位置，天然避免无效区间。
    4. **ensure_middle**：保留中间至少 `mid_offset` 个 token 的约束。
    """
    assert n > 0, "n must be positive"
    if min_len > max_len:                           # ① 顺序容错
        min_len, max_len = max_len, min_len

    mid_offset = 1 if ensure_middle else 0
    usable = n - 2 * mid_offset                     # 可真正用于抽取的窗口大小
    if usable <= 0:                                 # 极端：序列太短
        return mid_offset, mid_offset               # 返回空片段以避免错误

    # ② 裁剪长度上下界
    min_len = max(1, min(min_len, usable))
    max_len = max(min_len, min(max_len, usable))

    # ③ 先抽长度，确保后续一定有合法位置
    length = random.randint(min_len, max_len)

    if random.random() > p_reverse:
        # ─── 正向：先抽 start ──────────────────────
        start_min = mid_offset
        start_max = n - length - mid_offset
        # ④ 若 start_max < start_min，用一个退化窗口兜底
        start = random.randint(start_min, max(start_min, start_max))
        end = start + length
    else:
        # ─── 反向：先抽 end ───────────────────────
        end_min = length + mid_offset
        end_max = n - mid_offset
        end = random.randint(min(end_min, end_max), end_max)
        start = end - length

    return start, end


def random_add_mask_token(
    tokens: List[int],
    mask_token: int,
    mask_side: str = "both",
    reserve_tokens_num: int = 5,
    mask_token_minlen: int = 5,
    mask_token_maxlen: int = 10,
) -> List[int]:
    assert len(tokens) > 0
    assert mask_side in ["both", "right", "left"],f"{mask_side=}"

    left_mask_length = random.randint(mask_token_minlen, mask_token_maxlen)
    right_mask_length = random.randint(mask_token_minlen, mask_token_maxlen)
    left_mask = [mask_token] * left_mask_length if not mask_side=="right" else []
    right_mask = [mask_token] * right_mask_length if not mask_side=="left" else []

    return left_mask + tokens[:reserve_tokens_num] + right_mask


def random_sublist_index(
        tokens: List[int],
        mask_token: int,
        min_length: int = 2,
        max_length: int = 5,
) -> Tuple[int, int]:
    assert len(tokens) > 0

    n = len(tokens)
    if max_length == 0 or max_length > n:
        max_length = n
    if max_length < min_length or min_length < 1 or n < 3:
        return 0, 0

    # 确保子序列长度不超过列表长度-1
    subseq_length = random.randint(min_length, min(max_length, n-1))
    rmask_min_length = 2 if (n - subseq_length) > 2 else 0
    start_index = random.randint(1, n - subseq_length - rmask_min_length)
    end_index = start_index + subseq_length

    return start_index, end_index


def find_last_position(lst, sub_lst):
    # Join the list elements to a string for easy substring search
    lst_str = ''.join(map(str, lst))
    sub_lst_str = ''.join(map(str, sub_lst))

    # Find the last occurrence of the sublist string
    last_pos = lst_str.rfind(sub_lst_str)

    # Calculate the position of the last element of the sublist in the original list
    if last_pos != -1:
        last_element_pos = last_pos + len(sub_lst) - 1
    else:
        last_element_pos = -1  # This case should not occur as per the problem statement

    return last_element_pos


def find_longest_repeated_sublist(tokens, min_length=1):
    n = len(tokens)
    dp = [[0] * n for _ in range(n)]
    max_length = 0
    first_start_index = 0

    # Step One: Compute the longest repeated sublist using dynamic programming
    for i in range(n):
        for j in range(i + 1, n):
            if tokens[i] == tokens[j]:
                dp[i][j] = 1 if i == 0 or j == 0 else dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    first_start_index = i - max_length + 1

    # If the longest repeated sublist is shorter than min_length, return empty list
    if max_length < min_length:
        return [], []

    # Extract the sublist from the computed start index and length
    longest_sublist = tokens[first_start_index:first_start_index + max_length]

    # Step Two: Find all positions of the longest repeated sublist
    positions = []
    for i in range(n - max_length + 1):
        if tokens[i:i + max_length] == longest_sublist:
            positions.append((i, i + max_length - 1))

    return longest_sublist, positions


def count_ngrams(text, n):
    ngrams = [tuple(text[i:i+n]) for i in range(len(text) - n + 1)]
    return Counter(ngrams)


def mark_repeats_within_5grams(lst, min_ngram=1, max_ngram=5):
    def count_ngrams(text, n):
        ngrams = [tuple(text[i:i+n]) for i in range(len(text) - n + 1)]
        return ngrams

    mark = [1] * len(lst)
    seen_ngrams = defaultdict(int)

    for n in range(min_ngram, max_ngram+1):
        ngrams = count_ngrams(lst, n)
        for i in range(len(lst) - n + 1):
            ngram = tuple(lst[i:i+n])
            seen_ngrams[ngram] += 1
            if seen_ngrams[ngram] > 1:
                for j in range(i, i + n):
                    if mark[j] == 1:
                        mark[j] = -1

    return mark


def detect_three_part_segments(arr, value=None):
    if not arr:
        return None

    if value is None:
        left_value, right_value = arr[0], arr[-1]
    else:
        left_value = right_value = value

    left_end = next((i for i, x in enumerate(arr) if x != left_value), len(arr))
    right_start = next((i for i, x in enumerate(reversed(arr)) if x != right_value), len(arr))

    if left_end == 0 and right_start == 0:
        return [-1], [-1]

    left_segment = (0, left_end - 1) if left_end > 0 else (-1, -1)
    right_segment = (len(arr) - right_start, len(arr) - 1) if right_start > 0 else (-1, -1)

    return left_segment, right_segment


def split_three_parts(arr, value=None):
    """
    将列表 arr 拆分为三部分 [left, middle, right]：
      - left：从头开始，与 value（或 arr[0]）相同的连续元素段
      - right：从尾部开始，与 value（或 arr[-1]）相同的连续元素段
      - middle：剩余的中间段

    参数：
        arr (list): 待拆分的列表
        value (optional): 要匹配的值；若为 None，则分别使用 arr[0] 与 arr[-1] 作为左右两段的匹配值

    返回：
        tuple of lists: (left_part, middle_part, right_part)
    """
    # 空列表直接返回三空
    if not arr:
        return [], [], []

    # 确定左右两段的匹配值
    if value is None:
        left_val, right_val = arr[0], arr[-1]
    else:
        left_val = right_val = value

    # 找到左段结束位置：第一个不等于 left_val 的索引
    left_end = next((i for i, x in enumerate(arr) if x != left_val), len(arr))
    # 找到右段开始位置：倒序中第一个不等于 right_val 的位置，然后映射回正序
    right_len = next((i for i, x in enumerate(reversed(arr)) if x != right_val), len(arr))
    right_start = len(arr) - right_len

    # 切片
    left_part = arr[:left_end]
    right_part = arr[right_start:]
    # 防止左、右两段重叠导致 middle 反向或越界
    middle_part = arr[left_end:right_start] if left_end <= right_start else []

    return left_part, middle_part, right_part


def balance_two_values(
    lst: List[Any],
    v1: Any,
    v2: Any,
    fill: Any,
    ratio: float = 1.0
) -> List[Any]:
    """
    调整列表中 v1 和 v2 的数量，使得 count(v1) / count(v2) 接近 ratio。
    当比例不匹配时，随机削减过多的一方为 fill。

    参数：
      lst   -- 原始序列（不修改原列表）
      v1    -- 值1，作为比例基准的一方
      v2    -- 值2，作为比例目标的一方
      fill  -- 用于替换多余值的填充值
      ratio -- 期望的 count(v1) / count(v2)，必须为正数

    返回：
      平衡后的列表副本
    """
    if ratio <= 0:
        raise ValueError(f"ratio 必须为正数，当前收到：{ratio}")

    # 找出所有 v1 和 v2 的位置
    positions_v1 = [i for i, x in enumerate(lst) if x == v1]
    positions_v2 = [i for i, x in enumerate(lst) if x == v2]

    count_v1 = len(positions_v1)
    count_v2 = len(positions_v2)

    # 无需处理的情形
    if count_v1 == 0 or count_v2 == 0:
        return lst[:]

    # 当前实际比例
    actual_ratio = count_v1 / count_v2

    # 拷贝原列表以保持原始数据不变
    balanced = lst[:]

    if actual_ratio > ratio:
        # v1 过多，保留最多 floor(ratio * count_v2) 个 v1
        max_v1 = int(ratio * count_v2)
        to_remove = count_v1 - max_v1
        if to_remove > 0:
            for idx in random.sample(positions_v1, to_remove):
                balanced[idx] = fill
    elif actual_ratio < ratio:
        # v2 过多，保留最多 floor(count_v1 / ratio) 个 v2
        max_v2 = int(count_v1 / ratio)
        to_remove = count_v2 - max_v2
        if to_remove > 0:
            for idx in random.sample(positions_v2, to_remove):
                balanced[idx] = fill

    return balanced