import json
import os
import re


def read_json(path):
    """
    读取json文件
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    """
    读取jsonl文件
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(data, path):
    """
    写入数据至json文件
    """
    with open(path, 'w', encoding='utf8') as f_write:
        json.dump(data, f_write, indent=2, ensure_ascii=False)

    print('File path: {}, data size: {}'.format(path, len(data)))


def write_jsonl(data, path):
    """
    写入数据至jsonl文件
    """
    with open(path, 'w', encoding='utf8') as f_write:
        json.dump(data, f_write, indent=2, ensure_ascii=False)

    print('File path: {}, data size: {}'.format(path, len(data)))


def read_dir_file_name(path, suffix='json'):
    """
    读取文件夹下的所有文件名，并返回特定后缀的文件名
    """
    files_names = os.listdir(path)
    new_file_names = []
    for file_name in files_names:
        if file_name.split('.')[-1] == suffix:
            new_file_names.append(file_name)

    return new_file_names


def extract_split_keyword(path: str, special_tokens: list) -> str:
    """
    从文件/目录路径中提取唯一的数据切分关键字。
    """
    _PATTERN = re.compile(rf"(?:^|/)(?P<split>{'|'.join(special_tokens)})(?=/|_|\.|$)")
    matches = [m.group("split") for m in _PATTERN.finditer(path)]
    assert len(matches) == 1, (
        f"Expect exactly 1 split token in path, but got {matches or 'none'}: {path}"
    )
    return matches[0]