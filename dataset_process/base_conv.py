import json
import logging
import random
import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerBase,
)
from tqdm.auto import tqdm

logger = logging.getLogger("conv_data_LLM")

TokenizedText = List[str]
EncodedText = List[int]


def refine_node(node: str) -> str:
    if node == "Megamind":
        node = "MegaMind"

    return node


@dataclass
class Triple:
    subject: Union[str, EncodedText]
    predicate: Union[str, EncodedText]
    object: Union[str, EncodedText]
    sign_subject: int = 0
    sign_predicate: int = 0
    sign_object: int = 0

    def encode(
        self, tokenizer: PreTrainedTokenizerBase, sep_token_id: Optional[int] = None, add_sep: bool = True
    ) -> EncodedText:
        if add_sep:
            sep_token_id = sep_token_id or tokenizer.sep_token_id

        return (
            self._may_encode("subject", tokenizer)
            + ([sep_token_id] if add_sep else [])
            + self._may_encode("predicate", tokenizer)
            + ([sep_token_id] if add_sep else [])
            + self._may_encode("object", tokenizer)
        )

    def _may_encode(self, attr: str, tokenizer: PreTrainedTokenizerBase) -> EncodedText:
        val = getattr(self, attr, None)
        if val is None:
            return []
        elif isinstance(val, str):
            return tokenizer.encode(val, add_special_tokens=False)
        else:
            return val

    @classmethod
    def of(cls, triples, reverse_ent: bool = True) -> Iterable["Triple"]:
        for triple in triples:
            if len(triple) < 3:
                continue

            s, p, o = refine_node(triple[0]), triple[1], refine_node(triple[2])
            if not s or not p or not o:
                continue

            if reverse_ent and p.startswith("~"):
                p = p[1:]
                s, o = o, s

            yield Triple(s, p, o)

    @classmethod
    def of_permissive(cls, triples) -> Iterable["Triple"]:
        for triple in triples:
            # 确保三元组至少有三个元素，不足的用空字符串补齐
            triple = list(triple) + [''] * (3 - len(triple))
            s, p, o = refine_node(triple[0]), triple[1], refine_node(triple[2])

            if p.startswith("~"):
                p = p[1:]
                s, o = o, s

            yield Triple(s, p, o)


@dataclass
class SpecialTokens:
    entity_start: str = ' "'
    entity_end: str = '"'
    relation_start: str = ' '
    relation_end: str = ''
    triple_end: str = '.'

    def __init__(self, tokenizer):
        """
        把字段里保存的“字符串形式 token”全部替换为 tokenizer id，
        以便在后续代码里直接使用整数 id 而非字符串。
        """
        for attr in fields(self):
            token_str = getattr(self, attr.name)
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            setattr(self, attr.name, token_ids)


@dataclass
class Example:
    dlg_id: int
    history: List[str]
    speaker: Union[List[int], str]
    response: str
    kb_triples: List[Triple]
    render_kb: Optional[str]
    nega_kb_triples: List[Triple]
    nega_render_kb: List[str]
    retri_kb_triples: List[Triple]
    retri_render_kb: List[str]
    # New: DailyDialog labels -----------------------------------------------------------
    act_label: Optional[str] = None
    emotion_label: Optional[str] = None

    def __len__(self):
        return (
            sum(len(t) for t in self.history)
            + len(self.response)
            + (len(self.render_kb) if self.render_kb is not None else 0)
        )

    @property
    def json_dict(self) -> Dict[str, Any]:
        out_json = dict(
            dialogue_id=self.dlg_id,
            speaker=self.speaker,
            history=self.history,
            response=self.response,
            knowledge_base={},
        )

        if self.kb_triples:
            out_json["knowledge_base"]["paths"] = [
                [triple.subject, triple.predicate, triple.object] for triple in self.kb_triples
            ]

        if self.render_kb is not None:
            out_json["knowledge_base"]["render"] = self.render_kb

        # expose labels if present ------------------------------------------------------
        if self.act_label is not None:
            out_json["act"] = self.act_label
        if self.emotion_label is not None:
            out_json["emotion"] = self.emotion_label

        return out_json


@dataclass
class BaseDataArguments:
    max_history: int = 3  # Number of previous exchanges to keep in history
    max_seq_length: int = 1024  # Max sequence length (larger samples will be excluded from data)
    mmi: bool = False
    mmi_input_type: str = field(
        default="history",
        metadata={"choices": ["dialogue", "history", "response"]}
    )
    with_eos: bool = True
    reverse_ent: bool = True  # Whether to not reverse entity relationships
    spacy_tokenize: bool = False
    is_debug: bool = False
    is_zero_shot: bool = False
    include_triples: bool = True  # Whether to include triples in input sequences
    include_render: bool = False  # Whether to include render in input sequences
    include_nones: bool = False
    include_prompt: bool = True
    input_kg_from: str = field(
        default='target_kg',
        metadata={"choices": ["target_kg", "retrieved_kg", "posi_nega_kg", 'retri_nega_kg']}
    )
    exclude_kb: bool = False  # Whether to exclude knowledge from input sequences
    exclude_no_triple_seq: bool = True
    exclude_incomplete_triple: bool = False

    # special arguments
    nega_triples_num: int = 0
    retri_triples_num: int = 0
    input_triples_num: int = 0
    extract_triples_num: int = 0


class BaseConvDataset(Dataset):
    GEN_PROMPT = "You are a pirate chatbot who always generate a coherent response using the knowledge."
    MMI_PROMPT = "Given the dialogue history, predict the entity that should be used in the response. Use the entity with the highest probability of the closing double quote ("") generation as the target entity."

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizerBase,
        max_history: int = 3,
        max_seq_length: int = 1024,
        mmi: bool = False,
        mmi_input_type: str = 'history',
        nega_triples_num: int = 0,
        retri_triples_num: int = 0,
        input_triples_num: int = 0,
        extract_triples_num: int = 0,
        with_eos: bool = True,
        reverse_ent: bool = True,
        spacy_tokenize: bool = False,
        special_tokens: list = None,
        chosen_knowledge: list = None,
        is_generation: bool = False,
        is_debug: bool = False,
        is_zero_shot: bool = False,
        include_triples: bool = True,
        include_render: bool = False,
        include_nones: bool = False,
        include_prompt: bool = True,
        input_kg_from: str = 'target_kg',
        exclude_labels: bool = False,
        exclude_kb: bool = False,
        exclude_no_triple_seq: bool = True,
        exclude_incomplete_triple: bool = False,
        retrieved_kg: dict = {},
    ):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens if special_tokens else SpecialTokens(tokenizer)
        self.max_history = max_history
        self.max_seq_length = max_seq_length
        self.reverse_ent = reverse_ent
        self.mmi = mmi
        self.mmi_input_type = mmi_input_type
        self.nega_triples_num = nega_triples_num
        self.retri_triples_num = retri_triples_num
        self.input_triples_num = input_triples_num
        self.extract_triples_num = extract_triples_num
        self.with_eos = with_eos
        self.spacy_tokenize = spacy_tokenize
        self.chosen_knowledge = chosen_knowledge
        self.is_generation = is_generation
        self.is_zero_shot = is_zero_shot
        self.is_debug = is_debug
        self.include_triples = include_triples
        self.include_render = include_render
        self.include_nones = include_nones
        self.include_prompt = include_prompt
        self.input_kg_from = input_kg_from
        self.exclude_labels = exclude_labels
        self.exclude_kb = exclude_kb
        self.exclude_no_triple_seq = exclude_no_triple_seq
        self.exclude_incomplete_triple = exclude_incomplete_triple
        self.retrieved_kg = retrieved_kg

        if self.spacy_tokenize:
            self.nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))
        else:
            self.nlp = None

        data = data[:50] if is_debug else data
        data = self._process_retrieved_data(data)
        assert len(data) > 0, 'data is None!'

        self.data = list(self._flatten(data))
        if self.is_generation and self.chosen_knowledge:
            logger.warning('chosen_knowledge is setting to data. ')
            for index, example in enumerate(self.data):
                chosen_kg = self.chosen_knowledge[index]
                assert type(chosen_kg) in [list, str]
                # print(example)
                example.target_kb_triples = example.kb_triples
                example.target_render_kb = example.render_kb
                example.kb_triples = list(Triple.of_permissive(chosen_kg)) if type(chosen_kg) is list else example.kb_triples
                example.render_kb = chosen_kg if type(chosen_kg) is str else example.render_kb

        if len(self.data) < len(data):
            logger.warning(
                f"{len(data)} -> {len(self.data)} = {len(data) - len(self.data)} examples "
                f"~{100 * (1 - len(self.data) / len(data)):.1f}% reduction"
            )

    def __len__(self) -> int:
        return len(self.data)

    # -----------------------------------------------------------
    # Analysis helper – 建议放在 DialogueFillingDataset 末尾
    # -----------------------------------------------------------
    def describe(
        self,
        field: str = "input_ids",
        percentiles: tuple = (50, 75, 90, 95, 98, 99),
        truncate_len: int = None,
    ) -> dict:
        """
        统计数据集样本长度等信息，返回字典，便于日志记录或 YAML/JSON 配置。
        """
        # ---------- 1. 主循环 ----------
        lengths: list[int] = []
        for idx in tqdm(range(len(self))):
            sample = self[idx]

            if field not in sample:           # 安全兜底
                continue

            tensor = sample[field]
            # 兼容 1-D / 2-D（某些字段可能是 [n_seq, L]）
            if isinstance(tensor[0], (list, tuple)):
                seq_len = max(len(x) for x in tensor) if tensor else 0
            else:
                seq_len = len(tensor)
            lengths.append(seq_len)

        arr = np.asarray(lengths, dtype=np.int64)
        if arr.size == 0:
            return {"error": f"field {field!r} not found in any samples."}

        # ---------- 2. 统计值 ----------
        stats = {
            "count": int(arr.size),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=0)),
            "percentiles": {int(p): int(np.percentile(arr, p)) for p in percentiles},
        }

        # ---------- 3. 截断误差 ----------
        if truncate_len is not None:
            exceed = int((arr > truncate_len).sum())
            stats.update(
                truncate_len=int(truncate_len),
                num_exceed_truncate_len=exceed,
                ratio_exceed=float(exceed) / arr.size,
            )

        return stats

    def _process_retrieved_data(self, data):
        if self.exclude_kb:
            return data

        kg_type = "triple" if "top10_triple" in data[0] else "knowledge"
        for idx, dialogue in enumerate(data):
            dialogue["retri_knowledge_base"] = {}
            dialogue["retri_knowledge_base"]["paths"] = dialogue[f'top10_{kg_type}'][:3]
            dialogue["nega_knowledge_base"]["paths"] = dialogue[f'last10_{kg_type}'][:3]

        return data

    # ---------------------------------------------------------------------------------
    #  _flatten -----------------------------------------------------------------------
    # ---------------------------------------------------------------------------------

    def _flatten(self, data: List[Dict[str, Any]]) -> Iterable[Example]:
        """Drop‑in replacement that delegates specialised logic to helpers."""
        num_no_triple = 0
        num_incomplete_triple = 0

        self.data_idx_excluded = []
        self.data_idx_with_triples = []
        self.data_excluded = []
        data_idx_with_triples = 0

        for idx, dialogue in enumerate(data):
            history = [self._clean(t) for t in dialogue["history"]]
            if self.max_history > 0:
                history = history[-self.max_history:]
            if 'chosen_topic' in dialogue and not history:
                history = [dialogue['chosen_topic']]

            speaker = dialogue.get("speaker", "user")
            response = self._clean(dialogue["response"])

            # --- extract knowledge ---------------------------------------------------
            (
                kb_triples,
                render_kb,
                nega_kb_triples,
                nega_render_kb,
                retri_kb_triples,
                retri_render_kb,
                no_triple_flag,
                incomplete_triple_flag,
            ) = self._process_knowledge(dialogue)

            # counters for stats
            if no_triple_flag:
                num_no_triple += 1
            if incomplete_triple_flag:
                num_incomplete_triple += 1

            # possible skipping -------------------------------------------------------
            if no_triple_flag and self.exclude_no_triple_seq:
                self.data_idx_excluded.append(idx)
                continue

            # --- extract labels ------------------------------------------------------
            act_label, emotion_label = self._process_labels(dialogue)

            dialogue_id = dialogue["dialogue_id"]

            # ------- token/length sanity as before -----------------------------------
            if self.max_seq_length > 0:
                encoded_len = self.max_seq_length
                response_len = len(self.tokenizer.tokenize(' ' + response))
                while encoded_len >= self.max_seq_length:
                    if not history:
                        break
                    encoded_len = (
                        len(self.tokenizer.tokenize(" ".join(render_kb or "")))
                        + len(self.tokenizer.tokenize(" ".join(history)))
                        + response_len + 3
                    )
                    if encoded_len >= self.max_seq_length:
                        history.pop(0)
                if not history or response_len > 200:
                    self.data_idx_excluded.append(idx)
                    self.data_excluded.append(dialogue)
                    logger.warning(f"data {idx} is deleted! because of the {response_len=} or {len(history)=}")
                    continue

            if kb_triples or render_kb:
                self.data_idx_with_triples.append(data_idx_with_triples)

            data_idx_with_triples += 1

            yield Example(
                dlg_id=dialogue_id,
                history=history,
                speaker=speaker,
                response=response,
                kb_triples=kb_triples,
                render_kb=render_kb,
                nega_kb_triples=nega_kb_triples,
                nega_render_kb=nega_render_kb,
                retri_kb_triples=retri_kb_triples,
                retri_render_kb=retri_render_kb,
                act_label=act_label,
                emotion_label=emotion_label,
            )

        # --- logging -----------------------------------------------------------------
        num_with_triple = len(data) - num_no_triple - num_incomplete_triple
        logger.info(
            "#examples with triples {} ({:.1f}%) out of {} examples".format(
                num_with_triple, 100 * num_with_triple / len(data), len(data)
            )
        )

        logger.info(
            "#examples with no triples {} ({:.1f}%) and incomplete triples {} ({:.1f}%)".format(
                num_no_triple,
                100 * num_no_triple / len(data),
                num_incomplete_triple,
                100 * num_incomplete_triple / len(data),
            )
        )

    def _process_labels(self, dialogue: Dict[str, Any]):
        """Extract act / emotion labels (DailyDialog) if present."""
        act_label = dialogue.get("act", None)
        emotion_label = dialogue.get("emotion", None)
        return act_label, emotion_label

    def _process_knowledge(self, dialogue: Dict[str, Any]):
        """All knowledge-handling logic pulled out of _flatten for clarity."""
        kb_triples: List[Triple] = []
        render_kb: Optional[str] = None
        nega_kb_triples: List[Triple] = []
        nega_render_kb: List[str] = []
        retri_kb_triples: List[Triple] = []
        retri_render_kb: List[str] = []

        no_triple_flag = False
        incomplete_triple_flag = False

        # ---- target / gold KB -------------------------------------------------------
        if dialogue.get("knowledge_base"):
            kb = dialogue["knowledge_base"]
            if kb.get("paths"):
                kb_triples = list(Triple.of(kb["paths"], reverse_ent=self.reverse_ent))
                if len(kb_triples) < len(kb["paths"]):
                    incomplete_triple_flag = True
                    kb_triples = list(Triple.of_permissive(kb["paths"]))
            render_kb = self._clean(kb.get("render")) if kb.get("render") else None
        else:
            no_triple_flag = True

        # ---- negative KB ------------------------------------------------------------
        if "nega_knowledge_base" in dialogue:
            nkb = dialogue["nega_knowledge_base"]
            if "paths" in nkb:
                nega_kb_triples = list(Triple.of_permissive(nkb["paths"]))
            if "render" in nkb:
                nega_render_kb = nkb.get("render", [])

        # ---- retrieved KB -----------------------------------------------------------
        if "retri_knowledge_base" in dialogue:
            rkb = dialogue["retri_knowledge_base"]
            if "paths" in rkb:
                retri_kb_triples = list(Triple.of_permissive(rkb["paths"]))
            if "render" in rkb:
                retri_render_kb = rkb.get("render", [])

        return (
            kb_triples,
            render_kb,
            nega_kb_triples,
            nega_render_kb,
            retri_kb_triples,
            retri_render_kb,
            no_triple_flag,
            incomplete_triple_flag,
        )


    def _clean(self, text: str):
        # To resolve issues like 'I tried cherry shrimp last week, and couldn't stand it.                      '
        text = re.sub(r'\s+', ' ', text)
        return text

    def _encode(self, text: str,
                pre_encoded_seq: Dict = None,
                pre_encoded_signs: Dict = None,
                pre_encoded_labels: Dict = None,
                return_signs: bool = False,
                return_labels: bool = False
                ):
        if not pre_encoded_seq or len(text) < 2:
            return self.tokenizer.encode(text, add_special_tokens=False)

        if return_signs:
            assert pre_encoded_signs, 'pre_encoded_signs is None but return_signs'

        if return_labels:
            assert pre_encoded_labels, 'pre_encoded_labels is None but return_labels'

        # Step 1: Find all positions of sequences in pre_encoded_seq
        seq_positions = []
        for seq in pre_encoded_seq:
            start = 0
            while start < len(text):
                start = text.find(seq, start)
                if start == -1:
                    break
                signs = pre_encoded_signs[seq] if return_signs else []
                labels = pre_encoded_labels[seq] if return_labels else []
                seq_positions.append((start, start + len(seq), pre_encoded_seq[seq], signs, labels))
                start += len(seq)

        # Step 2: Sort positions by starting index
        seq_positions.sort(key=lambda x: x[0])

        # Step 3: Encode the text
        encoded_result = []
        signs_result, labels_result = [], []
        last_index = 0

        for start, end, encoded_seq, signs, labels in seq_positions:
            if start > last_index:
                # Encode the text between the last matched sequence and the current one
                encoded_text = self.tokenizer.encode(text[last_index:start], add_special_tokens=False)
                encoded_result.extend(encoded_text)
                signs_result.extend([0]*len(encoded_text))
                labels_result.extend([-100]*len(encoded_text))
            # Add the pre-encoded sequence
            encoded_result.extend(encoded_seq)
            signs_result.extend(signs)
            labels_result.extend(labels)
            last_index = end

        # Encode any remaining text after the last matched sequence
        if last_index < len(text):
            encoded_text = self.tokenizer.encode(text[last_index:], add_special_tokens=False)
            encoded_result.extend(encoded_text)
            signs_result.extend([0]*len(encoded_text))
            labels_result.extend([-100]*len(encoded_text))

        if return_signs and return_labels:
            return (encoded_result, signs_result, labels_result)
        elif return_signs:
            return (encoded_result, signs_result)
        elif return_labels:
            return (encoded_result, labels_result)
        else:
            return encoded_result

    def _word_tokenize(self, text: str, *, as_string: bool = True) -> Union[str, List[str]]:
        assert self.spacy_tokenize

        # To resolve issues like 'He also directed Batman R.I.P.. Have you seen that one?'
        text = re.sub(r"(\w+\.)\.\s", r"\1 . ", text)
        text = re.sub(r"(\.\w\.)\.\s", r"\1 . ", text)

        # To resolve issues like 'I like Neil Brown Jr..' and 'I think he plays for Real Madrid C.F..'
        if re.match(r".*\w+\.\.$", text) or re.match(r".*\.\w\.\.$", text):
            text = text[:-1] + " ."

        tokens = [tok.text for tok in self.nlp(text)]
        if as_string:
            return " ".join(tokens)
        else:
            return tokens

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.data[index]

        if self.spacy_tokenize:
            response = self._word_tokenize(" " + example.response)
            history = [self._word_tokenize(u) for u in example.history]
        else:
            response = example.response
            history = example.history

        history = [self._encode(h) for h in history]
        response = self._encode(response)

        nega_kb_triples, retri_kb_triples, target_triples = [], [], []
        nega_kb_render,  retri_kb_render,  target_render  = [], [], []

        if self.input_kg_from in ['target_kg', 'posi_nega_kg']:
            target_triples = example.kb_triples
            target_render = [example.render_kb]
        if self.input_kg_from in ['retrieved_kg', 'retri_nega_kg']:
            assert self.retri_triples_num > 0, f'input_kg_from=retrieved_kg or retri_nega_kg but {self.retri_triples_num=}!'
            retri_kb_triples = example.retri_kb_triples[:self.retri_triples_num]
            retri_kb_render = example.retri_render_kb[:self.retri_triples_num]
        if self.input_kg_from in ['posi_nega_kg', 'retri_nega_kg']:
            assert self.nega_triples_num > 0, f'input_kg_from=posi_nega_kg but {self.nega_triples_num=}!'
            nega_kb_triples = example.nega_kb_triples[:self.nega_triples_num]
            nega_kb_render = example.nega_render_kb[:self.nega_triples_num]

        triples = []
        if self.include_triples:
            for sign, kb_triples in enumerate([nega_kb_triples, [], retri_kb_triples, target_triples]):
                for t_idx, triple in enumerate(kb_triples):
                    if sign != 3 and triple in example.kb_triples:
                        sign_subject = sign_predicate = sign_object = 2
                    else:
                        sign_subject = sign_predicate = sign_object = sign - 1
                    encoded_triple = (
                        self.tokenizer.encode(triple.subject, add_special_tokens=False),
                        self.tokenizer.encode(triple.predicate, add_special_tokens=False),
                        self.tokenizer.encode(triple.object, add_special_tokens=False),
                        sign_subject,
                        sign_predicate,
                        sign_object,
                    )
                    triples.append(Triple(*encoded_triple))

            if nega_kb_triples:
                random.shuffle(triples)

            triples = triples[:(self.extract_triples_num if self.mmi else self.input_triples_num)]

        render = []
        if (self.include_render and target_render is not None):
            for sign, kb_renders in enumerate([nega_kb_render, [], retri_kb_render, target_render]):
                if kb_renders:
                    for kb_render in kb_renders:
                        render_sign = 2 if sign != 3 and kb_render == example.render_kb else sign - 1
                        render_index = random.randint(0, len(render))
                        render.insert(render_index, (self.tokenizer.encode(kb_render, add_special_tokens=False), render_sign))

            render = render[:(self.extract_triples_num if self.mmi else self.input_triples_num)]

        process_method = self._build_triple_from_dialogue if self.mmi else self._build_from_segments
        item_dict = process_method(
            index,
            triples,
            render,
            history,
            response,
        )

        return item_dict

    def _build_from_segments(
        self,
        index: int,
        kb_triples: List[List[str]],
        render: Optional[List[str]],
        history: List[List[str]],
        response: List[str],
    ) -> Dict[str, List[int]]:
        """ Builds a sequence of input from 3 segments: history, kb triples and response. """
        token_triples, signs_triples = self._triples_process(index, kb_triples)
        token_render, signs_render = self._render_process(index, render)
        token_history_lst, signs_history_lst = self._history_process(index, history)
        token_response, signs_response = self._response_process(index, response)

        speaker = 'user' if len(token_history_lst) % 2 == 1 else 'assistant'
        messages = [{"role": "system", "content": self.GEN_PROMPT}] if self.include_prompt else []

        pre_encoded_tokens, pre_encoded_signs = {}, {}
        if token_triples+token_render and not self.exclude_kb:
            messages.append({"role": "knowledge", "content": f"<|KB|>{str(token_triples+token_render)}<|KB|>"})
            pre_encoded_tokens[f"<|KB|>{str(token_triples+token_render)}<|KB|>"] = token_triples+token_render
            pre_encoded_signs[f"<|KB|>{str(token_triples+token_render)}<|KB|>"] = signs_triples+signs_render

        if token_history_lst:
            for i, (his, his_signs) in enumerate(zip(token_history_lst, signs_history_lst)):
                messages.append({"role": speaker, "content": f"<|HIS{i}|>{str(his)}<|HIS{i}|>"})
                pre_encoded_tokens[f"<|HIS{i}|>{str(his)}<|HIS{i}|>"] = his
                pre_encoded_signs[f"<|HIS{i}|>{str(his)}<|HIS{i}|>"] = his_signs
                speaker = self._other_speaker(speaker)

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_input, signs_input = self._encode(
            text=text_input,
            pre_encoded_seq=pre_encoded_tokens,
            pre_encoded_signs=pre_encoded_signs,
            return_signs=True,
        )

        postfix = [self.tokenizer.eos_token_id] if self.with_eos else []
        token_label = []
        if not self.is_generation:
            token_label = [-100]*len(token_input) + token_response + postfix
            token_input += token_response + postfix
            signs_input += signs_response + [1]

        return self.model_args(dict(
            input_ids=token_input,
            attention_mask=[1] * len(token_input),
            labels=token_label,
            signs=signs_input,
        ))

    def _build_triple_from_dialogue(
        self,
        index: int,
        kb_triples: List[Triple],
        render: Optional[List[int]],
        history: List[List[int]],
        response: List[int],
    ) -> Dict[str, List[int]]:
        """ history + response -> triple"""
        token_triples, signs_triples = self._triples_process(index, kb_triples, mmi=True)
        token_render, signs_render = self._render_process(index, render, mmi=True)
        token_history_lst, signs_history_lst = self._history_process(index, history, mmi=True)
        token_response, signs_response = self._response_process(index, response, mmi=True)

        token_history_lst = token_history_lst if self.mmi_input_type in ['dialogue', 'history'] else [[]]
        signs_history_lst = signs_history_lst if token_history_lst else [[]]
        token_response = token_response if self.mmi_input_type in ['dialogue', 'response'] else []
        signs_response = signs_response if token_response else [[]]

        speaker = 'user' if len(token_history_lst) % 2 == 1 else 'assistant'
        messages = [{"role": "system", "content": self.MMI_PROMPT}] if self.include_prompt else []

        pre_encoded_tokens, pre_encoded_signs = {}, {}
        if token_history_lst:
            for i, (his, his_signs) in enumerate(zip(token_history_lst, signs_history_lst)):
                messages.append({"role": speaker, "content": f"<|HIS{i}|>{str(his)}<|HIS{i}|>"})
                pre_encoded_tokens[f"<|HIS{i}|>{str(his)}<|HIS{i}|>"] = his
                pre_encoded_signs[f"<|HIS{i}|>{str(his)}<|HIS{i}|>"] = his_signs
                speaker = self._other_speaker(speaker)

        if token_response:
            messages.append({"role": "assistant", "content": f"<|RES|>{str(token_response)}<|RES|>"})
            pre_encoded_tokens[f"<|RES|>{str(token_response)}<|RES|>"] = token_response
            pre_encoded_signs[f"<|RES|>{str(token_response)}<|RES|>"] = signs_response

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_input, signs_input = self._encode(
            text=text_input,
            pre_encoded_seq=pre_encoded_tokens,
            pre_encoded_signs=pre_encoded_signs,
            return_signs=True,
        )

        postfix = [self.tokenizer.eos_token_id] if self.with_eos else []
        token_label = []
        if not self.is_generation:
            token_label = [-100]*len(token_input) + token_triples + token_render + postfix
            token_input += token_triples + token_render + postfix
            signs_input += signs_triples + signs_render + [1]*len(postfix)

        return self.model_args(dict(
            input_ids=token_input,
            attention_mask=[1] * len(token_input),
            labels=token_label,
            signs=signs_input,
        ))

    def _triples_process(self, index, kb_triples, mmi: bool = False):
        token_triples = []
        signs_triples = []
        for triple in kb_triples:
            token_triples.extend(
                self.special_tokens.entity_start
                + triple.subject
                + self.special_tokens.entity_end
                + self.special_tokens.relation_start
                + triple.predicate
                + self.special_tokens.relation_end
                + self.special_tokens.entity_start
                + triple.object
                + self.special_tokens.entity_end
                + self.special_tokens.triple_end
            )
            if mmi:
                signs_triples.extend(
                    [0]*len(self.special_tokens.entity_start)
                    + [0]*len(triple.subject)
                    + [triple.sign_subject]*len(self.special_tokens.entity_end)
                    + [0]*len(self.special_tokens.relation_start)
                    + [0]*len(triple.predicate)
                    + [triple.sign_predicate]*len(self.special_tokens.relation_end)
                    + [0]*len(self.special_tokens.entity_start)
                    + [0]*len(triple.object)
                    + [triple.sign_object]*len(self.special_tokens.entity_end)
                    + [0]*len(self.special_tokens.triple_end)
                )
        if not mmi:
            signs_triples = [0]*len(token_triples)

        return token_triples, signs_triples

    def _render_process(self, index, render, mmi=False):
        token_render = [item[0]+self.special_tokens.entity_end for item in render]
        token_render = [item for sub_lst in token_render for item in sub_lst]
        signs_render = [[0]*len(item[0])+[item[1] if mmi else 0] for item in render]
        signs_render = [item for sub_lst in signs_render for item in sub_lst]

        return token_render, signs_render

    def _history_process(self, index, history, mmi=False):
        token_history_lst = history
        signs_history_lst = [[0]*len(h) for h in history]

        return token_history_lst, signs_history_lst

    def _response_process(self, index, response, mmi=False):
        token_response = response
        return token_response, [0 if mmi else 1]*len(token_response)

    def model_args(self, batch):
        return {
            k: v
            for k, v in batch.items()
            if (self.include_nones or v is not None)
            and (not self.exclude_labels or k != "labels")
            and k not in ("triple_ids")
        }

    def _other_speaker(self, speaker: str):
        return 'assistant' if speaker == 'user' else 'user'


@dataclass
class BaseCollator:
    pad: int
    kg_pad: Optional[int] = None
    mask_pad: Optional[int] = 0
    label_pad: Optional[int] = -100
    as_tuple: bool = False
    fields: Iterable[str] = field(default_factory=list)
    padding_side: Optional[str] = 'right'
    pad_to_multiple_of: Optional[int] = None

    @property
    def special_pad_tokens(self) -> Dict[str, int]:
        raise NotImplementedError("This method should be implemented by subclasses")

    def _get_pad_token(self, field: str) -> int:
        pad = self.special_pad_tokens.get(field, None)
        return self.pad if pad is None else pad

    def __call__(self, batch):
        bsz = len(batch)
        padded_batch = {}
        for name in self.fields:
            if any(name not in x for x in batch):
                continue

            max_l = max(len(x[name]) for x in batch)
            if self.pad_to_multiple_of:
                max_l = int(self.pad_to_multiple_of * np.ceil(max_l / self.pad_to_multiple_of))

            pad_token = self._get_pad_token(name)

            padded_field = np.full((bsz, max_l), pad_token, dtype=np.int64)

            for bidx, x in enumerate(batch):
                seq_length = len(x[name])
                if seq_length <= 0:
                    continue
                if self.padding_side == 'right':
                    padded_field[bidx, :seq_length] = x[name]
                elif self.padding_side == 'left':
                    padded_field[bidx, -seq_length:] = x[name]
                else:
                    raise ValueError("Invalid padding_side. Choose either 'left' or 'right'.")

            padded_batch[name] = torch.from_numpy(padded_field)

        if self.as_tuple:
            return tuple(padded_batch.get(f, None) for f in self.fields)
        else:
            return padded_batch


@dataclass
class Collator(BaseCollator):
    fields: Iterable[str] = (
        "input_ids",
        "attention_mask",
        "labels",
        "signs",
    )
    sign_pad: Optional[int] = 0
    padding_side: Optional[str] = 'right'
    pad_to_multiple_of: Optional[int] = None

    @property
    def special_pad_tokens(self) -> Dict[str, int]:
        return dict(
            attention_mask=self.mask_pad,
            labels=self.label_pad,
            signs=self.sign_pad,
        )