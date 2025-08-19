import logging
import random
import json
import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Iterable, Sequence, Tuple

import spacy
import torch
import numpy as np

from ..load_data.conv_LLM import (
    Triple,
    TokenizedText,
    BaseConvDataset,
)

from .utils import (
    balance_two_values,
    find_closest_offsets_idx,
    response_random_uniform_extract,
    split_three_parts,
    random_index_exclude,
)

logger = logging.getLogger("dialogue_copy_data")


class Printer:
    def __init__(self):
        self.has_printed = False
        self.has_one_printed = False

    def print_once(self, message):
        if not self.has_printed:
            print(message+'\n')
            self.has_printed = True

    def print_twice(self, message):
        if not self.has_printed:
            print(message+'\n')
            self.has_printed = True
        if not self.has_one_printed:
            print(message+'\n')
            self.has_one_printed = True


printer0 = Printer()
printer1 = Printer()
printer2 = Printer()
printer3 = Printer()
printer4 = Printer()


@dataclass
class ExtendedSpecialTokens:
    mask: str = ' mask'
    bad:  str = '[BAD]'
    good: str = '[GOOD]'
    kg_bad:  str = '[KBAD]'
    kg_good: str = '[KGOOD]'

    entity_start: str = ' "'
    entity_end: str = '"'
    relation_start: str = ' '
    relation_end: str = ''
    triple_start: str = '['
    triple_end: str = ']'

    def __init__(self, tokenizer: Any):
        extra_token_strs = [self.bad, self.good, self.kg_bad, self.kg_good]

        # 过滤掉已存在的 token，避免重复 id
        vocab = tokenizer.get_vocab()
        tokens_to_add = [t for t in extra_token_strs if t not in vocab]

        if tokens_to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

        for attr in fields(self):
            token_str = getattr(self, attr.name)
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            setattr(self, attr.name, token_ids)


@dataclass
class GenerationState:
    mode: Optional[str] = field(
        default='response_filling',
        metadata={"choices": [
            'response_filling',
            'keyword_extract',
            'keyword_generate',
            'mask_pre_generate',
            'mask_post_generate',
            'mask_post_valid',
            'mask_pre_valid',
            'mask_random_append',
            'retrieved_keywords_valid',
            "retrieved_knowledges_valid",
            "retrieved_kwd_and_kg_valid",
            "retrieved_kwd_as_kg_valid",
        ]})
    keywords: List[List[int]]             = field(default_factory=list)
    keywords_gens: List[List[int]]        = field(default_factory=list)
    keywords_cands: List[Any]             = field(default_factory=list)
    keywords_scores: List[float]          = field(default_factory=list)
    keywords_is_good: List[bool]          = field(default_factory=list)
    knowledges: List[List[int]]           = field(default_factory=list)
    knowledges_cands: List[Any]           = field(default_factory=list)
    knowledges_scores: List[float]        = field(default_factory=list)
    knowledges_is_good: List[bool]        = field(default_factory=list)
    masked_pre: List[int]                 = field(default_factory=list)
    masked_post: List[int]                = field(default_factory=list)
    masked_pre_scores: List[List[float]]  = field(default_factory=list)
    masked_post_scores: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        choices = self.__dataclass_fields__['mode'].metadata['choices']
        if self.mode not in choices:
            raise ValueError(f"Invalid mode {self.mode!r}, expected one of {choices}")

    def print_state(self):
        info = []
        for f in fields(self):
            name = f.name
            if name == 'mode':
                continue
            value = getattr(self, name)
            if isinstance(value, list):
                info.append(f"{name}={len(value)}")
        # 组合并打印一行
        line = f"mode={self.mode} | " + ", ".join(info)
        logger.info(line)

    def get_non_empty_list_names(self) -> List[str]:
        return [
            f.name
            for f in fields(self)
            if f.name != 'mode'
            and isinstance(getattr(self, f.name), list)
            and len(getattr(self, f.name)) > 0
        ]


MMI_TASKS = ['NegaKW', 'NegaKG', 'NegaKWRS', 'NegaKWKG']


@dataclass
class AugmentDataArguments:
    unmask_min_length: int = 2
    unmask_max_length: int = 6
    add_mask_min_length: int = 5
    add_mask_max_length: int = 10
    bad_tok_ratio: float = 2.0
    keywords_valid_ppl_max: float = 5.0
    type_matter: bool = True
    ratio: float = 1.0
    is_mask_learning: bool = False
    use_block_diag_kw: bool = False
    mmi_nokg_learning: bool = False
    only_cls_learning: bool = False
    extract_kg_from: str = field(
        default='target_kg',
        metadata={"choices": ["context", "target_kg", "retrieved_kg", 'generated_kg']}
    )
    output_type: str = field(
        default="Response",
        metadata={"choices": ["Filling", "Response"]}
    )
    robust_type: str = field(
        default=None,
        metadata={"choices": [None, "Mask", "Unmask", "Fuse"]}
    )
    robust_value: int = 8
    mmi_type: str = field(
        default="Class",
        metadata={"choices": ["Triple", "Class", "Fuse"]}
    )
    learning_type: str = field(
        default=None,
        metadata={"choices": [None, "DirectCL", "IndirectCL"]}
    )
    ablation_type: str = field(
        default=None,
        metadata={"choices": [None, 'w/o_kw', 'w/o_mr', 'w/o_mk', 'w/o_df']}
    )
    mask_type: str = field(
        default='muti',
        metadata={"choices": ['muti', 'one', 'zero']}
    )


class DialogueFillingDataset(BaseConvDataset):
    FIL_PROMPT = "This is a dialogue completion task. Replace every mask to one word in the masked response to generate a complete and coherent response using the knowledge. "
    KWD_PROMPT = "This is a keyword generation task. Based on the provided text, generate a single keyword that will be used in the subsequent response. Ensure that this keyword is relevant to the text and can be naturally incorporated into the reply."
    MSK_PROMPT = "This is a masked sentence generation task. Using the keyword of dialogue, compose a coherent sentence that includes this keyword and incorporates an appropriate number of masks. These masks will be filled in later to complete the response."
    MMI_PROMPT = "This is a knowledge completion task. Replace every mask to one word in the masked knowledge to generate a complete and coherent knowledge. "

    def __init__(
        self,
        ablation_type: str = None,
        add_mask_min_length: int = 5,
        add_mask_max_length: int = 10,
        bad_tok_ratio: float = 0.5,
        ratio: float = 1.0,
        train_task_type: str = None,
        output_type: str = "Filling",
        type_matter: bool = True,
        robust_type: str = None,
        robust_value: int = 8,
        mmi_type: str = None,
        mask_type: str = 'muti',
        muti_output: bool = False,
        learning_type: str = None,
        extract_kg_from: str = 'knowledge',
        generation_state: GenerationState = None,
        is_mask_learning: bool = False,
        keywords_valid_ppl: list = None,
        keywords_valid_ppl_max: float = 5.0,
        unmask_min_length: int = 2,
        unmask_max_length: int = 6,
        use_block_diag_kw: bool = False,
        mmi_nokg_learning: bool = False,
        only_cls_learning: bool = False,
        **hparams
    ):
        self.unmask_min_length = unmask_min_length
        self.unmask_max_length = unmask_max_length
        self.add_mask_min_length = add_mask_min_length
        self.add_mask_max_length = add_mask_max_length
        self.type_matter = type_matter  # hyperparameter of task EntMix and SpanMix
        self.ratio = ratio  # hyperparameter of task EntMix and SpanMix
        self.train_task_type = train_task_type
        self.output_type = output_type
        self.robust_type = robust_type
        self.robust_value = robust_value
        self.mmi_type = mmi_type
        self.mask_type = mask_type
        self.muti_output = False
        self.learning_type = learning_type
        self.ablation_type = ablation_type
        self.extract_kg_from = extract_kg_from
        self.generation_state = generation_state or GenerationState()
        self.keywords_valid_ppl = keywords_valid_ppl
        self.keywords_valid_ppl_max = keywords_valid_ppl_max
        self.is_mask_learning = is_mask_learning
        self.bad_tok_ratio = bad_tok_ratio
        self.use_block_diag_kw = use_block_diag_kw
        self.mmi_nokg_learning = mmi_nokg_learning
        self.only_cls_learning = only_cls_learning

        assert type(self.train_task_type) == str
        if self.train_task_type == "LM":
            self.ori_generate = False
            super().__init__(**hparams)
            return

        is_mmi_task  = train_task_type in MMI_TASKS
        is_valid_mode = self.generation_state.mode in [
            "retrieved_keywords_valid",
            "retrieved_knowledges_valid",
            "retrieved_kwd_and_kg_valid",
            "retrieved_kwd_as_kg_valid"
        ]
        if is_mmi_task or is_valid_mode:
            hparams['mmi'] = True
            hparams['mmi_input_type'] = 'history'
        if is_mmi_task:
            hparams['input_kg_from'] = 'retri_nega_kg'
        if is_valid_mode and hparams['retri_triples_num'] < hparams['extract_triples_num']:
            hparams['retri_triples_num'] = hparams['extract_triples_num']

        super().__init__(**hparams)

        self.ner = spacy.load("en_core_web_sm")
        self.ori_generate = False
        if self.is_generation and self.generation_state.mode == 'lm':
            self.ori_generate = True

        if not self.is_generation:
            if 'Span' in train_task_type or 'Fuse' in train_task_type:
                with open(hparams["dataset_path"].replace("_nega_triples.jsonl", "_type_spans.json"), "r", encoding="utf-8") as f:
                    self.type_spans = json.load(f)
                with open(hparams["dataset_path"].replace("_nega_triples.jsonl", "_dial_spans.json"), "r", encoding="utf-8") as f:
                    self.dial_spans = []
                    for idx, line in enumerate(f):
                        if idx not in self.exclude_idx:
                            self.dial_spans.append(json.loads(line))
                    assert len(self.dial_spans) == len(self.data)

        if self.is_zero_shot and 'keywords' in self.generation_state.mode:
            self.prompt = self.KWD_PROMPT
        elif self.is_zero_shot and self.generation_state.mode in ['pre_masked', 'post_masked']:
            self.prompt = self.MSK_PROMPT
        elif self.mmi:
            self.prompt = self.MMI_PROMPT
        else:
            self.prompt = self.FIL_PROMPT

    def _good_bad_ids(self, Type='mask') -> Tuple[int, int]:
        """Return (good_id, bad_id) shorthand as a cached property."""
        if Type == 'mask':
            good_id = self.special_tokens.good[0]
            bad_id = self.special_tokens.bad[0]
        elif Type == 'kg':
            good_id = self.special_tokens.kg_good[0]
            bad_id = self.special_tokens.kg_bad[0]
        else:
            raise ValueError(f'_good_bad_ids {Type=} is wrong!')
        return good_id, bad_id

    # ================================
    # 类别 2: _build_from_segments 方法替换
    # ================================

    def _build_from_segments(
        self, index, kb_triples, render, history, response
    ):
        # 基础编码
        if self.ori_generate or self.train_task_type == "LM":
            return super()._build_from_segments(index, kb_triples, render, history, response)

        t_triples, s_triples = self._triples_process(index, kb_triples)
        t_render, s_render = self._render_process(index, render)
        h_list, s_list = self._history_process(index, history)
        t_target, t_masked, t_unmask = self._response_process(index, response)

        if self.is_generation and not t_target:
            ori_sign = self.ori_generate
            self.ori_generate = True
            ori_output = super()._build_from_segments(index, kb_triples, render, history, response)
            self.ori_generate = ori_sign
            return ori_output

        # 构建上下文并编码一次
        messages, pre_toks, pre_sgs = self._prepare_context(
            t_triples, s_triples, t_render, s_render, h_list, s_list
        )
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs, signs = self._encode(
            chat_text, pre_toks, pre_sgs, return_signs=True
        )

        outputs, outputs_lbs, outputs_sgs = self._outputs_process(
            index=index,
            token_target=t_target,
            token_masked=t_masked,
            token_unmask=t_unmask
        )
        input_ids = inputs + outputs
        label_ids = [-100] * len(inputs) + outputs_lbs
        signs_all = signs + outputs_sgs
        return self.model_args({
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': label_ids,
            'signs': signs_all,
            'token_target': t_target,
            'token_masked': t_masked,
            'token_unmask': t_unmask,
        })

    def _prepare_context(
        self, t_triples, s_triples, t_render, s_render, h_list, s_list
    ):
        """构建 messages、pre_encoded_tokens 和 pre_encoded_signs"""
        messages = [{"role": "system", "content": self.prompt}] if self.include_prompt else []
        messages = [{"role": "system", "content": self.prompt}] if self.include_prompt else []
        pre_toks, pre_sgs = {}, {}
        # 加入 KB
        if t_triples + t_render and not self.exclude_kb:
            kb_tag = f"<|KB|>{str(t_triples + t_render)}<|KB|>"
            messages.append({"role": "knowledge", "content": kb_tag})
            pre_toks[kb_tag] = t_triples + t_render
            pre_sgs[kb_tag] = s_triples + s_render
        # 加入历史
        speaker = 'user' if len(h_list) % 2 == 1 else 'assistant'
        for i, (his, his_sgs) in enumerate(zip(h_list, s_list)):
            his_tag = f"<|HIS{i}|>{str(his)}<|HIS{i}|>"
            messages.append({"role": speaker, "content": his_tag})
            pre_toks[his_tag] = his
            pre_sgs[his_tag] = his_sgs
            speaker = self._other_speaker(speaker)
        return messages, pre_toks, pre_sgs


    # ================================
    # 类别 2: _outputs_process 方法
    # ================================

    def _outputs_process(  # type: ignore[override]
        self,
        index: int,
        token_target: List[int],
        token_masked: List[int],
        token_unmask: List[int],
        mmi: bool = False,
    ) -> Dict[str, List[int]]:
        """Build labels for model response in a modular fashion.

        Parameters
        ----------
        index: int
            Sample index in the current batch.
        token_target / token_masked / token_unmask: List[int]
            Pre‑tokenised IDs corresponding to keywords, masked content and
            final response, respectively.
        mmi: bool, optional
            Whether we are constructing labels for mutual‑information (knowledge)
            training instead of plain response generation.
        """
        output_type = "knowledge" if mmi else "response"
        eos = self.tokenizer.eos_token_id
        sign_init = 0 if self.is_generation and "mask" in self.generation_state.mode else 1

        # -------------------------------------------------------------------
        # Branch 1 ‑ keyword‑only generation modes
        # -------------------------------------------------------------------
        if self.is_generation and "keywords" in self.generation_state.mode:
            return self._process_keywords(output_type, token_target)

        # -------------------------------------------------------------------
        # Common keyword section used by *all* subsequent branches
        # -------------------------------------------------------------------
        labels_text, pre_tokens, pre_signs, pre_labels = self._build_keyword_section(
            token_target, output_type, eos, sign_init
        )
        # -------------------------------------------------------------------
        # Branch 2 ‑ mask‑based generation variants
        # -------------------------------------------------------------------
        if self.is_generation and self.generation_state.mode in {
            "mask_pre_generate", "mask_pre_valid",
            "mask_post_generate", "mask_post_valid",
            "response_filling"
        }:
            return self._process_masked_generation(
                index,
                labels_text,
                pre_tokens,
                pre_signs,
                pre_labels,
                token_target,
                token_masked,
                output_type,
                eos,
                sign_init,
            )

        # -------------------------------------------------------------------
        # Branch 3 ‑ default / non‑generation path
        # -------------------------------------------------------------------
        return self._process_standard(
            labels_text,
            pre_tokens,
            pre_signs,
            pre_labels,
            token_masked,
            token_unmask,
            output_type,
            eos,
            sign_init,
        )

# ---------------------------------------------------------------------------
# Helper 1 ‑ processing *only* keyword‑generation modes
# ---------------------------------------------------------------------------

    def _process_keywords(
        self,
        output_type: str,
        token_target: List[int],
    ):
        """Return early for `keywords_predict` / `keywords_valid`."""
        prefix = "" if self.is_zero_shot else "###"

        if self.generation_state.mode == "keywords_predict":
            header = f"{prefix} keywords of {output_type}: \n"
            tokens = self._encode(header)
            return tokens, [-100] * len(tokens), [0] * len(tokens)

        elif self.generation_state.mode == "keywords_valid":
            header = f"{prefix} keywords of {output_type}: \n"
            kw_tag = f"<|KW|>{token_target}<|KW|>"
            tokens = {kw_tag: token_target + self.special_tokens.bad}
            signs = {kw_tag: [0] * len(token_target) + [1]}
            tokens, signs = self._encode(header + kw_tag, tokens, signs, return_signs=True)
            labels = [i if signs[idx] != 0 else -100 for idx, i in enumerate(tokens)]
            return tokens, labels, signs
        else:
            assert 0, f'{self.generation_state.mode =} is error!'

# ---------------------------------------------------------------------------
# Helper 2 ‑ build the *common* keyword section
# ---------------------------------------------------------------------------

    def _build_keyword_section(
        self,
        token_target: List[int],
        output_type: str,
        eos: int,
        sign_init: int,
        is_nega_data: bool = False,
    ):
        """Assemble the keyword block shared by all non‑keyword branches."""
        eos = self.special_tokens.bad[0] if is_nega_data else eos
        sign_init = 0
        prefix = "" if self.is_zero_shot else "###"
        kw_tag = f"<|KW|>{token_target}<|KW|>"
        labels_text = f"{prefix} keywords of {output_type}: \n{kw_tag}"

        try:
            pre_tokens = {kw_tag: token_target + [eos]}
        except:
            assert 0, token_target
        sign_keyword = 0 if not self.is_generation and self.ablation_type == "w/o_kw" else sign_init

        pre_signs = {kw_tag: [sign_keyword] * len(token_target) + [sign_keyword]}
        pre_labels = {kw_tag: ((token_target + [eos]) if sign_keyword else ([-100]*len(token_target)+[-100]))}
        return labels_text, pre_tokens, pre_signs, pre_labels

# ---------------------------------------------------------------------------
# Helper 3 ‑ handle pre_masked / post_masked / filling
# ---------------------------------------------------------------------------

    def _process_masked_generation(
        self,
        index: int,
        labels_text: str,
        pre_tokens: Dict[str, List[int]],
        pre_signs: Dict[str, List[int]],
        pre_labels: Dict[str, List[int]],
        token_target: List[int],
        token_masked: List[int],
        output_type: str,
        eos: int,
        sign_init: int,
    ):
        """Internal handler for mask‑centric generation modes."""
        prefix = "" if self.is_zero_shot else "###"
        min_len, max_len = self.add_mask_min_length, self.add_mask_max_length
        good_id, bad_id = self._good_bad_ids('mask')

        def _finalise() -> Dict[str, List[int]]:
            mt_tag = f"<|MT|>{token_masked}<|MT|>"
            block = f"{prefix} masked {output_type}: \n{mt_tag}"
            if self.generation_state.mode == "response_filling":
                block += f"{prefix} {output_type}: \n"
            full_text = labels_text + block
            pre_tokens[mt_tag] = token_masked
            pre_signs[mt_tag] = signs_masked
            pre_labels[mt_tag] = labels_masked
            tokens, signs, labels = self._encode(full_text, pre_tokens, pre_signs, pre_labels, True, True)
            return tokens, labels, signs

        # ── ①  mask_pre_generate / mask_pre_valid  ───────────────────
        if self.generation_state.mode in {"mask_pre_generate", "mask_pre_valid"}:
            pre_signs = {k: [0 for s in v] for k, v in pre_signs.items()}
            pre_labels = {k: [-100 for s in v] for k, v in pre_labels.items()}
            token_masked = self._mask([0] * max_len)
            if "valid" in self.generation_state.mode:
                # 修补 _process_masked_generation -> mask_pre_valid
                split = random.randint(min_len, max_len - 1)   # 真正应该是 GOOD 的槽位
                signs_masked = [0]*min_len + [1]*(max_len-min_len)
                labels_masked = [-100]*min_len + [bad_id]*(max_len-min_len)
                labels_masked[split] = good_id
            else:
                valid_tokens = [eos, -1, 0, 1]
                assert any(token not in valid_tokens for token in token_target), f"{token_target=}"

                index = 0
                while index < len(token_target)-1 and token_target[index] not in valid_tokens:
                    index += 1

                signs_masked = [0]*min_len + [1]*(max_len-min_len)
                labels_masked = [-100]*min_len + [token_target[index]]*(max_len-min_len)

            return _finalise()

        # ── ②  mask_post_generate / mask_post_valid  ─────────────────
        if self.generation_state.mode in {"mask_post_generate", "mask_post_valid"}:
            assert self.generation_state.masked_pre and self.generation_state.masked_pre[index] is not [], f'{index=}'
            pre_signs = {k: [0 for s in v] for k, v in pre_signs.items()}
            pre_labels = {k: [-100 for s in v] for k, v in pre_labels.items()}
            token_masked = (
                self._mask([0]*(min_len + self.generation_state.masked_pre[index]))
                + token_target
                + self._mask([0]*(max_len))
            )
            if "valid" in self.generation_state.mode:
                eos = bad_id

            pre_len = min_len + self.generation_state.masked_pre[index] + len(token_target)
            signs_masked = ([0]*pre_len + [0]*min_len + [1]*(max_len-min_len))
            labels_masked = ([-100]*pre_len + [-100]*min_len + [eos]*(max_len-min_len))
            if "valid" in self.generation_state.mode:
                split = random.randint(min_len, max_len - 1)
                labels_masked[pre_len + split] = good_id
            return _finalise()

        # ── ③  response_filling  (依赖于已经得到的 mask 计数) ────────
        if (
            self.generation_state.mode == "response_filling"
            and len(self.generation_state.masked_pre) > index
            and len(self.generation_state.masked_post) > index
        ):
            left_len  = min_len + self.generation_state.masked_pre[index]
            right_len = min_len + self.generation_state.masked_post[index]
            token_masked = (
                self._mask([0] * left_len)
                + token_target
                + self._mask([0] * right_len)
                + [eos]
            )
            signs_masked = [sign_init] * len(token_masked)
            labels_masked = token_masked if sign_init != 0 else [-100] * len(token_masked)
            return _finalise()

        # ── ④  兜底：沿用原随机策略 ────────────────────────────────
        token_masked = self.add_mask_token(" ", token_target) + [eos]
        signs_masked = [sign_init] * len(token_masked)
        labels_masked = token_masked if sign_init != 0 else [-100] * len(token_masked)
        return _finalise()

# ---------------------------------------------------------------------------
# Helper 4 ‑ default / non‑generation path
# ---------------------------------------------------------------------------

    def _process_standard(
        self,
        labels_text: str,
        pre_tokens: Dict[str, List[int]],
        pre_signs: Dict[str, List[int]],
        pre_labels: Dict[str, List[int]],
        token_masked: List[int],
        token_unmask: List[int],
        output_type: str,
        eos: int,
        sign_init: int,
    ) -> Dict[str, List[int]]:
        """Handle classic training mode plus ablation switches."""
        prefix = "" if self.is_zero_shot else "###"
        good_id, bad_id = self._good_bad_ids('mask')
        mask_id = self.special_tokens.mask[0]

        # Identify mask segments:  mask ... unmask ... mask
        pro_mask, unmask, post_mask = split_three_parts(token_masked, value=mask_id)
        pro_sign, unmask_sign, post_sign = [0]*len(pro_mask), [0]*len(unmask), [0]*len(post_mask)
        pro_label, unmask_label, post_label = [-100]*len(pro_mask), unmask, [-100]*len(post_mask)
        if 'NegaMK' in self.train_task_type and self.ablation_type != "w/o_mk":
            if pro_sign:
                pro_sign = [-1]*len(pro_mask)
                pro_sign[-1] = 1
                pro_label = [bad_id]*len(pro_mask)
                pro_label[-1] = good_id
            if post_sign:
                post_sign = [-1]*len(post_mask)
                post_sign[-1] = 1
                post_label = [bad_id]*len(post_mask)
                post_label[-1] = good_id

        if self.robust_type in ["Mask", "Fuse"] and self.ablation_type != "w/o_mk":
            self.robust_mask(pro_mask, pro_sign, pro_label, max_cap=self.robust_value)
            self.robust_mask(post_mask, post_sign, post_label, max_cap=self.robust_value)

        token_masked = pro_mask + unmask + post_mask
        sign_mask = pro_sign + unmask_sign + post_sign
        label_mask = pro_label + unmask_label + post_label
        if 'NegaMK' in self.train_task_type and self.ablation_type != "w/o_mk":
            balance_two_values(sign_mask, v1=1, v2=-1, fill=0, ratio=self.bad_tok_ratio)

        mt_tag = f"<|MT|>{token_masked}<|MT|>"
        labels_text += f"{prefix} masked {output_type}: \n{mt_tag}"
        pre_tokens[mt_tag] = token_masked + [eos]
        pre_signs[mt_tag] = sign_mask + [0]
        pre_labels[mt_tag] = label_mask + [-100]

        tokens, signs, labels = self._encode(labels_text, pre_tokens, pre_signs, pre_labels, True, True)
        return tokens, labels, signs

    def robust_mask(self, tokens, signs, labels,
                    alpha=1.0, beta=0.35, p_del=0.2,
                    max_cap=8):
        """
        动态增删 MASK：
        - 增：在末尾追加 Δ 个 <mask>
        - 删：裁掉序列右侧 Δ 个 token
        Δ = round(alpha * L**beta)，再随机取 ± 方向
        """
        good_id, bad_id = self._good_bad_ids('mask')
        L = len(tokens)
        if L == 0:
            return  # 空序列直接跳过

        # --- 1) 计算 Δ ---
        delta = int(round(alpha * (L ** beta)))
        delta = max(1, min(delta, max_cap))   # 至少 1，至多 max_cap

        # --- 2) 决定增删方向 ---
        if random.random() < p_del:           # 删除
            delta = -delta

        # --- 3) 应用修改 ---
        if delta > 0:                         # 增：追加 Δ 个 MASK
            tokens += [self.special_tokens.mask[0]] * delta
            signs  += [-1 if 'NegaMK' in self.train_task_type else 0] * delta
            labels += [bad_id if 'NegaMK' in self.train_task_type else -100] * delta
        else:                                 # 删：只要删除量 < L
            cut = max(0, L + delta)           # delta 为负
            tokens[:] = tokens[:cut]
            signs[:]  = signs[:cut]
            labels[:] = labels[:cut]

    # ================================
    # 类别 3: triples 处理方法
    # ================================

    def _build_triple_from_dialogue(
        self,
        index: int,
        kb_triples: List[Triple],
        render: Optional[List[int]],
        history: List[List[int]],
        response: List[int],
    ) -> Dict[str, List[int]]:
        """ history + response -> triple"""
        task = self.train_task_type
        if not self.is_generation and "Nega" in task and "RS" in task:
            if random.random() < 0.5:
                nega_response = self.data[random_index_exclude(len(self.data), index)].response
                res_as_kg = (self._encode(nega_response), -1)
            else:
                res_as_kg = (response, 1)
            render += [res_as_kg]
            random.shuffle(render)

        token_triples, signs_triples, labels_triples, triples_lens = self._triples_process(
            index, kb_triples, mmi=True, return_labels=True, return_seg_lens=True
        )
        token_render, signs_render, labels_render, render_lens = self._render_process(
            index, render, mmi=True, return_labels=True, return_seg_lens=True
        )
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
            messages, tokenize=False, add_generation_prompt=True
        )
        token_input, signs_input = self._encode(
            text_input, pre_encoded_tokens, pre_encoded_signs, return_signs=True
        )

        input_ids = token_input + token_triples + token_render
        label_ids = [-100] * len(token_input) + labels_triples + labels_render
        signs_all = signs_input + signs_triples + signs_render

        is_NegaKW, _, _ = self.mmi_task_type()      # 你已有的判别函数
        attn_mask = [1] * len(input_ids)

        return self.model_args({
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': label_ids,
            'signs': signs_all,
        })

    # ================================
    # 类别 3: model_args 处理方法
    # ================================

    def model_args(self, batch):
        return {
            k: v
            for k, v in batch.items()
            if (self.include_nones or v is not None)
            and (not self.exclude_labels or k != "labels")
            and (not self.is_generation or k != "token_target")
            and k not in ("triple_ids")
        }

    # ================================
    # 类别 3: triples 处理方法
    # ================================

    def _triples_process(
        self,
        index: int,
        kb_triples: List[Triple],
        mmi: bool = False,
        return_labels: bool = False,
        return_seg_lens: bool = False,
    ):
        """
        依据三元组构造 token / sign / label 序列。
        当 `self.is_debug` 为 True 时，自动检查：
          • token-sign-label 长度一致
          • GOOD / BAD 样本分布
          • sign 与 label 是否一致
        """
        IGNORE = -100
        tok_all, sign_all, lab_all, seg_lens = [], [], [], []

        # -------- 1. 建立正向实体集合（加速匹配）---------------------------
        pos_ent_set = []
        if self.train_task_type in MMI_TASKS and self.data[index].kb_triples:
            for t in self.data[index].kb_triples:
                pos_ent_set.extend([t.subject, t.object])

        # -------- 2. 遍历三元组 -------------------------------------------
        for tri in kb_triples:
            # 2-1. 组装三元组 token 序列
            tok_tri = (
                self.special_tokens.triple_start
                + self.special_tokens.entity_start + tri.subject + self.special_tokens.entity_end
                + self.special_tokens.relation_start + tri.predicate + self.special_tokens.relation_end
                + self.special_tokens.entity_start + tri.object   + self.special_tokens.entity_end
                + self.special_tokens.triple_end
            )
            tok_all.extend(tok_tri)
            seg_lens.append(len(tok_tri))

            # ---------- 若无 MMI，直接写零填充 ------------------------------
            if not mmi:
                sign_all.extend([0] * len(tok_tri))
                lab_all .extend([IGNORE] * len(tok_tri))
                continue

            # 2-2. MMI 情况下构造 sign / label
            is_nega_kw, is_nega_kg, ldict = self.mmi_task_type()

            # subject
            subj_flag  = 2 if tuple(tri.subject) in pos_ent_set else tri.sign_subject
            subj_signs = [0] * len(tri.subject)
            subj_labs  = [
                t if subj_flag != 0 else IGNORE
                for t in tri.subject
            ]

            # predicate
            pred_signs = [0 if self.only_cls_learning else tri.sign_predicate] * len(tri.predicate)
            pred_labs  = [
                t if tri.sign_predicate != 0 else IGNORE
                for t in tri.predicate
            ]

            # object
            obj_flag   = 2 if tuple(tri.object) in pos_ent_set else tri.sign_object
            obj_signs  = [0 if self.only_cls_learning else obj_flag] * len(tri.object)
            obj_labs   = [
                t if obj_flag != 0 else IGNORE
                for t in tri.object
            ]

            # entity / triple 结束标记
            end_ent_sign = tri.sign_subject if is_nega_kw else 0
            end_ent_lab  = ldict[tri.sign_subject] if is_nega_kw else IGNORE

            end_tri_sign = tri.sign_subject if is_nega_kg else 0
            end_tri_lab  = ldict[tri.sign_subject] if is_nega_kg else IGNORE

            sign_tri = (
                [0] * len(self.special_tokens.triple_start)
                + [0] * len(self.special_tokens.entity_start) + subj_signs + [end_ent_sign] * len(self.special_tokens.entity_end)
                + [0] * len(self.special_tokens.relation_start) + pred_signs + [0] * len(self.special_tokens.relation_end)
                + [0] * len(self.special_tokens.entity_start) + obj_signs  + [end_ent_sign] * len(self.special_tokens.entity_end)
                + [end_tri_sign] * len(self.special_tokens.triple_end)
            )
            lab_tri = (
                [IGNORE] * len(self.special_tokens.triple_start)
                + [IGNORE] * len(self.special_tokens.entity_start) + subj_labs + [end_ent_lab] * len(self.special_tokens.entity_end)
                + [IGNORE] * len(self.special_tokens.relation_start) + pred_labs + [IGNORE] * len(self.special_tokens.relation_end)
                + [IGNORE] * len(self.special_tokens.entity_start) + obj_labs  + [end_ent_lab] * len(self.special_tokens.entity_end)
                + [end_tri_lab] * len(self.special_tokens.triple_end)
            )

            sign_all.extend(sign_tri)
            lab_all .extend(lab_tri)

        # -------- 3. 额外处理：learning_type 相关覆盖 ----------------------
        if mmi and not self.learning_type:
            good_id, bad_id = self._good_bad_ids('kg')
            new_sign, new_lab = [], []
            for s, l in zip(sign_all, lab_all):
                if s < 0:   # 负 sign → 视为无监督
                    if l in [good_id, bad_id]:   # 负 sign → 视为无监督
                        new_sign.append(1)
                        new_lab.append(l)
                    else:
                        new_sign.append(0)
                        new_lab.append(IGNORE)
                else:
                    new_sign.append(s)
                    new_lab.append(l)
            sign_all, lab_all = new_sign, new_lab          # 覆盖原列表

        # -------- 4. 调试模式：一致性与分布验证 ----------------------------
        if getattr(self, "is_debug", False) and self.train_task_type == 'NegaKW' and tok_all:
            # 4-1. 长度一致
            assert len(tok_all) == len(sign_all) == len(lab_all), \
                f"[DEBUG] 长度不一致 token={len(tok_all)}, sign={len(sign_all)}, label={len(lab_all)}"

            # 4-2. GOOD / BAD 计数
            kg_good, kg_bad = self._good_bad_ids("kg")
            g_cnt = sum(l == kg_good for l in lab_all)
            b_cnt = sum(l == kg_bad  for l in lab_all)
            logger.info(f"[DEBUG] sample={index}  GOOD={g_cnt}  BAD={b_cnt}  ratio(B/G)={b_cnt/(g_cnt or 1):.3f}")
            if b_cnt == 0:
                logger.warning(f"[DEBUG] sample={index} **没有 BAD 标签**，注意样本不平衡。")

            # 4-3. sign-label 对齐校验
            for pos, (s, l) in enumerate(zip(sign_all, lab_all)):
                if l != IGNORE:
                    assert s != 0, f"[DEBUG] {sign_all=} {lab_all=} {pos=} 有 label 却 sign 为 0"
                else:
                    # 没有 label 的位置允许任何 sign，但可添加更多规则
                    pass

        # -------- 5. 返回 -------------------------------------------------
        ret: list[Any] = [tok_all, sign_all]
        if return_labels:
            ret.append(lab_all)
        if return_seg_lens:
            ret.append(seg_lens)

        return tuple(ret)

    def mmi_task_type(self):
        good_id, bad_id = self._good_bad_ids('kg')

        if self.is_generation:
            is_NegaKW = is_NegaKG = False
            label_dict = [bad_id, bad_id, good_id, bad_id]
            if self.generation_state.mode in ['retrieved_keywords_valid', 'retrieved_kwd_and_kg_valid', "retrieved_kwd_as_kg_valid"]:
                is_NegaKW = True
            if self.generation_state.mode in ['retrieved_knowledges_valid', 'retrieved_kwd_and_kg_valid']:
                is_NegaKG = True
        else:
            is_NegaKW = True if 'Nega' in self.train_task_type and 'KW' in self.train_task_type else False
            is_NegaKG = True if 'Nega' in self.train_task_type and 'KG' in self.train_task_type else False
            label_dict = [-100, good_id, good_id, bad_id]

        return is_NegaKW, is_NegaKG, label_dict

    # ================================
    # 类别 4: render 处理方法
    # ================================

    def _render_process(
        self,
        index: int,
        render_input_list: List[Tuple[List[int], int]],
        mmi: bool = False,
        return_labels=False,
        return_seg_lens: bool = False
    ):
        token_render_processed_all, signs_render_processed_all, labels_render_processed_all = [], [], []
        seg_lens = []
        is_NegaKW, is_NegaKG, label_dict = self.mmi_task_type()
        good_id, bad_id = self._good_bad_ids('kg')
        # resp_text  = self.data[index].response.lower()  # 假设为 str
        # kb_texts   = self.data[index].render_kb.lower()

        for render_item_tokens, original_sign in render_input_list:
            cur_tok, cur_sig, cur_lab = [], [], []

            # [TRIPLE_START]
            cur_tok.extend(self.special_tokens.triple_start)
            cur_sig.extend([0] * len(self.special_tokens.triple_start))
            cur_lab.extend([-100] * len(self.special_tokens.triple_start))

            text = self.tokenizer.decode(render_item_tokens)
            ents = self._extract_topic_entities(index, text, mmi)
            nokg_sign = original_sign if self.mmi_nokg_learning else 0
            last = 0

            if not is_NegaKW and ents:
                for label, ent_text, (st, ed) in ents:
                    # 1) plain text 前缀
                    if st > last:
                        pre = self.tokenizer.encode(text[last:st], add_special_tokens=False)
                        cur_tok.extend(pre); cur_sig.extend([nokg_sign]*len(pre)); cur_lab.extend(pre)

                    # 2) entity 本身
                    ent_ids = self.tokenizer.encode(ent_text, add_special_tokens=False)
                    cur_tok.extend(self.special_tokens.entity_start)
                    cur_sig.extend([0]*len(self.special_tokens.entity_start))
                    cur_lab.extend([-100]*len(self.special_tokens.entity_start))

                    # sign / label
                    sign_ent = original_sign
                    if ent_text in self.data[index].render_kb: sign_ent = 1
                    elif ent_text in self.data[index].response: sign_ent = 2

                    cur_tok.extend(ent_ids)
                    cur_sig.extend([sign_ent]*len(ent_ids))
                    cur_lab.extend(ent_ids)

                    cur_tok.extend(self.special_tokens.entity_end)
                    cur_sig.extend([sign_ent if is_NegaKW else 0] * len(self.special_tokens.entity_end))
                    cur_lab.extend([label_dict[sign_ent] if is_NegaKW else -100] * len(self.special_tokens.entity_end))

                    last = ed

            # 3) 尾部剩余文本
            if last < len(text):
                tail = self.tokenizer.encode(text[last:], add_special_tokens=False)
                cur_tok.extend(tail); cur_sig.extend([nokg_sign]*len(tail)); cur_lab.extend(tail)

            # [TRIPLE_END]
            cur_tok.extend(self.special_tokens.triple_end)
            cur_sig.extend([original_sign if is_NegaKG and mmi else 0] * len(self.special_tokens.triple_end))
            cur_lab.extend([label_dict[original_sign] if is_NegaKG and mmi else -100] * len(self.special_tokens.triple_end))

            # 汇总
            token_render_processed_all.extend(cur_tok)
            signs_render_processed_all.extend(cur_sig)
            labels_render_processed_all.extend(cur_lab)
            seg_lens.append(len(cur_tok))

        # balance GOOD / BAD
        g_cnt = labels_render_processed_all.count(good_id)
        b_cnt = labels_render_processed_all.count(bad_id)

        # 只有足够长时才平衡；ratio 用 self.bad_tok_ratio
        if is_NegaKW and (g_cnt + b_cnt) > 10:
            labels_render_processed_all = balance_two_values(
                labels_render_processed_all,
                v1=good_id, v2=bad_id,
                fill=-100,
                ratio=self.bad_tok_ratio
            )

        # 清洗负 sign
        if mmi and not self.learning_type:
            new_sig, new_lab = [], []
            for s, l in zip(signs_render_processed_all, labels_render_processed_all):
                if s < 0:
                    if l in (good_id, bad_id):
                        new_sig.append(1); new_lab.append(l)
                    else:
                        new_sig.append(0); new_lab.append(-100)
                else:
                    new_sig.append(s); new_lab.append(l)
            signs_render_processed_all, labels_render_processed_all = new_sig, new_lab

        ret: list[Any] = [token_render_processed_all, signs_render_processed_all]
        if return_labels:
            ret.append(labels_render_processed_all)
        if return_seg_lens:
            ret.append(seg_lens)

        return tuple(ret)


    def _extract_topic_entities(self, index, text: str, mmi: bool) -> List[Tuple[str, str, Tuple[int, int]]]:
        """
        返回 [(label, ent_text, (start, end)), ...]
        第一项如果存在必为 (“TOPIC”, topic_text, span)
        """
        topic = None; topic_end = 0
        m = re.search(r'^[\[\s"]*([\w\s\-]+?)\s*[:：]', text)
        if m:
            topic_txt = m.group(1).strip()
            if topic_txt and topic_txt != "no_passages_used":
                topic_end = m.end(1)
                topic = ("TOPIC", topic_txt, (m.start(1), m.end(1)))

        ents = self.nering(text, ['ent'], start_pos=topic_end) if mmi else []      # spaCy 实体
        ents = list(ents) if ents else []

        if mmi and topic:
            ents = [e for e in ents if e[1].strip() != topic[1]]
            ents.insert(0, topic)

        # 随机打散非 topic 部分，截前 10，按起始位置排序
        if ents:
            head, tail = ents[:1], ents[1:]
            random.shuffle(tail)
            ents = (head + tail)[:10] if topic else tail[:10]
            ents.sort(key=lambda x: x[2][0])

        return ents

    # ================================
    # 类别 5: response 处理方法
    # ================================

    def _response_process(self, index, response, mmi=False, ori_gen=False):
        if ori_gen or mmi or self.ori_generate or self.train_task_type == "LM":
            return super()._response_process(index, response, mmi)

        if self.is_generation:
            # 获取原始输出
            raw_output = self._generate_target_tokens(
                index,
                self.extract_kg_from,
                muti_output=self.muti_output,
            )
            # 根据 muti_output 分支处理
            if not raw_output:
                return raw_output, [], []

            if self.muti_output:
                target_list, mask_list, unmask_list = self._generation_multi_output(raw_output)
            else:
                target_list, mask_list, unmask_list = self._generation_single_output(raw_output)
            return target_list, mask_list, unmask_list
        else:
            target_tokens, masked_tokens, unmask_tokens = self._process_masked_tokens(index, response)

            return target_tokens, masked_tokens, unmask_tokens

    def _generate_target_tokens(self, index, extract_kg_from, generate_signs=False, muti_output=False):
        target_tokens = []
        if extract_kg_from in ['context', 'target_kg', "negative_kg", 'retrieved_kg', 'generated_kg']:
            if self.generation_state.keywords:
                keywords = self.generation_state.keywords[index]
                if keywords:
                    keywords_str = self.tokenizer.decode(keywords).lstrip()
                    if "no_passages_used" not in keywords_str:
                        target_tokens = self.tokenizer.encode(" "+keywords_str, add_special_tokens=False)
                else:
                    target_tokens = []
            else:
                if extract_kg_from == 'context':
                    target_tokens = self.kg_extract_and_encode(self.data[index].history[-1])
                    if not target_tokens and len(self.data[index].history) > 1:
                        target_tokens = self.kg_extract_and_encode(self.data[index].history[-2])

                elif extract_kg_from == 'target_kg':
                    kb_triples = self.data[index].kb_triples if self.data[index].kb_triples else self.data[index].render_kb
                    target_tokens = self.kg_select_and_encode(kb_triples) if self.data[index].kb_triples else self.kg_extract_and_encode(kb_triples, max_num=5)
                    target_tokens = target_tokens if muti_output else target_tokens[-1]

                elif extract_kg_from == 'negative_kg':
                    kb_triples = self.data[index].nega_kb_triples if self.data[index].nega_kb_triples else self.data[index].nega_render_kb
                    target_tokens = self.kg_select_and_encode(kb_triples) if self.data[index].nega_kb_triples else self.kg_extract_and_encode(kb_triples, max_num=5)
                    target_tokens = target_tokens if muti_output else random.choice(target_tokens)

                elif extract_kg_from == 'retrieved_kg':
                    kb_triples = self.data[index].retri_kb_triples if self.data[index].retri_kb_triples else self.data[index].retri_render_kb
                    target_tokens = self.kg_select_and_encode(kb_triples) if self.data[index].retri_kb_triples else self.kg_extract_and_encode(kb_triples, max_num=5)
                    # if not target_tokens:
                    #     target_tokens = self.kg_extract_and_encode(self.data[index].history[-1])
                    #     if not target_tokens and len(self.data[index].history) > 1:
                    #         target_tokens = self.kg_extract_and_encode(self.data[index].history[-2])

            if not target_tokens and self.generation_state.keywords_gens and self.generation_state.keywords_gens[index]:
                target_tokens = self.generation_state.keywords_gens[index]

            # elif self.generation_state.knowledges and self.generation_state.knowledges[index]:
            #     knowledges = self.generation_state.knowledges[index]
            #     knowledges = self.tokenizer.decode(knowledges)
            #     target_tokens = " " + re.findall(r'"(.*?)"', knowledges)[-1]
            #     target_tokens = self.tokenizer.encode(target_tokens)

        assert self.is_generation or target_tokens, f'target_tokens is None, {index=} {extract_kg_from=}'

        return target_tokens

    def _generation_multi_output(self, target_list: List[List[int]]) -> (List[List[int]], List[List[int]], List[List[int]]):
        """
        处理多输出模式：保持二维输出结构，并为每个子输出生成对应的 mask 和 unmask 列表。
        返回 (target_list, mask_list, unmask_list)。
        """
        # 确保 target_list 为 2D
        if not (isinstance(target_list, list) and target_list and isinstance(target_list[0], list)):
            target_list = [target_list]
        mask_list = [[] for _ in target_list]
        unmask_list = [[] for _ in target_list]
        return target_list, mask_list, unmask_list

    def _generation_single_output(self, target_list: Any) -> (List[int], List[Any], List[Any]):
        """
        处理单输出模式：展开任何多余的二维嵌套，返回 (target, [], [])。
        """
        # 如果是多层嵌套，取第一层
        while isinstance(target_list, list) and target_list and isinstance(target_list[0], list):
            target_list = random.choice(target_list)
        # 单输出时无 mask/unmask
        return target_list, [], []

    def _process_masked_tokens(self, index, response):
        prefix_seq, mask_seq, postfix_seq = [], [], []
        if "SeqMask" in self.train_task_type:
            prefix_seq, mask_seq, postfix_seq = self._seq_mask(response)
        elif "EntMask" in self.train_task_type:
            prefix_seq, mask_seq, postfix_seq = self._ent_mask(index)
        elif "SpanMask" in self.train_task_type:
            prefix_seq, mask_seq, postfix_seq = self._span_mask(index)

        target_tokens = mask_seq
        masked_tokens = self._mask(prefix_seq) + self._unmask(mask_seq) + self._mask(postfix_seq)
        unmask_tokens = prefix_seq + (self.special_tokens.mask if self.output_type == "Filling" else mask_seq) + postfix_seq

        return target_tokens, masked_tokens, unmask_tokens

    def _mask(self, tokens: TokenizedText):
        if self.is_generation:
            return self.special_tokens.mask * (len(tokens))

        if self.mask_type == 'muti':
            return self.special_tokens.mask * (len(tokens))
        elif self.mask_type == 'one':
            return self.special_tokens.mask
        else:
            return []

    def _mask_sign(self, tokens: TokenizedText, sign: int):
        if self.is_mask_learning:
            return [sign] * len(tokens)
        return [0 if tokens[i] == self.special_tokens.mask[0] else s for i, s in enumerate([sign] * len(tokens))]

    def _unmask(self, tokens: TokenizedText):
        # tokens = tokens.copy()
        if not self.is_generation and self.robust_type in ["Unmask", "Fuse"]:
            for _ in range(random.randint(0, 3)):
                random_token = random.randint(0, len(self.tokenizer)-1)
                self._insert_token_to_random_side(tokens, random_token)
        return tokens

    def _insert_token_to_random_side(self, lst: list, token: int):
        lst.insert(0, token) if random.choice([1, 0]) else lst.append(token)

    def _seq_mask(self, response: TokenizedText):
        n = len(response)
        if n < self.unmask_min_length + 2:      # 太短直接全句
            return [], response, []

        start, end = response_random_uniform_extract(
            n=n,
            min_len=self.unmask_min_length,
            max_len=self.unmask_max_length,
            ensure_middle=True,
            p_reverse=0.5,
        )

        prefix_seq   = response[:start]
        extractedSeq = response[start:end]
        postfix_seq  = response[end:]

        return prefix_seq, extractedSeq, postfix_seq

    def nering(
        self,
        text: str,
        categories: List[str] = ["ent"],
        max_num: int = 10,
        priority: bool = False,
        start_pos: int = 0,
        min_num: int = 3,
    ) -> List[Tuple[List[str], str, Tuple[int, int]]]:
        """
        按指定类别抽取 span，并合并重叠片段。

        Parameters
        ----------
        text : str
            原始文本。
        categories : list[str], optional
            抽取类别及其优先级顺序，默认 ['ent']。
        max_num : int, default 10
            最多返回的 span 数。
        priority : bool, default False
            True  → 命中首个非空类别即停止遍历后续类别；
            False → 遍历全部类别或直到达到 max_num。
        start_pos : int, default 0
            只保留 **起始字符索引 ≥ start_pos** 的 span。
        """
        # ---------- 参数检查 ----------
        assert isinstance(categories, list), "`categories` must be a list"
        assert isinstance(max_num, int) and max_num > 0, "`max_num` must be a positive int"
        assert isinstance(start_pos, int) and start_pos >= 0, "`start_pos` must be a non-negative int"

        valid_pos = {
            "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
            "SCONJ", "SYM", "VERB", "X", "SPACE",
        }
        for cat in categories:
            assert cat == "ent" or cat in valid_pos, f"Unknown category: {cat!r}"

        # ---------- spaCy 解析 ----------
        doc = self.ner(text)
        raw: list[tuple[str, int, int]] = []

        def _extract(cat: str) -> list[tuple[str, int, int]]:
            if cat == "ent":
                return [
                    (ent.label_, ent.start_char, ent.end_char)
                    for ent in doc.ents
                    if ent.start_char >= start_pos                # ← 位置过滤
                ]
            return [
                (tok.pos_, tok.idx, tok.idx + len(tok.text))
                for tok in doc
                if tok.pos_ == cat and tok.idx >= start_pos       # ← 位置过滤
            ]

        # ---------- 抽取并可选短路 ----------
        for cat in categories:
            spans = _extract(cat)
            if spans:
                if max_num and len(raw) + len(spans) >= max_num:
                    spans = random.sample(spans, k=max_num-len(raw))

                raw.extend(spans)
                if (priority and len(raw) > min_num) or (max_num and len(raw) >= max_num):
                    break

        if not raw:
            return []

        # ---------- 截断 & 合并重叠 ----------
        raw.sort(key=lambda x: x[1])

        merged: list[tuple[list[str], str, Tuple[int, int]]] = []
        cur_labels, cur_start, cur_end = [raw[0][0]], raw[0][1], raw[0][2]

        for label, start, end in raw[1:]:
            if start <= cur_end:                       # overlap → merge
                cur_end = max(cur_end, end)
                cur_labels.append(label)
            else:
                merged.append(
                    [list(dict.fromkeys(cur_labels)), text[cur_start:cur_end], (cur_start, cur_end)]
                )
                cur_labels, cur_start, cur_end = [label], start, end

        merged.append(
            [list(dict.fromkeys(cur_labels)), text[cur_start:cur_end], (cur_start, cur_end)]
        )
        return merged


    def _search_seq_index(self, seq, text: str, offset_mapping, seq_index=None):
        if seq_index:
            start, end = seq_index[0], seq_index[1]
        else:
            start = text.find(seq)
            end = start + len(seq)

        start_idx, end_idx = find_closest_offsets_idx(offset_mapping, start, end)
        return start_idx, end_idx

    def _ent_mask(self, posi_index: int, text: str = None):
        if self.spacy_tokenize:
            response = self._word_tokenize(" " + self.data[posi_index].response)
        else:
            response = self.data[posi_index].response
        response = text if text is not None else response
        res_tokens = self.tokenizer(response, return_offsets_mapping=True, add_special_tokens=False)
        res_ids = res_tokens["input_ids"]
        posi_ents = self.nering(response)

        if not posi_ents or random.random() > self.ratio:
            return self._seq_mask(res_ids)
        ent_type, posi_ent, ent_index = random.choice(posi_ents)
        # change posi_ent to nega_ent
        start_idx, end_idx = self._search_seq_index(posi_ent, response, res_tokens["offset_mapping"], ent_index)

        return (res_ids[:start_idx], res_ids[start_idx:end_idx], res_ids[end_idx:])

    def _span_mask(self, posi_index: int):
        response = self.data[posi_index].response
        posi_spans = self.dial_spans[posi_index]['spans']

        res_tokens = self.tokenizer(response, return_offsets_mapping=True, add_special_tokens=False)
        res_ids = res_tokens["input_ids"]

        if posi_spans and random.random() < self.ratio:
            span_type, posi_span = random.choice(posi_spans)
            # change posi_span to nega_ent
            start_idx, end_idx = self._search_seq_index(posi_span, response, res_tokens["offset_mapping"])
            return res_ids[:start_idx], res_ids[start_idx:end_idx], res_ids[end_idx:]
        else:
            return self._seq_mask(res_ids)

    def kg_select_and_encode(self, kb_triples: List[Triple], generate_signs: bool = False) -> List[TokenizedText]:
        def prefix_random(): return random.choice([' ', ' '])
        kg_list = []
        for triple in kb_triples:
            if triple.object != '':
                prefix = prefix_random()
                kg_list.append(self.tokenizer.encode(prefix+triple.object, add_special_tokens=False))
            elif triple.subject != '':
                prefix = prefix_random()
                kg_list.append(self.tokenizer.encode(prefix+triple.subject, add_special_tokens=False))

        return kg_list

    def kg_extract_and_encode(self, render: List[str], max_num: int = 1) -> List[TokenizedText]:
        render = render[0] if type(render) == list else render
        if 'no_passages_used' in render:
            return []
        if len(render) < 10 and len(render.split()) == 1:
            return self.tokenizer.encode(' '+render, add_special_tokens=False)

        ents = self.nering(render, ['ent', 'NOUN'])
        ent_list = [' '+e for _, e, _ in ents]

        if not ent_list:
            logger.warning(f'{render=} cannot detect any entity.')
            return []
        else:
            random.shuffle(ent_list)
            encodings = self.tokenizer.batch_encode_plus(
                ent_list[:max_num], add_special_tokens=False, return_attention_mask=False
            )
            return encodings["input_ids"]

    def add_mask_token(self, prefix: str, tokens: TokenizedText) -> TokenizedText:
        left_mask_length = random.randint(self.add_mask_min_length, self.add_mask_max_length)
        right_mask_length = random.randint(self.add_mask_min_length, self.add_mask_max_length)
        left_mask = self.special_tokens.mask * left_mask_length
        right_mask = self.special_tokens.mask * right_mask_length

        return left_mask + tokens[:5] + right_mask


@dataclass
class BaseCollator:
    pad: int
    kg_pad: Optional[int] = None
    mask_pad: Optional[int] = 0
    label_pad: Optional[int] = -100
    as_tuple: bool = False
    fields: Iterable[str] = field(default_factory=list)
    padding_side: str = 'right'
    pad_to_multiple_of: Optional[int] = None

    @property
    def special_pad_tokens(self) -> Dict[str, int]:
        raise NotImplementedError

    def _get_pad_token(self, field: str) -> int:
        return self.special_pad_tokens.get(field, self.pad)

    def __call__(self, batch: Dict[str, Sequence]):
        padded: Dict[str, torch.Tensor] = {}
        for name in self.fields:
            if name not in batch:
                continue
            values = batch[name]
            pad_token = self._get_pad_token(name)
            padded[name] = self._pad_2d(values, pad_token) if self._is_2d(values) else self._pad_1d(values, pad_token)
        return tuple(padded.get(f) for f in self.fields) if self.as_tuple else padded

    def _is_2d(self, values: Sequence) -> bool:
        return bool(values and isinstance(values[0], (list, tuple, np.ndarray))
                    and values[0] and isinstance(values[0][0], (list, tuple, np.ndarray)))

    def _pad_1d(self, values: Sequence[Sequence[int]], pad_token: int) -> torch.Tensor:
        bsz = len(values)
        max_len = max((len(x) for x in values), default=0)
        if self.pad_to_multiple_of:
            max_len = int(self.pad_to_multiple_of * np.ceil(max_len / self.pad_to_multiple_of))

        arr = np.full((bsz, max_len), pad_token, dtype=np.int64)
        for i, seq in enumerate(values):
            l = len(seq)
            if not l:
                continue
            if self.padding_side == 'right':
                arr[i, :l] = seq
            else:
                arr[i, -l:] = seq
        return torch.from_numpy(arr)

    def _pad_2d(self, values: Sequence[Sequence[Sequence[int]]], pad_token: int) -> torch.Tensor:
        bsz = len(values)
        max_n = max((len(item) for item in values), default=0)
        all_lens = [len(seq) for item in values for seq in item]
        max_l = max(all_lens, default=0)
        if self.pad_to_multiple_of:
            max_l = int(self.pad_to_multiple_of * np.ceil(max_l / self.pad_to_multiple_of))

        arr = np.full((bsz, max_n, max_l), pad_token, dtype=np.int64)
        for i, item in enumerate(values):
            for j, seq in enumerate(item):
                l = len(seq)
                if not l:
                    continue
                if self.padding_side == 'right':
                    arr[i, j, :l] = seq
                else:
                    arr[i, j, -l:] = seq
        return torch.from_numpy(arr)

@dataclass
class Collator(BaseCollator):
    # ---------- 1. 字段 ----------
    fields: Iterable[str] = ("input_ids", "attention_mask", "labels", "signs")
    raw_fields: Iterable[str] = ("token_target", "token_masked", "token_unmask")

    # ---------- 2. pad 值 ----------
    sign_pad:  int   = 0
    mask_pad:  int   = 0          # 传统 1-D/2-D mask（0/1）
    block_pad: float = float("-inf")       # 4-D / block-diag mask
    label_pad: int   = -100

    truncate_len: Optional[int] = None

    # -------------------------------------------------
    #              内部工具：mask 相关
    # -------------------------------------------------
    def _is_4d(self, m):
        """严格判断是否形如 (1,1,L,L) 的 4‑D tensor / 嵌套 list。"""
        if torch.is_tensor(m):
            return m.dim() == 4
        if isinstance(m, (list, tuple)) and m:
            return self._is_4d(m[0])
        return False

    def _mask1d_to_4d(self, vec):
        if not torch.is_tensor(vec):
            vec = torch.as_tensor(vec, dtype=torch.float32)
        if vec.dim() == 2:
            vec = vec.squeeze(0)
        L = vec.numel()
        m = torch.zeros(1, 1, L, L, dtype=torch.float32, device=vec.device)
        m[0, 0, torch.triu(torch.ones(L, L, dtype=torch.bool, device=vec.device), 1)] = self.block_pad
        return m

    def _pad_4d(self, masks):
        ts = [m if torch.is_tensor(m) else torch.as_tensor(m, dtype=torch.float32) for m in masks]
        L_max = max(t.size(-1) for t in ts)
        if self.pad_to_multiple_of:
            import math
            L_max = int(self.pad_to_multiple_of * math.ceil(L_max / self.pad_to_multiple_of))

        out = torch.full((len(ts), 1, L_max, L_max), self.block_pad, dtype=ts[0].dtype, device=ts[0].device)
        for i, t in enumerate(ts):
            L = t.size(-1)
            out[i, :, :L, :L] = t
        return out

    # -------------------------------------------------
    #          其余辅助（未改动，可复用父类）
    # -------------------------------------------------
    @property
    def special_pad_tokens(self) -> Dict[str, int]:
        return {
            "attention_mask": self.mask_pad,
            "labels":         self.label_pad,
            "signs":          self.sign_pad,
        }

    # -------------------------------------------------
    #                 核心入口
    # -------------------------------------------------
    def __call__(self, batch: Sequence[Dict[str, Sequence[int]]]):
        # ---------- 1. list → dict ----------
        grouped, raw_grouped = {}, {}
        for f in self.fields:
            if any(f in s for s in batch):
                grouped[f] = [s.get(f, []) for s in batch]
        for f in self.raw_fields:
            if any(f in s for s in batch):
                raw_grouped[f] = [s.get(f, []) for s in batch]

        # ---------- 2. padding ----------
        padded: Dict[str, torch.Tensor] = {}

        for name, values in grouped.items():
            pad_token = self._get_pad_token(name)

            # ------ 专门处理 attention_mask ------
            if name == "attention_mask":
                if any(self._is_4d(v) for v in values):
                    std = [v if self._is_4d(v) else self._mask1d_to_4d(v) for v in values]
                    padded[name] = self._pad_4d(std)
                else:
                    padded[name] = self._pad_1d(values, pad_token)

            else:
                padded[name] = self._pad_1d(values, pad_token)

        # ---------- 3. 附加 raw 字段 ----------
        padded.update(raw_grouped)

        # ---------- 4. 全局截断（可选） ----------
        if self.truncate_len is not None:
            tl = self.truncate_len
            for k in ("input_ids", "attention_mask", "labels", "signs"):
                if k not in padded:
                    continue
                t = padded[k]
                # 4-D 与 2-D 分开裁
                if torch.is_tensor(t) and t.dim() == 4 and t.size(-1) > tl:
                    padded[k] = t[:, :, -tl:, -tl:]
                elif torch.is_tensor(t) and t.dim() == 2 and t.size(1) > tl:
                    padded[k] = t[:, -tl:]

        # ---------- 5. 返回 ----------
        if self.as_tuple:
            order = list(self.fields) + list(self.raw_fields)
            return tuple(padded.get(f) for f in order)
        return padded
