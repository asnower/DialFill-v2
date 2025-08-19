"""
Edited from https://github.com/huggingface/transformers/blob/v3.4.0/examples/lightning_base.py
"""
import argparse
import logging
import os
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelForCausalLM,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
    "conversation": AutoModelForSeq2SeqLM,
}

# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    "constant": get_constant_schedule_with_warmup,  # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        *,
        config: Optional[PretrainedConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[Any] = None,
    ) -> None:
        super().__init__()

        # NB: save *all* passed‑in hparams for Lightning loggers & checkpoints
        self.save_hyperparameters(hparams)

        # House‑keeping
        self.step_count: int = 0
        self.output_dir = Path(self.hparams.output_dir)
        self.cache_dir = self.hparams.cache_dir

        # Core components
        self.config = self._init_config(config)
        self.tokenizer = self._init_tokenizer(tokenizer)
        self.model = self._init_model(model)
        self._resize_embeddings_if_needed()
        self._apply_quantization_and_peft()
        self.generation_config = self._build_generation_config()

        # Misc
        self.debug_vars: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _init_config(self, cfg: Optional[PretrainedConfig]) -> PretrainedConfig:
        """Load or update the HF ``config`` object."""
        if cfg is not None:
            return cfg

        config = AutoConfig.from_pretrained(
            self._maybe_best_model_path(self.hparams.model_name_or_path),
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            # "Phi‑3" models require Flash‑Attention 2; keep it configurable
            **({"attn_implementation": "flash_attention_2"}
               if "Phi-3" in self.hparams.model_name_or_path else {}),
        )

        if self.hparams.peft_pretrain_model and self.hparams.do_train:
            config.inference_mode = False

        # propagate extra dropout/layerdrop values from CLI to config
        for p in ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout"):
            cli_val = getattr(self.hparams, p, None)
            if cli_val is not None and hasattr(config, p):
                setattr(config, p, cli_val)

        return config

    # ------------------------------------------------------------------
    def _init_tokenizer(self, tok: Optional[PreTrainedTokenizer]) -> PreTrainedTokenizer:
        if tok is not None:
            return tok

        path = self.hparams.adapter_name_or_path or self.hparams.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name or path,
            cache_dir=self.cache_dir,
        )

        # guarantee pad‑token existence
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # ------------------------------------------------------------------
    def _init_model(self, mdl: Optional[Any]):
        if mdl is not None:
            return mdl

        model_cls = MODEL_MODES[self.hparams.model_mode]
        model = model_cls.from_pretrained(
            self._maybe_best_model_path(self.hparams.model_name_or_path),
            from_tf=self.hparams.model_name_or_path.endswith(".ckpt"),
            config=self.config,
            cache_dir=self.cache_dir,
            **({"quantization_config": self.create_quantization_config()}
               if self.hparams.quantization else {}),
        )
        logger.info("Model loaded from %s", self.hparams.model_name_or_path)
        return model

    # ------------------------------------------------------------------
    def _resize_embeddings_if_needed(self) -> None:
        """Synchronise embedding matrix with tokenizer size (for new tokens)."""
        current, required = (
            self.model.get_input_embeddings().weight.size(0),
            len(self.tokenizer),
        )
        if required <= current:
            return  # nothing to do

        num_new = required - current
        logger.warning(
            "Tokenizer vocab (%d) > model embeddings (%d); resizing by +%d rows.",
            required, current, num_new,
        )
        self.model.resize_token_embeddings(required)
        self.model.config.vocab_size = required

        # naïve init = global mean/std; customise if needed
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight
            emb[-num_new:].normal_(mean=emb.mean().item(), std=emb.std().item())
            if hasattr(self.model, "get_output_embeddings") and (
                emb.data_ptr() != self.model.get_output_embeddings().weight.data_ptr()
            ):
                self.model.get_output_embeddings().weight[-num_new:].copy_(emb[-num_new:])

        # ensure model uses same pad‑token id as tokenizer
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    def _apply_quantization_and_peft(self) -> None:
        """Wrap model with quantisation and/or PEFT adapters as requested."""
        if self.hparams.quantization:
            logger.info("Applying k‑bit quantisation wrappers …")
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )

        if self.hparams.peft_pretrain_model:
            logger.info("Loading PEFT adapter from %s", self.hparams.adapter_name_or_path)
            self.model = PeftModel.from_pretrained(self.model, self.hparams.adapter_name_or_path)
        elif self.hparams.peft_train:
            logger.info("Creating PEFT config …")
            self.model = self.create_peft_config(self.model, self.hparams)

    # ------------------------------------------------------------------
    def _build_generation_config(self) -> GenerationConfig:
        """Consolidate generation kwargs coming from CLI / hparams."""
        common_kwargs = dict(
            max_new_tokens=self.hparams.max_length,
            min_length=self.hparams.min_length,
            num_beams=self.hparams.num_beams,
            do_sample=self.hparams.do_sample,
            temperature=self.hparams.temperature,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=self.hparams.num_return_sequences,
            use_cache=True,
        )
        if self.hparams.do_sample:
            common_kwargs.update(top_k=self.hparams.top_k, top_p=self.hparams.top_p)
        return GenerationConfig(**common_kwargs)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _maybe_best_model_path(self, path: str) -> str:
        """If ``path`` points to a checkpoint folder with *best_model*, return it."""
        best = Path(path) / "best_model" / self.hparams.model_type
        return str(best) if best.exists() else path

    @staticmethod
    def create_quantization_config():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16,
        )

    @staticmethod
    def create_peft_config(model, hparams):
        peft_conf = LoraConfig(
            r=hparams.lora_r,
            lora_alpha=hparams.lora_alpha,
            lora_dropout=hparams.lora_dropout,
            bias="none",
            task_type=hparams.lora_task_type,
            modules_to_save=None,
            **({"target_modules": hparams.lora_target_modules} if hparams.lora_target_modules else {}),
        )

        model = get_peft_model(model, peft_conf)
        model.print_trainable_parameters() 
        return model

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        total_steps = self.total_steps()
        if self.hparams.warmup_ratio > 0:
            warmup_steps = self.hparams.warmup_ratio * total_steps
        else:
            warmup_steps = self.hparams.warmup_steps

        if self.hparams.lr_scheduler != "constant":
            scheduler = get_schedule_func(self.opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        else:
            scheduler = get_schedule_func(self.opt, num_warmup_steps=warmup_steps)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                scale_parameter=False,
                relative_step=False,
            )

        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def get_number_of_gpus(self):
        if self.hparams.gpus == -1:
            return torch.cuda.device_count()
        else:
            return self.hparams.gpus

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.get_number_of_gpus())
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        lm_warmup_steps = (len(self.total_dataset) / effective_batch_size) * self.hparams.num_lm_warmup_epochs
        dataset_size = len(self.total_dataset)

        return lm_warmup_steps + (dataset_size / effective_batch_size) * (
            self.hparams.max_epochs - self.hparams.num_lm_warmup_epochs
        )

    def setup(self, mode):
        if mode in ("test", "infer", "predict"):
            self.dataset_size = len(self.test_dataloader().dataset)
        elif mode == "train":
            self.train_dataset = self.get_dataset("train", self.hparams.train_dataset_path)
            self.dataset_size = len(self.train_dataset)

    def get_dataset(self, mode: str, data_path: str, **kwargs):
        raise NotImplementedError("You must implement this for your task")

    def get_collator(self, mode: str):
        return None

    def get_dataloader(
        self,
        mode: str,
        batch_size: int,
        shuffle: bool = False,
        data_path: Optional[str] = None,
        dataset=None,
        **kwargs,
    ):
        if dataset is None:
            assert data_path is not None
            dataset = self.get_dataset(mode, data_path, **kwargs)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.get_collator(mode),
            num_workers=self.hparams.num_workers,
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "train",
            self.hparams.train_batch_size,
            shuffle=True,
            dataset=self.train_dataset,
        )

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "valid",
            self.hparams.eval_batch_size,
            shuffle=False,
            data_path=self.hparams.eval_dataset_path,
        )

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "test", self.hparams.eval_batch_size, shuffle=False, data_path=self.hparams.test_dataset_path
        )

    def predict_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "predict", self.hparams.eval_batch_size, shuffle=False, data_path=self.hparams.predict_dataset_path
        )

    def forward(self, **inputs):
        max_len = getattr(self.model.config, "n_positions", self.model.config.max_position_embeddings)
        inputs_to_keep = {'input_ids', 'attention_mask'}
        filtered_inputs = {k: inputs[k] for k in inputs_to_keep if k in inputs}

        if filtered_inputs['input_ids'].size(1) > max_len:
            batch_size = filtered_inputs['input_ids'].size(1)
            logger.warning(f'{batch_size=} > {max_len=}!')

        return self.model(**filtered_inputs)

    def training_step(self, raw_batch, batch_nb):
        labels = raw_batch.pop('labels')
        signs = raw_batch.pop("signs")

        lm_output = self(**raw_batch)

        loss = self.CL_loss_calculate(lm_output.logits, labels, signs)

        ppl = torch.clamp(torch.exp(loss), min=0)
        self.log("train/lm_ppl", ppl, prog_bar=False)
        self.log("ppl", ppl, prog_bar=True, logger=False)
        self.log("train/lm_loss", loss, prog_bar=False)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/lr", current_lr, prog_bar=False)
        self.log("lr", current_lr, prog_bar=True, logger=False)

        self.log("train/loss", loss, prog_bar=False)

        return loss

    @torch.no_grad()
    def validation_step(self, raw_batch, batch_nb):
        val_loss = 0
        val_output = {}

        labels = raw_batch.pop('labels')
        signs = raw_batch.pop("signs")

        lm_output = self(**raw_batch)

        val_loss += self.CL_loss_calculate(lm_output.logits, labels, signs)
        val_output["val_loss"] = val_loss.detach().cpu()

        num_tokens = (labels > 0).int().sum().detach().cpu()
        val_output["num_tokens"] = num_tokens.detach().cpu()

        self.val_output_list.append(val_output)

    def CL_loss_calculate(
        self,
        logits: Tensor,   # B × L × V    (float16 / bfloat16 under AMP)
        labels: Tensor,   # B × L
        signs:  Tensor,   # B × L
    ) -> Tensor:
        # ---------- 1. 预处理 ----------
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        signs  = signs [...,  1:].contiguous()

        logits_f = logits.view(-1, logits.size(-1))
        labels_f = labels.view(-1)
        signs_f  = signs .view(-1)

        # 计算权重
        delta0 = getattr(self.hparams, "sign_delta0", 0.8)
        decay  = getattr(self.hparams, "sign_linear_decay", 0.1)
        weights = self._sign_weight(signs_f, delta0=delta0, decay=decay).to(logits_f)

        # ---------- 2. 正样本交叉熵 ----------
        pos_loss = F.cross_entropy(
            logits_f, labels_f,
            reduction="none"
        )
        pos_mask  = signs_f > 0
        pos_loss  = (pos_loss * pos_mask.float() * weights).sum()
        pos_tok   = torch.clamp(((pos_mask) & (labels_f != -100)).float().sum(), min=1.0)
        loss      = pos_loss / pos_tok

        # ---------- 3. 负样本对比损失 ----------
        probs      = F.softmax(logits_f, dim=-1, dtype=logits.dtype)
        neg_probs  = (logits_f.new_tensor(1.0) - probs).clamp_min_(1e-5)
        neg_log    = neg_probs.log()

        neg_loss   = F.nll_loss(neg_log, labels_f, reduction="none")
        neg_mask   = signs_f < 0
        neg_loss   = (neg_loss * neg_mask.float() * weights).sum()
        neg_tok    = torch.clamp(((neg_mask) & (labels_f != -100)).float().sum(), min=1.0)

        loss = loss + self.hparams.loss_ratio_CL * neg_loss / neg_tok

        return loss

    # ---------- 阶梯权重：增幅线性衰减 --------------------
    def _sign_weight(
        self,
        signs: torch.Tensor,
        delta0: float = 0.8,
        decay:  float = 0.1,
    ) -> torch.Tensor:
        """
        |sign| == 1  -> weight = 1
        |sign| >= 2  -> weight  = 1 + Σ max(0, Δ₀ - d·(j-1))
        """
        abs_s = signs.abs().float()
        k = (abs_s - 1).clamp_min_(0)                         # 0,1,2,...

        # 基本公式：仍在增长区间时
        weight = 1 + delta0 * k - 0.5 * decay * k * (k - 1)

        # ----------- 处理饱和区 -----------
        if decay > 0:
            k_sat = math.floor(delta0 / decay) + 1            # Python 标量
            w_sat = 1 + delta0 * (k_sat - 1) - 0.5 * decay * (k_sat - 1) * (k_sat - 2)

            # 把标量转换到同 device / dtype，避免 broadcast 警告
            w_sat_t = weight.new_tensor(w_sat)                # 同 device & dtype
            weight = torch.where(k >= k_sat, w_sat_t, weight)
        # 若 decay == 0，则无需饱和处理（恒定增幅）

        return weight


    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output_list = []
        return

    def on_validation_epoch_end(self):
        val_loss_mean = torch.stack([x["val_loss"] / x["num_tokens"] for x in self.val_output_list]).mean().detach().cpu()
        self.log("valid/lm_loss", val_loss_mean, prog_bar=True)
        val_ppl = torch.exp(val_loss_mean)
        self.log("valid/lm_ppl", val_ppl, prog_bar=True)

    def test_step(self, raw_batch, batch_no):
        B = raw_batch["input_ids"].shape[0]

        outputs = self.model.generate(raw_batch["input_ids"], generation_config=self.generation_config)
        if not self.model.config.is_encoder_decoder:
            outputs = outputs[:, len(raw_batch["input_ids"][0]):]
        generated_ids = (
            outputs.reshape(B, self.hparams.num_return_sequences, -1).detach().cpu().tolist()
        )
        input_ids = (
            raw_batch["input_ids"].reshape(B, self.hparams.num_return_sequences, -1).detach().cpu().tolist()
        )

        self.test_output_list.append([
            {
                "input": [t for t in input_ids[i][0] if t != self.tokenizer.pad_token_id],
                "generated": [t for t in generated_ids[i]],
            }
            for i in range(B)
        ])

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output_list = []
        return

    @pl.utilities.rank_zero_only
    def on_test_epoch_end(self):
        dataset = self.get_dataset("test", self.hparams.test_dataset_path)
        self._save_gen_output(dataset.data, self.test_output_list)

    def _save_gen_output(self, data, outputs):
        """
        Persist generation results to disk.

        New features:
            1. Save **all** additional keys present in each model output dict under
               `out_json["model_outputs"]`.
            2. Gracefully fallback for non-serialisable objects.

        Args:
            data (list[Dialogue]): dataset slice aligned with `outputs`.
            outputs (list[list[dict]]): nested list returned by the generation loop.
        """
        # --------- 路径解析 ----------
        if self.hparams.generation_name:
            output_file = os.path.join(self.output_dir,
                                       f"{self.hparams.generation_name}.jsonl")
        else:
            output_file = self._test_output_file(self.hparams.generation_type)
        logger.warning(f"response generation output saved in `{output_file}`")

        # --------- 主循环 ----------
        with open(output_file, "w") as writer:
            idx = 0
            for batch in outputs:
                for output in batch:
                    dialogue = data[idx]
                    idx += 1

                    # ========== 基础字段 ==========
                    response = getattr(dialogue, "original_response",
                                       dialogue.response)
                    out_json = {
                        "speaker": dialogue.speaker,
                        "history": dialogue.history,
                        "response": response,
                        "knowledge_base": {},
                        "nega_knowledge_base": {},
                    }

                    # ========== 知识三元组 ==========
                    if dialogue.kb_triples:
                        triples = (
                            dialogue.target_kb_triples
                            if hasattr(dialogue, "target_kb_triples")
                            else dialogue.kb_triples
                        )
                        out_json["knowledge_base"]["paths"] = [
                            [t.subject, t.predicate, t.object] for t in triples
                        ]
                        out_json["nega_knowledge_base"]["paths"] = [
                            [t.subject, t.predicate, t.object]
                            for t in dialogue.nega_kb_triples
                        ]

                    if dialogue.render_kb is not None:
                        render_kb = (
                            dialogue.target_render_kb
                            if hasattr(dialogue, "target_render_kb")
                            else dialogue.render_kb
                        )
                        out_json["knowledge_base"]["render"] = render_kb

                    # ========== 生成结果 ==========
                    if output.get("generated"):
                        out_json["input"] = self.tokenizer.decode(
                            output["input"], skip_special_tokens=False
                        ).strip()
                        out_json["generated_response"] = [
                            self.tokenizer.decode(gen,
                                                  skip_special_tokens=True).strip()
                            for gen in output["generated"]
                        ]
                    else:
                        out_json["generated_response"] = [
                            gen for gen in dialogue.response
                        ]

                    # ========== 捕获其余键 ==========
                    reserved = {"generated", "input"}   # 已在上面单独处理
                    extra_keys = {
                        k: self._to_serialisable(v)
                        for k, v in output.items()
                        if k not in reserved
                    }
                    if extra_keys:
                        out_json.setdefault("model_outputs", {}).update(extra_keys)

                    # --------- 写入磁盘 ----------
                    writer.write(json.dumps(out_json) + "\n")

    def _to_serialisable(self, obj):
        if isinstance(obj, list):
            if all(isinstance(i, int) for i in obj):
                obj = self.tokenizer.decode(
                    obj, skip_special_tokens=False
                ).strip()
                return obj
            if isinstance(obj[0], list) and all(isinstance(i, int) for i in obj[0]):
                obj = self.tokenizer.batch_decode(
                    obj, skip_special_tokens=False
                )
                return obj
        if isinstance(obj, list) and all(isinstance(i, int) for i in obj):
            obj = self.tokenizer.decode(
                obj, skip_special_tokens=False
            ).strip()
            return obj
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    def _test_output_file(self, generation_type: str = 'output', base_dir: str = None):
        output_name = generation_type
        output_name += "_gen"

        output_name += f"_maxl{self.hparams.max_length}"
        output_name += f"_hist{self.hparams.max_history}"

        if self.hparams.num_return_sequences > 1:
            output_name += f"_num{self.hparams.num_return_sequences}"
        if self.hparams.do_sample:
            if self.hparams.top_k > 0:
                output_name += f"_topk{self.hparams.top_k}"
            if self.hparams.top_p > 0:
                output_name += f"_topp{self.hparams.top_p}"
        elif self.hparams.num_beams:
            output_name += f"_beam{self.hparams.num_beams}"
        else:
            output_name += "_greedy"
        if self.hparams.temperature != 1.0:
            output_name += f"_temp{self.hparams.temperature}"
        if self.hparams.repetition_penalty != 1.0:
            output_name += f"_repp{self.hparams.repetition_penalty}"

        return os.path.join(base_dir or self.hparams.model_name_or_path, f"{output_name}.jsonl")

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_model")

        self.model.config.save_step = self.step_count
        lm_save_path = save_path.joinpath(self.hparams.model_type)
        logger.warning(f'model saved at path: {lm_save_path}')
        self.model.save_pretrained(lm_save_path)
        self.tokenizer.save_pretrained(lm_save_path)


@dataclass
class GenericArguments:
    do_train: bool = False  # Whether to run training
    do_generate: bool = False  # Whether to run generations on the test set
    do_eval: bool = False  # Whether to run evaluation
    do_test: bool = False  # Whether to run prediction on the test set
    do_debug: bool = False  # Whether to run debug mode
    seed: int = 42  # Random seed for initialization
    deterministic: bool = False  # Enables cudnn.deterministic for reproducibility
    overwrite_output_dir: bool = False  # Whether to overwrite the model's directory
    wandb: bool = False  # Use wandb for logging
    namestr: str = "exp1"  # Additional info to describe experiments
    output_dir: str = "./checkpoints"  # Path of the checkpoint directories
    train_dataset_path: str = ""  # Path or URL of the train dataset
    eval_dataset_path: str = ""  # Path or URL of the validation dataset
    test_dataset_path: str = ""  # Path or URL of the test dataset
    predict_dataset_path: str = ""  # Path or URL of the predict dataset
    model_type: str = 'lm'
    generation_type: str = 'output'
    generation_name: str = None
    num_lm_warmup_epochs: int = 0  # Number of LM epochs


@dataclass
class GenerationArguments:
    top_k: int = 0
    top_p: float = 0.9
    num_beams: int = 4
    num_return_sequences: int = 1
    temperature: float = 1.0
    max_length: int = 30
    min_length: int = 2
    do_sample: bool = False
    repetition_penalty: float = 1.0


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = "gpt2"  # Model path or name from huggingface.co/models
    adapter_name_or_path: Optional[str] = None  # Model path or name from huggingface.co/models
    model_mode: Optional[str] = "language-modeling"  # Model mode
    config_name: Optional[str] = ""  # Pretrained config name or path if different from model_name
    cache_dir: Optional[str] = None  # Directory to store pre-trained models downloaded from s3
    tokenizer_name: str = None  # Pretrained tokenizer name or path if different from model_name
    reserved_special_token: str = None
    dropout: float = 0.0  # Dropout rate

    # Optimization parameters
    max_epochs: int = 20  # Number of training epochs
    learning_rate: float = 6.25e-5  # Initial learning rate for Adam
    lr_scheduler: str = "linear"  # Learning rate scheduler type
    weight_decay: float = 0.0  # Weight decay
    adam_epsilon: float = 1e-8  # Epsilon for Adam optimizer
    warmup_steps: int = 0  # Number of warmup steps
    warmup_ratio: float = 0.0  # Warmup ratio proportional to training steps, overrides warmup_steps
    max_grad_norm: float = 1.0  # Max gradient norm for clipping
    adafactor: bool = False  # Whether to use Adafactor optimizer
    accumulate_grad_batches: int = 4  # Number of gradient accumulation steps
    num_workers: int = 1  # Number of workers for DataLoader
    train_batch_size: int = 32  # Training batch size
    eval_batch_size: int = 32  # Evaluation batch size
    gpus: int = -1  # Number of GPUs to use (-1 means all available)
    fp16: bool = False  # Whether to use 16-bit (mixed) precision
    bf16: bool = False  # Whether to use bfloat16 precision
    eval_interval: float = 0.5  # Evaluation interval as fraction of epoch
    pad_to_multiple_of: Optional[int] = None  # Pad sequence to multiple of given value
    loss_ratio_CL: float = 1.0  # The loss ratio of constrative learning.
    sign_delta0: float = 0.8  # The loss ratio of constrative learning.
    sign_linear_decay: float = 0.1  # The loss ratio of constrative learning.
    truncate_len: int = 1024

    # Early stopping parameters
    patience: int = 5  # Number of validation steps to wait for improvement before stopping
    min_delta: float = 0.0  # Minimum change to qualify as an improvement


@dataclass
class PeftArguments:
    quantization: bool = False
    peft_pretrain_model: bool = False  # Whether it is a PEFT pretrain model
    peft_train: bool = False  # Whether to train P
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None  # field(default_factory=lambda: ['o_proj', 'k_proj', 'q_proj', 'v_proj'])
    lora_task_type: str = "CAUSAL_LM"