from transformers import (
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)

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

def tokenizer_and_model_init(model_name_or_path="gpt2", model_mode="language-modeling", return_model=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if not return_model:
        return tokenizer
    model_type = MODEL_MODES[model_mode]
    model = model_type.from_pretrained(model_name_or_path)
    return tokenizer, model