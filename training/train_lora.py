"""Optional offline LoRA fine-tuning for the Qwen evidence extractor."""
import argparse
import json
import os
from pathlib import Path

from innovation.assess import EXTRACTION


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('qwen', 'train', 'dev', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    train, dev = records(args.train), records(args.dev)
    if not train or not dev or {row['group'] for row in train} & {row['group'] for row in dev}:
        parser.error('Use nonempty, group-disjoint training and development data.')
    if args.output.exists():
        parser.error('Choose a new output directory.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        parser.error('LoRA training requires CUDA.')
    options = dict(local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen, **options)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen, device_map={'': 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16), **options)
    if getattr(model, 'is_loaded_in_4bit', False):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'],
                                            lora_dropout=0.05, task_type='CAUSAL_LM'))
    model.config.use_cache = False

    def encode(row):
        prefix = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': EXTRACTION}, {'role': 'user', 'content': row['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        reply = tokenizer.encode(row['response'], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(ids) + len(reply) > 2048:
            raise ValueError('Training example exceeds the 2048-token budget.')
        return dict(input_ids=ids + reply, attention_mask=[1] * (len(ids) + len(reply)),
                    labels=[-100] * len(ids) + reply)

    trainer = Trainer(
        model=model, train_dataset=[encode(row) for row in train],
        eval_dataset=[encode(row) for row in dev],
        args=TrainingArguments(output_dir=str(args.output), per_device_train_batch_size=1,
                               per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                               num_train_epochs=1, learning_rate=2e-4, eval_strategy='epoch',
                               save_strategy='no', report_to=[]),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)


if __name__ == '__main__':
    main()
