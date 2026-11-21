"""Run one offline assessment within a subprocess wall-time budget."""
import argparse
from functools import cached_property
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .assess import FactReviewRequest, assess
from .pdf_parser import Page, read_pdf


def fact_token_constraint(tokenizer, count, prompt_length, eos_token_id):
    """Permit only a JSON array of exactly count binary decisions."""
    bracket, zero, one, comma, closing = tokenizer(
        ['[', '0', '1', ',', ']'], add_special_tokens=False)['input_ids']
    if tokenizer.decode(bracket + zero + comma + one + closing, skip_special_tokens=True) != '[0,1]':
        raise ValueError('Tokenizer cannot encode the constrained fact response')
    if len(zero) != 1 or len(one) != 1:
        raise ValueError('Tokenizer must encode each review decision as one token')
    schedule = [[token] for token in bracket]
    for index in range(count):
        if index:
            schedule.extend([token] for token in comma)
        schedule.append([zero[0], one[0]])
    schedule.extend([token] for token in closing)
    schedule.append([eos_token_id])

    def allowed_tokens(_batch_id, input_ids):
        return schedule[len(input_ids) - prompt_length]

    return allowed_tokens, len(schedule)


class Models:
    def __init__(self, args):
        self.args = args

    @cached_property
    def retrieval(self):
        from .two_stage_retrieval import TwoStageRetriever
        return TwoStageRetriever(self.args.encoder, self.args.reranker,
                                 device=self.args.qwen_device, candidate_k=16, final_k=5)

    @cached_property
    def nli(self):
        from .offline_nli import OfflineNLI
        return OfflineNLI(self.args.nli)

    @cached_property
    def qwen(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        options = dict(local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(self.args.qwen, **options)
        device = self.args.qwen_device
        if device.startswith('cuda'):
            import gc

            self.__dict__.pop('retrieval', None)
            gc.collect()
            torch.cuda.empty_cache()
            from transformers import BitsAndBytesConfig
            index = torch.device(device).index or 0
            options.update(device_map={'': index}, dtype=torch.bfloat16,
                           quantization_config=BitsAndBytesConfig(
                               load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
        else:
            options['dtype'] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(self.args.qwen, **options)
        if self.args.qwen_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.args.qwen_adapter, local_files_only=True)
        return tokenizer, model.eval()

    def generate(self, instruction, data, deadline):
        import torch
        tokenizer, model = self.qwen
        prompt = tokenizer.apply_chat_template(
            [dict(role='system', content=instruction),
             dict(role='user', content=data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        length = inputs.input_ids.shape[1]
        remaining = deadline - time.monotonic()
        if remaining <= 2:
            raise TimeoutError('Qwen generation budget')
        if length > 8192:
            raise ValueError('Candidate prompt exceeded 8192 tokens')
        generation_options = dict(max_new_tokens=160, do_sample=False, max_time=remaining - 1)
        if isinstance(data, FactReviewRequest):
            eos = model.generation_config.eos_token_id
            if isinstance(eos, (list, tuple)):
                eos = eos[0] if eos else None
            if eos is None:
                eos = tokenizer.eos_token_id
            constraint, limit = fact_token_constraint(tokenizer, data['check_count'], length, eos)
            generation_options.update(prefix_allowed_tokens_fn=constraint, max_new_tokens=limit)
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_options)
        return tokenizer.decode(generated[0, length:], skip_special_tokens=True)


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    deadline = args.deadline
    checklist = json.loads(Path(args.checklist).read_text(encoding='utf-8'))
    criteria = {item['criterion_id']: item['description'].partition(': ')[2]
                for item in checklist['criteria'] if item['criterion_id'] in
                {f'APCTT-C{level:02d}' for level in range(1, 10)}}
    if len(criteria) != 9 or not all(criteria.values()):
        raise ValueError('Expected APCTT criteria C01 through C09.')
    if args.pages_json:
        page_data = json.loads(Path(args.pages_json).read_text(encoding='utf-8'))
        pages = [Page('report', row['page'], row['text']) for row in page_data]
    else:
        pages = read_pdf(args.pdf, 'report', remaining_ms=lambda: max(0, int((deadline - time.monotonic()) * 1000)))
    models = Models(args)

    def check(pairs):
        if deadline - time.monotonic() <= 2:
            raise TimeoutError('NLI verification budget')
        return models.nli.predict_batch(pairs, batch_size=4)

    result = assess(args.target, pages, criteria, lambda q, t: models.retrieval.retrieve(q, t),
                    models.generate, check, deadline)
    if args.qwen_device == 'cuda' and 'qwen' in models.__dict__:
        result['peak_cuda_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 2)
        result['peak_cuda_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pdf')
    source.add_argument('--pages-json', help='Pre-extracted pages for model-only evaluation')
    for name in ('target', 'encoder', 'reranker', 'qwen', 'nli'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--checklist', default='checklist.json')
    parser.add_argument('--qwen-device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--qwen-adapter', type=Path)
    parser.add_argument('--threads', default=4, type=int)
    parser.add_argument('--timeout', default=180, type=int)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.deadline is not None:
        try:
            result = run(args)
        except TimeoutError as error:
            result = dict(status='timeout', predicted_trl=None, error=str(error))
    else:
        start = time.monotonic()
        try:
            process = subprocess.run(
                [sys.executable, '-m', 'innovation', *sys.argv[1:], '--deadline', str(start + args.timeout)],
                capture_output=True, text=True, encoding='utf-8', timeout=args.timeout,
                env=dict(os.environ, PYTHONUTF8='1'))
            if process.returncode:
                result = dict(status='error', predicted_trl=None, error=process.stderr[-3000:])
            else:
                result = json.loads(process.stdout)
        except subprocess.TimeoutExpired:
            result = dict(status='timeout', predicted_trl=None)
        result['seconds'] = round(time.monotonic() - start, 3)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
