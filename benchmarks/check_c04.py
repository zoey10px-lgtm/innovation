"""Run the same isolated C04 reviewer on two exact benchmark source spans."""
import argparse
import csv
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path
import time
from types import SimpleNamespace

from innovation.__main__ import Models
from innovation.assess import review_facts, sentence_windows
from innovation.pdf_parser import read_pdf


BENCHMARKS = Path(__file__).resolve().parent
KEY = 'APCTT-C04'
CASES = ((3, 515, 47, False, False), (101, 157, 10, True, True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf-dir', required=True, type=Path)
    parser.add_argument('--qwen', required=True, type=Path)
    parser.add_argument('--checklist', type=Path, default=BENCHMARKS.parent / 'checklist.json')
    parser.add_argument('--qwen-device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()

    checklist = json.loads(args.checklist.read_text(encoding='utf-8'))
    criterion = next(row['description'].partition(': ')[2]
                     for row in checklist['criteria'] if row['criterion_id'] == KEY)
    with (BENCHMARKS / 'pdf_sources.csv').open(encoding='utf-8-sig', newline='') as stream:
        sources = {int(row['case_id']): row for row in csv.DictReader(stream)}
    prepared = []
    for case_id, sentence_id, page, expected, retry in CASES:
        case = json.loads((BENCHMARKS / 'cases' / f'{case_id}.json').read_text(encoding='utf-8'))
        source = sources[case_id]
        path = args.pdf_dir / source['original_filename']
        with path.open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != source['sha256']:
                parser.error(f'Benchmark PDF checksum differs: {path}')
        sentence = list(sentence_windows(read_pdf(str(path), 'report')))[sentence_id - 1]
        if sentence.page != page or not sentence.extended_quote:
            parser.error(f'Expected sentence pair {sentence_id} on page {page}: {path}')
        sentence = replace(sentence, quote=sentence.extended_quote, context=sentence.extended_context)
        prepared.append((case_id, sentence_id, expected, retry, case, sentence))

    import torch
    torch.set_num_threads(4)
    models = Models(SimpleNamespace(qwen=str(args.qwen), qwen_device=args.qwen_device, qwen_adapter=None))
    start = time.monotonic()
    print(f'Loading Qwen on {args.qwen_device}...', flush=True)
    _ = models.qwen
    print(f'Qwen load: {time.monotonic() - start:.2f} s', flush=True)
    matched = 0
    for case_id, sentence_id, expected, retry, case, sentence in prepared:
        start = time.monotonic()
        trace = {}
        try:
            facts = review_facts({KEY: sentence}, {KEY: criterion}, case['target'], {KEY: str(sentence_id)},
                                 models.generate, start + 180, trace, expanded=retry)
        except TimeoutError:
            facts = {}
            trace['error'] = 'timeout'
        flags = facts.get(KEY)
        accepted = all(flags) if flags is not None else None
        matched += accepted is expected
        print(json.dumps(dict(case=case_id, page=sentence.page, sentence_id=sentence_id,
                              expected=expected, accepted=accepted, facts=flags,
                              seconds=round(time.monotonic() - start, 3), trace=trace)), flush=True)
    print(f'Expected decisions matched: {matched}/{len(prepared)}', flush=True)
    return 0 if matched == len(prepared) else 1


if __name__ == '__main__':
    raise SystemExit(main())
