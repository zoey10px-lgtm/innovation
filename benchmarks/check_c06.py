"""Run the C06 Qwen reviewer on two pinned benchmark excerpts."""
import argparse
import csv
from hashlib import file_digest
import json
from pathlib import Path
import time
from types import SimpleNamespace

from innovation.__main__ import Models
from innovation.assess import review_facts, sentence_windows
from innovation.pdf_parser import read_pdf


BENCHMARKS = Path(__file__).resolve().parent
KEY = 'APCTT-C06'
CASES = ((3, 1, False), (102, 8, True))


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
    for case_id, sentence_id, expected in CASES:
        case = json.loads((BENCHMARKS / 'cases' / f'{case_id}.json').read_text(encoding='utf-8'))
        source = sources[case_id]
        path = args.pdf_dir / source['original_filename']
        with path.open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != source['sha256']:
                parser.error(f'Benchmark PDF checksum differs: {path}')
        sentence = list(sentence_windows(read_pdf(str(path), 'report')))[sentence_id - 1]
        if sentence.page != 1:
            parser.error(f'Expected sentence {sentence_id} on page 1: {path}')
        prepared.append((case_id, sentence_id, expected, case['target'], sentence))

    import torch
    torch.set_num_threads(4)
    models = Models(SimpleNamespace(qwen=str(args.qwen), qwen_device=args.qwen_device, qwen_adapter=None))
    print(f'Loading Qwen on {args.qwen_device}...', flush=True)
    _ = models.qwen
    matched = 0
    for case_id, sentence_id, expected, target, sentence in prepared:
        facts = review_facts({KEY: sentence}, {KEY: criterion}, target, {KEY: str(sentence_id)},
                             models.generate, time.monotonic() + 180, {})
        flags = facts.get(KEY)
        accepted = all(flags) if flags is not None else None
        matched += accepted is expected
        print(json.dumps(dict(case=case_id, sentence_id=sentence_id, expected=expected,
                              accepted=accepted, facts=flags)), flush=True)
    print(f'Expected decisions matched: {matched}/{len(prepared)}', flush=True)
    return 0 if matched == len(prepared) else 1


if __name__ == '__main__':
    raise SystemExit(main())
