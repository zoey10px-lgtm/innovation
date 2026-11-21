"""Compare both engines on the same 12 reviewed development targets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
# Includes earlier development errors; no final evaluation IDs or labels.
IDS = ('ARCHIVE-TechPort_125310_381274', 'ARCHIVE-TechPort_96541_315818', 'ARCHIVE-NTRS_20140005007',
       'ARCHIVE-NTRS_20230012183', 'ARCHIVE-NTRS_20100039424', 'ARCHIVE-NTRS_20110015293',
       'ARCHIVE-NTRS_20050010075', 'ARCHIVE-NTRS_20205005704', 'ARCHIVE-NTRS_20100027418',
       'HISTORY-02', 'HISTORY-04', 'HISTORY-06')


def load_cases(dataset):
    with ZipFile(dataset) as archive:
        cases = {c['doc_id']: c for c in map(json.loads, archive.open('checked_evidence.jsonl'))
                 if c['doc_id'] in IDS and c['reviewed_trl']}
        pages = {c['doc_id']: c for c in map(json.loads, archive.open('preprocessed_pages.jsonl'))
                 if c['doc_id'] in IDS}
    return [(cases[i], pages[i]) for i in IDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--pdf-dir', type=Path, help='Original development PDFs, by filename')
    parser.add_argument('--output', type=Path, default=Path('page-comparison.jsonl'))
    args = parser.parse_args()
    store = Path(os.environ['APCTT_MODEL_STORE'])
    flags = {'scincl': '--encoder', 'bge-reranker-v2-m3': '--reranker',
             'qwen3-4b-instruct-2507-nf4': '--qwen', 'deberta-v3-large-zeroshot-v2.0-c': '--nli'}
    model_args = []
    for m in json.loads((ROOT / 'model-profile.json').read_text()):
        path = store / m['repo'] / m['revision']
        if not path.is_dir():
            parser.error(f'Missing checkpoint: {path}')
        model_args.extend([flags[m['id']], str(path.resolve())])
    cases = load_cases(args.dataset)
    scores = {'legacy': 0, 'pages': 0}
    within = dict.fromkeys(scores, 0)
    with tempfile.TemporaryDirectory() as tmp, args.output.open('x', encoding='utf8') as out:
        for case, document in cases:
            if args.pdf_dir:
                import hashlib
                source = args.pdf_dir / case['filename']
                if hashlib.sha256(source.read_bytes()).hexdigest() != document['pdf_sha256']:
                    raise ValueError(f'PDF hash mismatch: {source}')
                inputs = ['--pdf', str(source.resolve())]
            else:
                source = Path(tmp) / 'input.json'
                source.write_text(json.dumps(document['pages']), encoding='utf8')
                inputs = ['--pages-json', str(source)]
            for mode in scores:
                cmd = [sys.executable, '-m', 'innovation', *inputs, '--target', case['target'], *model_args]
                if mode == 'legacy':
                    cmd.append('--legacy')
                proc = subprocess.run(cmd, cwd=ROOT, text=True, encoding='utf8', capture_output=True)
                result = json.loads(proc.stdout) if proc.returncode == 0 else dict(status='error', error=proc.stderr[-2000:])
                timely = result.get('status') == 'completed' and result.get('seconds', 181) <= 180
                within[mode] += timely
                scores[mode] += timely and result.get('predicted_trl') == case['reviewed_trl']
                out.write(json.dumps(dict(id=case['doc_id'], mode=mode, reference=case['reviewed_trl'], result=result)) + '\n')
                out.flush()
                print(case['doc_id'], mode, result['status'], result.get('predicted_trl'), result.get('seconds'), flush=True)
    print('Exact and within budget:', scores, '/12 each. Completed within 180s:', within)
    print('Page engine gate:', 'PASS' if scores['pages'] >= 9 and within['pages'] == 12 else 'NOT PASSED')


if __name__ == '__main__':
    main()
