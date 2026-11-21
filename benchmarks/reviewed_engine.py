"""Run the pinned model stack on reviewed targets in a preprocessed data ZIP."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('model_store', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    profile = json.loads(Path('model-profile.json').read_text())
    names = ['qwen', 'encoder', 'reranker', 'nli']
    ids = ['qwen3-4b-instruct-2507-nf4', 'scincl', 'bge-reranker-v2-m3', 'deberta-v3-large-zeroshot-v2.0-c']
    paths = {m['id']: args.model_store / m['repo'] / m['revision'] for m in profile}
    model_args = []
    for name, model_id in zip(names, ids):
        if not paths[model_id].is_dir():
            parser.error(f'Missing {name} checkpoint: {paths[model_id]}')
        model_args.extend(['--' + name, str(paths[model_id])])
    with ZipFile(args.dataset) as archive:
        cases = [r for r in map(json.loads, archive.open('checked_evidence.jsonl')) if r['reviewed_trl']]
        needed = {r['doc_id'] for r in cases}
        pages = {r['doc_id']: r['pages'] for r in map(json.loads, archive.open('preprocessed_pages.jsonl'))
                 if r['doc_id'] in needed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp, args.output.open('x', encoding='utf-8') as output:
        source = Path(temp) / 'pages.json'
        correct = 0
        for i, case in enumerate(cases, 1):
            source.write_text(json.dumps([{'page': p['page'], 'text': p['text']}
                                          for p in pages[case['doc_id']]]), encoding='utf-8')
            process = subprocess.run([sys.executable, '-m', 'innovation', '--pages-json', str(source),
                                      '--target', case['target'], *model_args], capture_output=True, text=True)
            result = json.loads(process.stdout) if process.returncode == 0 else dict(status='error', error=process.stderr[-2000:])
            correct += result.get('predicted_trl') == case['reviewed_trl']
            output.write(json.dumps(dict(doc_id=case['doc_id'], target=case['target'],
                                         reference=case['reviewed_trl'], prior=case['known_prior_use'],
                                         result=result)) + '\n')
            output.flush()
            print(f'{i}/{len(cases)}: {correct}/{i} exact, latest status {result["status"]}', flush=True)


if __name__ == '__main__':
    main()
