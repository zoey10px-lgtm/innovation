"""Run identical offline CLI settings across the benchmark cases."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('benchmarks/cases'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('model_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    options = args.model_args
    if options and options[0] == '--':
        options = options[1:]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.cases.glob('*.json'), key=lambda p: int(p.stem)):
        case = json.loads(path.read_text(encoding='utf-8'))
        process = subprocess.run([sys.executable, '-m', 'innovation', '--pdf', case['pdf'],
                                  '--target', case['target'], *options],
                                 capture_output=True, text=True, encoding='utf-8')
        if process.returncode:
            result = dict(status='error', predicted_trl=None, error=process.stderr[-3000:])
        else:
            result = json.loads(process.stdout)
        record = dict(id=case['id'], target=case['target'], reference=case['reference'], **result)
        (args.output / path.name).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        rows.append(record)
        print(record['id'], record['reference'], record['predicted_trl'],
              record['status'], record.get('seconds'), flush=True)
    if not rows:
        parser.error('No benchmark cases found')
    fields = ['id', 'target', 'reference', 'predicted_trl', 'status', 'seconds', 'peak_cuda_reserved_mib', 'peak_cuda_allocated_mib']
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(sum(row['reference'] == row['predicted_trl'] for row in rows), '/', len(rows))


if __name__ == '__main__':
    main()
