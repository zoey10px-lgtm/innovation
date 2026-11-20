"""Prepare Qwen extraction examples from independently labeled evidence pairs."""
import argparse
import hashlib
import json
from pathlib import Path


def convert(source, output):
    rows = []
    with source.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['label'] not in (0, 1):
                raise ValueError('Evidence label must be binary')
            key = row['criterion_id']
            if key not in {f'APCTT-C{level:02d}' for level in range(1, 10)}:
                continue
            query = (f"Target: {row['target']}\nCriteria:\n{key}: {row['criterion']}\n\n"
                     f"Report excerpts:\nSENTENCE ID 1\nCENTER: {row['passage']}\n\n"
                     'Return only a JSON object of existing IDs.')
            rows.append(dict(group=row['group'], label_origin=row.get('label_origin'),
                             prompt=query, response=json.dumps({key: '1'} if row['label'] else {})))
    with output.open('w', encoding='utf-8') as handle:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=handle)
    print(json.dumps({'examples': len(rows), 'groups': len({row['group'] for row in rows}),
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
