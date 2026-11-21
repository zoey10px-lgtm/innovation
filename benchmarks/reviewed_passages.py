"""Run the CPU passage profile on every documented assessment target."""
import argparse
import json
from pathlib import Path

from innovation.passage_profile import assess_passages
from innovation.pdf_parser import Page


def evaluate(data):
    documents = {r['doc_id']: r for r in map(json.loads, (data/'preprocessed_pages.jsonl').open())}
    for case in map(json.loads, (data/'checked_evidence.jsonl').open()):
        if case['reviewed_trl'] is None:
            continue
        doc = documents[case['doc_id']]
        pages = [Page(doc['doc_id'], p['page'], p['text']) for p in doc['pages']]
        result = assess_passages(case['target'], pages)
        yield dict(doc_id=case['doc_id'], target=case['target'], known_prior_use=case['known_prior_use'],
                   reference=case['reviewed_trl'], reference_page=case['physical_pdf_page'], **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(list(evaluate(args.data)), indent=2) + '\n')
