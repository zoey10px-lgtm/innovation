"""Evaluate full-text classifiers against weak labels and independently reviewed targets.

Run from the repository root with the extracted dataset directory as the argument.
Only original PDF page text enters the predictors. Development uses weak labels.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


LEVEL = re.compile(r'\b(?:TRLs?|technology\s+readiness\s+levels?)\s*(?:of|is|was|at|[:=(\-–])?\s*([1-9])\b', re.I)
FUTURE = re.compile(r'\b(?:goal|target of|objective|aim|will|expected|planned|to achieve|to reach)\b', re.I)
CURRENT = re.compile(r'\b(?:achieved|demonstrated|currently|current|reached|matured|assessed|validated|attained)\b', re.I)


def load(path):
    with (path / 'inventory_all.csv').open(encoding='utf-8-sig') as stream:
        rows = {r['doc_id']: r for r in csv.DictReader(stream)}
    docs = {}
    groups = {}
    fingerprints = {}
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            key = parents[key]
        return key

    for line in (path / 'preprocessed_pages.jsonl').open():
        doc = json.loads(line)
        key = doc['doc_id']
        docs[key] = '\n'.join(p['text'] for p in doc['pages'])
        group = rows[key]['related_report_group'] or key
        groups[key] = group
        # Join identical normalized bodies, including duplicates with different IDs.
        normalized = ' '.join(docs[key].casefold().split())
        fingerprint = hashlib.sha256(normalized.encode()).digest()
        other = fingerprints.setdefault(fingerprint, group)
        parents[root(group)] = root(other)
    # Link near-duplicate versions before partitioning. No labels enter this step.
    vector = TfidfVectorizer(ngram_range=(2, 2), min_df=2, max_features=120000)
    matrix = vector.fit_transform(docs.values())
    similarities = (matrix @ matrix.T).toarray()
    keys = list(docs)
    for i, j in zip(*np.where(np.triu(similarities, 1) >= .85)):
        parents[root(groups[keys[i]])] = root(groups[keys[j]])
    reviewed = [json.loads(line) for line in (path / 'checked_evidence.jsonl').open()]
    return rows, docs, {key: root(group) for key, group in groups.items()}, reviewed


def reported_level(text):
    scores = Counter()
    for match in LEVEL.finditer(text):
        prefix = re.split(r'[.!?;]', text[max(0, match.start()-110):match.start()])[-1]
        if FUTURE.search(prefix):
            continue
        scores[int(match[1])] += 4 if CURRENT.search(prefix) else 1
    return scores.most_common(1)[0][0] if scores else None


def fit_predict(train, test, docs, rows):
    model = make_pipeline(TfidfVectorizer(sublinear_tf=True, max_features=50000,
                                         ngram_range=(1, 2), min_df=2),
                          LinearSVC(class_weight='balanced', random_state=32))
    model.fit([docs[i] for i in train], [int(rows[i]['source_trl']) for i in train])
    return model.predict([docs[i] for i in test])


def run(path):
    rows, docs, groups, reviewed = load(path)
    excluded = {groups[r['doc_id']] for r in reviewed}
    excluded.update(groups[i] for i in docs if rows[i]['known_prior_use'] == 'True')
    weak = np.array([i for i in docs if groups[i] not in excluded and
                     rows[i]['source_label_type'] == 'weak_mention' and rows[i]['source_trl']])
    predictions = []

    def record(key, target, actual, predicted, split, fold):
        declared = reported_level(docs[key])
        predictions.append(dict(doc_id=key, target=target, group=groups[key], split=split,
                                fold=fold, actual=int(actual), svm=int(predicted),
                                reported_or_svm=declared or int(predicted)))

    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(weak, groups=[groups[i] for i in weak])):
        for key, pred in zip(weak[test], fit_predict(weak[train], weak[test], docs, rows)):
            record(key, '', rows[key]['source_trl'], pred, 'weak_oof', fold)
    cases = [r for r in reviewed if r['reviewed_trl'] is not None]
    prior_groups = {groups[i] for i in docs if rows[i]['known_prior_use'] == 'True'}
    for r, pred in zip(cases, fit_predict(weak, [r['doc_id'] for r in cases], docs, rows)):
        split = 'reviewed_prior' if groups[r['doc_id']] in prior_groups else 'reviewed_no_known_prior'
        record(r['doc_id'], r['target'], r['reviewed_trl'], pred, split, None)
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    predictions = run(args.data)
    args.output.write_text(json.dumps(predictions, indent=2) + '\n')
    for split in sorted({r['split'] for r in predictions}):
        part = [r for r in predictions if r['split'] == split]
        print(split, len(part), {m: sum(r[m] == r['actual'] for r in part)
                                 for m in ('svm', 'reported_or_svm')})
