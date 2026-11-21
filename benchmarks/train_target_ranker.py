"""Train and evaluate a CPU passage-and-level ranker on reviewed targets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from benchmarks.generalization import LEVEL, load


def words(text):
    return set(re.findall(r'[a-z0-9]{2,}', text.lower())) - ENGLISH_STOP_WORDS


def passages(pages):
    for page in pages:
        tokens = list(re.finditer(r'\S+', page['text']))
        for start in range(0, len(tokens), 42):
            end = min(start + 84, len(tokens))
            yield page['page'], page['text'][tokens[start].start():tokens[end-1].end()]
            if end == len(tokens):
                break


class Ranker:
    def fit(self, cases, documents, groups, use_target, local_mentions=False):
        self.local_mentions = local_mentions
        self.use_target = use_target
        self.vector = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                      stop_words='english')
        excerpts = [c['evidence_excerpt'] for c in cases]
        vectors = self.vector.fit_transform(excerpts)
        def prototypes(excluded_group=None):
            matrix = np.zeros((9, vectors.shape[1]))
            for level in range(1, 10):
                indices = [i for i, c in enumerate(cases) if c['reviewed_trl'] == level
                           and groups[c['doc_id']] != excluded_group]
                if indices:
                    matrix[level-1] = np.asarray(vectors[indices].mean(axis=0)).ravel()
            return normalize(matrix)

        features, labels, weights = [], [], []
        for case in cases:
            self.prototypes = prototypes(groups[case['doc_id']])
            spans, matrix = self.features(case['target'], documents[case['doc_id']])
            anchor = words(case['evidence_excerpt'])
            correct_page = [i for i,(page, _) in enumerate(spans) if page == case['physical_pdf_page']]
            i = max(correct_page, key=lambda j: len(anchor & words(spans[j][1])))
            positive = i * 9 + case['reviewed_trl'] - 1
            negative = np.flatnonzero(np.arange(len(matrix)) != positive)
            # Exclude overlapping windows of the same level on the annotated page.
            negative = [j for j in negative if j % 9 != positive % 9 or j//9 not in correct_page]
            features.extend([matrix[positive], *matrix[negative]])
            labels.extend([1] + [0]*len(negative))
            weights.extend([1] + [1/len(negative)]*len(negative))
        self.prototypes = prototypes()
        self.rank = LogisticRegression(max_iter=500).fit(features, labels, sample_weight=weights)
        return self

    def features(self, target, pages):
        spans = list(passages(pages))
        texts = [s for _, s in spans]
        sim = self.vector.transform(texts) @ self.prototypes.T
        query = words(target) if self.use_target else set()
        terms = [words(s) for s in texts]
        freq = Counter(w for term in terms for w in term & query)
        idf = {w: np.log1p(len(terms)/(1+freq[w])) for w in query}
        total = sum(idf.values()) or 1
        rows = []
        for i, text in enumerate(texts):
            relevance = sum(idf[w] for w in terms[i] & query)/total
            levels = Counter()
            local_overlap = Counter()
            for match in LEVEL.finditer(text):
                level = int(match[1])
                levels[level] += 1
                # Only associate the printed stage with names in its own clause.
                left = re.split(r'[.!?;]\s+|\n\s*\n', text[:match.start()])[-1]
                right = re.split(r'[.!?;]\s+|\n\s*\n', text[match.end():], maxsplit=1)[0]
                near = words(' '.join(left.split()[-18:] + right.split()[:18]))
                value = sum(idf[w] for w in near & query) / total
                local_overlap[level] = max(local_overlap[level], value)
            for level in range(1, 10):
                explicit = int(level in levels)
                affinity = sim[i,level-1]
                row = [relevance, explicit, explicit*relevance, affinity, affinity*relevance]
                if self.local_mentions:
                    row.append(local_overlap[level])
                rows.append(row)
        return spans, np.asarray(rows)

    def predict(self, target, pages):
        spans, x = self.features(target, pages)
        index = int(self.rank.decision_function(x).argmax())
        page, span = spans[index//9]
        return dict(level=index%9+1, page=page, span=span)


def run(data):
    rows, _, groups, reviewed = load(data)
    prior = {groups[k] for k in groups if rows[k]['known_prior_use']=='True'}
    cases = [r for r in reviewed if r['reviewed_trl']]
    documents = {r['doc_id']:r['pages'] for r in map(json.loads, (data/'preprocessed_pages.jsonl').open())}
    out = []
    for group in sorted({groups[r['doc_id']] for r in cases}):
        train = [r for r in cases if groups[r['doc_id']] in prior - {group}]
        tests = [r for r in cases if groups[r['doc_id']]==group]
        results = [dict(target=r['target'], group=group, prior=group in prior,
                        actual=r['reviewed_trl'], actual_page=r['physical_pdf_page']) for r in tests]
        for key, use_target, local in [('without_target', False, False), ('with_target', True, False),
                                       ('within_clause', True, True)]:
            model = Ranker().fit(train, documents, groups, use_target, local)
            for case, result in zip(tests, results):
                result[key] = model.predict(
                    case['target'], documents[case['doc_id']])
        out.extend(results)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    results = run(args.data)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    for prior in [False, True]:
        subset = [r for r in results if r['prior']==prior]
        print('prior', prior, len(subset), {key:sum(r[key]['level']==r['actual'] for r in subset)
                                           for key in ('with_target','without_target','within_clause')})
