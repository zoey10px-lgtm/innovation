"""CPU-only, target-specific stage selection from page-grounded evidence."""
import argparse
from collections import Counter
import json
import math
import re

from .assess import COMMON_TARGET_WORDS, COMPLETED, PLANNED, TOKEN, sentence_windows
from .pdf_parser import read_pdf


# Each stage requires an activity and its setting or object in the same passage.
SIGNALS = {
    1: (r'principle|phenomen|theoretic|fundamental', r'observ|studied|analysis|theory'),
    2: (r'concept|design|propos', r'application|method|approach|instrument|technique'),
    3: (r'proof.of.concept|feasibility|critical function', r'measur|experiment|shown|demonstrat|result'),
    4: (r'breadboard|laboratory|bench.test', r'tested|validat|measured|result'),
    5: (r'representative environment|simulated environment|thermal.vacuum|relevant environment',
        r'component|breadboard|subsystem'),
    6: (r'prototype|engineering model|protoflight', r'field.test|chamber|relevant environment|flight.test'),
    7: (r'flight experiment|flight demonstration|prototype|demonstrator', r'in.flight|in.orbit|on.orbit|operational environment|field trial'),
    8: (r'qualification|qualified|acceptance test', r'flight|final system|requirements'),
    9: (r'in service|operational use|deployed|operated|operations', r'for .{0,20}(?:years|months)|since \d{4}|operational mission'),
}
SIGNALS = {k: tuple(re.compile(p, re.I | re.S) for p in v) for k, v in SIGNALS.items()}
STAGE = re.compile(r'\b(?:TRL|technology\s+readiness\s+level)\s*[-:(=]?\s*([1-9])\b', re.I)
STOP = COMMON_TARGET_WORDS | set('its was that this with are and has have project development using based assessment assessed'.split())


def assess_passages(target, pages):
    words = set(TOKEN.findall(target.casefold())) - STOP
    passages = list(sentence_windows(pages))
    frequency = Counter()
    for passage in passages:
        frequency.update(words.intersection(TOKEN.findall(passage.context.casefold())))
    weights = {w: math.log1p(len(passages) / (1 + frequency[w])) for w in words}
    best = None
    for passage in passages:
        quote = ' '.join(passage.quote.lower().split())
        overlap = words.intersection(TOKEN.findall(passage.context.lower()))
        if not overlap:
            continue
        relevance = sum(weights[w] for w in overlap) / max(sum(weights.values()), 1)
        declared = {int(match[1]) for match in STAGE.finditer(quote)}
        for level, patterns in SIGNALS.items():
            if not all(pattern.search(quote) for pattern in patterns):
                continue
            completed = bool(COMPLETED.search(quote))
            if level >= 3 and PLANNED.search(quote) and not completed:
                continue
            score = 3 * relevance + 2 * (level in declared) + .6 * completed + .05 * level
            if best is None or score > best[0]:
                best = (score, level, passage)
    if best is None:
        return dict(status='abstained', predicted_trl=None, evidence={})
    _, level, passage = best
    evidence = {f'APCTT-C{level:02d}': dict(document_id=passage.document_id, page=passage.page, quote=passage.quote)}
    return dict(status='completed', predicted_trl=level, evidence=evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    print(json.dumps(assess_passages(args.target, read_pdf(args.pdf, 'report'))))
