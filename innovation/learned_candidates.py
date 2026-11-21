"""Nominate a sentence per criterion from a frozen supervised passage ranker."""
from collections import Counter
import json
import math
import re

LEVEL = re.compile(r'\b(?:TRLs?|technology\s+readiness\s+levels?)\s*(?:of|is|was|at|[:=(\-–])?\s*([1-9])\b', re.I)


class LearnedCandidates:
    def __init__(self, path):
        data = json.loads(path.read_text(encoding='utf-8'))
        self.stop = set(data['stop'])
        self.vocab = data['vocabulary']
        self.idf = data['idf']
        self.prototypes = data['prototypes']
        self.coef = data['coef']

    def rank(self, target, sentences):
        query = set(re.findall(r'[a-z0-9]{2,}', target.lower())) - self.stop
        term_sets = [set(re.findall(r'[a-z0-9]{2,}', s.quote.lower())) & query for s in sentences]
        frequency = Counter(w for terms in term_sets for w in terms)
        weights = {w: math.log1p(len(sentences)/(1+frequency[w])) for w in query}
        total = sum(weights.values()) or 1
        scores = {}
        for index, sentence in enumerate(sentences):
            text = sentence.quote.lower()
            terms = [w for w in re.findall(r'\b\w\w+\b', text) if w not in self.stop]
            counts = Counter(terms + [a+' '+b for a,b in zip(terms,terms[1:])])
            vector = {self.vocab[w]: (1+math.log(n))*self.idf[self.vocab[w]]
                      for w,n in counts.items() if w in self.vocab}
            norm = math.sqrt(sum(v*v for v in vector.values())) or 1
            relevance = sum(weights[w] for w in term_sets[index])/total
            levels = {int(m[1]) for m in LEVEL.finditer(text)}
            for level, prototype in enumerate(self.prototypes, 1):
                affinity = sum(v*prototype[j] for j,v in vector.items())/norm
                explicit = int(level in levels)
                features = (relevance, explicit, explicit*relevance, affinity, affinity*relevance)
                score = sum(c*x for c,x in zip(self.coef,features))
                if level not in scores or score > scores[level][0]:
                    scores[level] = (score,index)
        return {f'APCTT-C{level:02d}': str(index+1) for level,(_,index) in scores.items()}
