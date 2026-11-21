"""APCTT participant-kit assessment entry point."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from .__main__ import Models
from .assess import assess
from .learned_candidates import LearnedCandidates
from .pdf_parser import read_pdf


class AssessmentModule:
    def load(self, config, asset_root):
        self.checklist = config['checklist']
        self.settings = config.get('module', {}).get('settings', {})
        self.device = str(config.get('compute', {}).get('device', 'cuda'))
        self.model_store = Path(os.environ['APCTT_MODEL_STORE'])
        profile = json.loads((Path(__file__).resolve().parent.parent / 'model-profile.json')
                             .read_text(encoding='utf-8'))
        self.paths = {m['id']: self.model_store / m['repo'] / m['revision'] for m in profile}
        self.asset_root = Path(asset_root)
        ranker = self.asset_root / "trained_ranker.json"
        self.ranker = LearnedCandidates(ranker) if ranker.is_file() else None

    def assess(self, case, context):
        import torch

        time_left = context.remaining_ms() / 1000
        deadline = time.monotonic() + max(time_left, 0)
        torch.set_num_threads(int(self.settings.get('cpu_threads', 4)))
        ids = [row['criterion_id'] for row in self.checklist['criteria']]
        criteria = {row['criterion_id']: row['description'].partition(': ')[2]
                    for row in self.checklist['criteria']
                    if row['criterion_id'] in {f'APCTT-C{level:02d}' for level in range(1, 10)}}
        if len(criteria) != 9 or not all(criteria.values()):
            raise ValueError('Expected APCTT criteria C01 through C09.')
        target = case['assessment_target']
        target_text = str(target.get('description') or target['target_id'])
        args = SimpleNamespace(
            encoder=self.paths['scincl'], reranker=self.paths['bge-reranker-v2-m3'],
            qwen=self.paths['qwen3-4b-instruct-2507-nf4'],
            nli=self.paths['deberta-v3-large-zeroshot-v2.0-c'],
            qwen_adapter=None, qwen_device=self.device,
        )
        models = Models(args)
        pages = [page for document in case['documents'] for page in
                 read_pdf(document['path'], document['document_id'], remaining_ms=context.remaining_ms)]
        try:
            outcome = assess(target_text, pages, criteria, lambda q, t: models.retrieval.retrieve(q, t),
                             models.generate, lambda pairs: models.nli.predict_batch(pairs, batch_size=4),
                             deadline, nominate=self.ranker.rank if self.ranker else None)
        except TimeoutError:
            outcome = {'status': 'failed', 'predicted_trl': None, 'evidence': {}}
        evidence = outcome['evidence']
        references = {key: [] for key in ids}
        for key, row in evidence.items():
            references[key] = [dict(document_id=row['document_id'], page=row['page'],
                                    supporting_span=row['quote'])]
        return {
            'schema_version': case['schema_version'], 'case_id': case['case_id'],
            'checklist_version': self.checklist['version'], 'assessment_target': dict(target),
            'assessment_status': outcome['status'], 'predicted_trl': outcome['predicted_trl'],
            'abstention_reason': 'insufficient evidence' if outcome['status'] == 'abstained' else None,
            'error_code': 'ASSESSMENT_TIME_BUDGET' if outcome['status'] == 'failed' else None,
            'criterion_status': {key: 'met' if key in evidence else 'insufficient evidence' for key in ids},
            'evidence_reference': references,
            'missing_evidence': {key: 'No accepted supporting sentence.' for key in ids if key not in evidence},
        }

    def reset_case(self):
        pass

    def close(self):
        pass
