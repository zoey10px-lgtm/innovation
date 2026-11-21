import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from innovation.adapter import AssessmentModule


class KitAdapterTest(unittest.TestCase):
    def test_kit_output_maps_verbatim_citation_to_document(self):
        obj = AssessmentModule()
        checklist = json.loads(Path('checklist.json').read_text(encoding='utf-8'))
        context = SimpleNamespace(remaining_ms=lambda: 180000)
        case = dict(schema_version='apctt-pdf-example-v1', case_id='test',
                    assessment_target=dict(target_id='pump', description='Pump X'),
                    documents=[dict(document_id='doc-2', path='report.pdf')])
        mock = dict(status='completed', predicted_trl=4, evidence={
            'APCTT-C04': dict(document_id='doc-2', page=7, quote='Pump X was tested.')})
        with patch.dict('os.environ', {'APCTT_MODEL_STORE': '/offline/models'}):
            obj.load({'checklist': checklist, 'compute': {'device': 'cpu'}}, Path('assets'))
        with patch.dict('sys.modules', torch=SimpleNamespace(set_num_threads=lambda _: None)), patch('innovation.adapter.read_pdf', return_value=[]), patch(
                'innovation.adapter.assess_pages', return_value=mock):
            result = obj.assess(case, context)
        self.assertEqual(result['predicted_trl'], 4)
        self.assertEqual(result['evidence_reference']['APCTT-C04'][0],
                         dict(document_id='doc-2', page=7, supporting_span='Pump X was tested.'))
        self.assertEqual(result['criterion_status']['APCTT-C10'], 'insufficient evidence')


if __name__ == '__main__':
    unittest.main()
