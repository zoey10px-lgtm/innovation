import time
import unittest
from types import SimpleNamespace

from innovation.assess import assess, sentence_windows, shortlist


class AssessmentTest(unittest.TestCase):
    def setUp(self):
        self.pages = [SimpleNamespace(number=1, text=(
            'The project describes the Pump X design. '
            'Pump X was tested in a laboratory and delivered 50 liters per minute. '
            'Future work will focus on testing a different pump.'))]
        self.criteria = {'APCTT-C04': 'The target completed testing in the laboratory.'}

    def evaluate(self, outputs, score):
        calls = []
        responses = iter(outputs)

        def generate(*args):
            calls.append(args)
            return next(responses)

        def verify(pairs):
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all('50 liters' in quote and 'Future work' not in quote for quote, _ in pairs))
            return [SimpleNamespace(entailment=score, contradiction=0)] * len(pairs)

        result = assess('Pump X', self.pages, self.criteria, lambda *_: [(1, 0.9)],
                        generate, verify, time.monotonic() + 5)
        return result, calls

    def test_verified_quote_is_exact_and_page_grounded(self):
        result, calls = self.evaluate(['{"APCTT-C04":"2"}', '["APCTT-C04","APCTT-C09"]'], .95)
        self.assertEqual(result['predicted_trl'], 4)
        self.assertEqual(len(calls), 2)
        cite = result['evidence']['APCTT-C04']
        self.assertEqual(cite['page'], 1)
        self.assertIn(cite['quote'], self.pages[0].text)

    def test_nli_veto(self):
        result, calls = self.evaluate(['{"APCTT-C04":"2"}'], .30)
        self.assertIsNone(result['predicted_trl'])
        self.assertEqual(len(calls), 1)

    def test_llm_veto(self):
        result, _ = self.evaluate(['{"APCTT-C04":"2"}', '[]'], .95)
        self.assertEqual(result['status'], 'abstained')

    def test_reject_unknown_id_and_malformed_json(self):
        for raw in ('{"APCTT-C04":"99","APCTT-C09":"2"}', 'not json'):
            result, calls = self.evaluate([raw], .95)
            self.assertFalse(result['evidence'])
            self.assertEqual(len(calls), 1)

    def test_page_budget(self):
        pages = [SimpleNamespace(number=i, text=self.pages[0].text) for i in range(1, 10)]
        sentences = list(sentence_windows(pages))
        ranked = [(i, 1.0) for i in range(len(sentences))]
        selected = shortlist(sentences, [ranked])
        self.assertLessEqual(len({item.page for item in selected.values()}), 5)

    def test_expired_deadline(self):
        with self.assertRaises(TimeoutError):
            assess('Pump X', self.pages, self.criteria, None, None, None, time.monotonic() - 1)


if __name__ == '__main__':
    unittest.main()
