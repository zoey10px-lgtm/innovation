import json
import time
import unittest
from types import SimpleNamespace

from innovation.__main__ import review_token_constraint
from innovation.assess import ReviewRequest, assess, sentence_windows, shortlist


class AssessmentTest(unittest.TestCase):
    def setUp(self):
        self.pages = [SimpleNamespace(number=1, text=(
            'The project describes the Pump X design. '
            'Pump X was tested in a laboratory and delivered 50 liters per minute. '
            'Future work will focus on testing a different pump.'))]
        self.criteria = {'APCTT-C04': 'The target completed testing in the laboratory.'}

    def evaluate(self, outputs, score, criterion_id="APCTT-C04", contradiction=0):
        calls = []
        responses = iter(outputs)

        def generate(*args):
            calls.append(args)
            return next(responses)

        checked_quotes = []

        def verify(pairs):
            self.assertGreaterEqual(int(criterion_id[-2:]), 3)
            self.assertEqual(len(pairs), 1)
            quote, claim = pairs[0]
            self.assertIn('50 liters', quote)
            if not checked_quotes:
                self.assertNotIn('Future work', quote)
            checked_quotes.append(quote)
            self.assertIn('actually performed', claim)
            self.assertNotIn(self.criteria['APCTT-C04'], claim)
            return [SimpleNamespace(entailment=score, contradiction=contradiction)]

        result = assess('Pump X', self.pages, {criterion_id: self.criteria['APCTT-C04']}, lambda *_: [(1, 0.9)],
                        generate, verify, time.monotonic() + 5)
        return result, calls

    def test_verified_quote_is_exact_and_page_grounded(self):
        result, calls = self.evaluate(['{"APCTT-C04":"2"}', '["APCTT-C04","APCTT-C09"]'], .95)
        self.assertEqual(result['predicted_trl'], 4)
        self.assertEqual(len(calls), 2)
        cite = result['evidence']['APCTT-C04']
        self.assertEqual(cite['page'], 1)
        self.assertIn(cite['quote'], self.pages[0].text)

    def test_json_fence_is_accepted(self):
        result, _ = self.evaluate(['```json\n{"APCTT-C04":"2"}\n```', '```json\n["APCTT-C04"]\n```'], .95)
        self.assertEqual(result['predicted_trl'], 4)

    def test_nli_veto(self):
        result, calls = self.evaluate(['{"APCTT-C04":"2"}'], .30)
        self.assertIsNone(result['predicted_trl'])
        self.assertEqual(len(calls), 1)

    def test_high_levels_use_only_the_work_performed_check(self):
        for level in (8, 9):
            key = f'APCTT-C{level:02d}'
            proposal, approval = f'{{"{key}":"2"}}', f'["{key}"]'
            with self.subTest(criterion=key):
                result, calls = self.evaluate([proposal, approval], .8835, key)
                self.assertEqual(result['predicted_trl'], level)
                self.assertEqual(len(result['trace']['nli'][key]), 1)
                self.assertEqual(len(calls), 2)
                for score, contradiction in ((.84, 0), (.90, .11)):
                    result, calls = self.evaluate([proposal], score, key, contradiction)
                    self.assertEqual(result['status'], 'abstained')
                    self.assertEqual(len(calls), 1)

    def test_principles_and_concepts_still_go_through_qwen_review(self):
        for level in (1, 2):
            key = f'APCTT-C{level:02d}'
            proposal = f'{{"{key}":"2"}}'
            with self.subTest(criterion=key):
                result, calls = self.evaluate([proposal, f'["{key}"]'], 0, key)
                self.assertEqual(result['predicted_trl'], level)
                self.assertNotIn('nli', result['trace'])
                self.assertEqual(len(calls), 2)
                result, _ = self.evaluate([proposal, '[]'], 0, key)
                self.assertEqual(result['status'], 'abstained')

    def test_review_accepts_short_id_only_for_verified_criterion(self):
        result, _ = self.evaluate(
            ['{"APCTT-C08":"2"}', '["C08","C09","C10",8,null]'], .8835, 'APCTT-C08')
        self.assertEqual(result['predicted_trl'], 8)
        self.assertEqual(list(result['evidence']), ['APCTT-C08'])

    def test_llm_veto(self):
        result, _ = self.evaluate(['{"APCTT-C04":"2"}', '[]'], .95)
        self.assertEqual(result['status'], 'abstained')

    def test_failed_center_retries_with_exact_next_sentence(self):
        result_sent = 'The laboratory measured a repeatable pump flow under operating load.'
        text = f'A pump test was scheduled for last May.\r\n{result_sent}'
        pages = [SimpleNamespace(number=5, text=text)]
        scores = iter([.30, .92])
        batches = []

        def verify(pairs):
            batches.append(pairs[0][0])
            return [SimpleNamespace(entailment=next(scores), contradiction=0)]

        def generate(instruction, payload, deadline):
            if not isinstance(payload, ReviewRequest):
                return '{"APCTT-C04":"1"}'
            self.assertIn(text, payload['candidates']['APCTT-C04']['excerpt'])
            return '["C04"]'

        result = assess('Pump X', pages, self.criteria, lambda *_: [(0, .9)],
                        generate, verify, time.monotonic() + 5)
        self.assertEqual(batches, ['A pump test was scheduled for last May.', text])
        self.assertEqual(result['evidence']['APCTT-C04']['quote'], text)
        self.assertEqual(result['evidence']['APCTT-C04']['page'], 5)
        self.assertEqual([item['span'] for item in result['trace']['nli']['APCTT-C04']],
                         ['center', 'center_and_next'])
        self.assertEqual(list(sentence_windows([SimpleNamespace(number=6, text=result_sent)]))[0].extended_quote, '')

    def test_invalid_review_has_its_own_error(self):
        result, _ = self.evaluate(['{"APCTT-C04":"2"}', 'C04 was rejected, reasoning follows'], .95)
        self.assertEqual(result['error'], 'review_format_error')
        self.assertTrue(result['trace']['review_format_error'])
        self.assertFalse(result['evidence'])

    def test_decoding_constraint_only_allows_candidate_subsets(self):
        def tokenizer(options, add_special_tokens=False):
            return {'input_ids': [list(map(ord, option)) for option in options]}

        eos = 1000
        allowed, limit = review_token_constraint(tokenizer, ['APCTT-C04', 'APCTT-C08'], 2, eos)
        outputs = set()

        def visit(tokens):
            self.assertLess(len(tokens), limit)
            for token in allowed(0, SimpleTokens([1, 2, *tokens])):
                if token == eos:
                    outputs.add(tuple(json.loads(''.join(map(chr, tokens)))))
                else:
                    visit([*tokens, token])

        class SimpleTokens(list):
            def __getitem__(self, key):
                return SimpleTokens(super().__getitem__(key))

            def tolist(self):
                return list(self)

        visit([])
        self.assertEqual(outputs, {(), ('C04',), ('C08',), ('C04', 'C08')})

    def test_reject_unknown_id_and_malformed_json(self):
        for raw in ('{"APCTT-C04":"99","APCTT-C09":"2"}', 'not json'):
            result, calls = self.evaluate([raw], .95)
            self.assertFalse(result['evidence'])
            self.assertEqual(len(calls), 1)

    def test_pdf_bullets_separate_completed_and_future_work(self):
        pages = [SimpleNamespace(number=1, text='– Pump X completed testing in the lab\r\n– Future work will test another prototype')]
        items = list(sentence_windows(pages))
        self.assertEqual(len(items), 2)
        self.assertNotIn('Future', items[0].quote)

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
