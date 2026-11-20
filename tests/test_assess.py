import json
import time
import unittest
from types import SimpleNamespace

from innovation.__main__ import fact_token_constraint
from innovation.assess import FactReviewRequest, assess, decode, sentence_windows, shortlist


def page(text, number=1):
    return SimpleNamespace(number=number, text=text)


def score(entailment, contradiction=0):
    return SimpleNamespace(entailment=entailment, contradiction=contradiction)


class AssessmentTest(unittest.TestCase):
    def run_case(self, pages, criteria, retrieval, outputs, verify):
        calls = []
        responses = iter(outputs)

        def generate(instruction, payload, deadline):
            calls.append(payload)
            return next(responses)

        result = assess('Pump X', pages, criteria, retrieval, generate, verify, time.monotonic() + 20)
        return result, calls

    def test_fallback_reaches_reviewer_when_qwen_omits_criterion(self):
        text = 'Pump X was tested in a laboratory and delivered 50 liters per minute.'
        criteria = {'APCTT-C04': 'The target completed testing in the laboratory.'}
        result, calls = self.run_case([page(text, 7)], criteria, lambda *_: [(0, .9)],
                                      ['{}', '[1,1,1]'], lambda pairs: [score(.94) for _ in pairs])
        self.assertEqual(result['predicted_trl'], 4)
        self.assertEqual(result['evidence']['APCTT-C04'], {'document_id': 'report', 'page': 7, 'quote': text})
        self.assertIsInstance(calls[1], FactReviewRequest)
        self.assertEqual(result['trace']['validation_candidates']['APCTT-C04'], ['1'])

    def test_alternate_evidence_is_checked_after_first_candidate_is_rejected(self):
        pages = [
            page('Pump X was tested in a generic experimental setup, and some results were reported.'),
            page('Pump X was tested in a laboratory and delivered 50 liters per minute.', 2),
        ]
        criteria = {'APCTT-C04': 'The target completed testing in the laboratory.'}
        result, calls = self.run_case(pages, criteria, lambda *_: [(0, .99), (1, .9)],
                                      ['{"APCTT-C04":"1"}', '[1,0,1]', '[1,1,1]'],
                                      lambda pairs: [score(.94) for _ in pairs])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(isinstance(payload, FactReviewRequest) for payload in calls[1:]))
        self.assertEqual(result['predicted_trl'], 4)
        self.assertEqual(result['evidence']['APCTT-C04']['page'], 2)
        self.assertEqual([row['candidate_ids']['APCTT-C04'] for row in result['trace']['review_rounds']], ['1', '2'])

    def test_all_required_facts_must_hold_even_after_nli_passes(self):
        payload = 'Pump X was tested in a laboratory and delivered 50 liters per minute.'
        result, _ = self.run_case([page(payload)], {'APCTT-C07': 'Near-final prototype operated in the field.'},
                                  lambda *_: [(0, 1)], ['{"APCTT-C07":"1"}', '[1,1,0]'],
                                  lambda pairs: [score(.99) for _ in pairs])
        self.assertEqual(result['status'], 'abstained')

    def test_work_evidence_is_extended_only_after_center_fails(self):
        text = ('A pump test was scheduled for last May.\r\n'
                'The laboratory measured a repeatable pump flow under operating load.')
        batches = []
        scores = iter([.2, .93])

        def verify(pairs):
            batches.append(pairs[0][0])
            return [score(next(scores))]

        result, calls = self.run_case([page(text, 5)], {'APCTT-C04': 'Laboratory tests.'},
                                      lambda *_: [(0, .9)], ['{"APCTT-C04":"1"}', '[1,1,1]'], verify)
        self.assertEqual(batches, ['A pump test was scheduled for last May.', text])
        self.assertIn(text, calls[1]['candidates']['APCTT-C04']['excerpt'])
        self.assertEqual(result['evidence']['APCTT-C04']['quote'], text)
        self.assertEqual(result['evidence']['APCTT-C04']['page'], 5)

    def test_concepts_do_not_need_completed_work(self):
        def verify(_):
            self.fail('C02 must not require proof of completed work')

        result, _ = self.run_case([page('Pump X is designed to deliver lunar oxygen through a five kilometer pipeline.')],
                                  {'APCTT-C02': 'A specific design concept.'}, lambda *_: [(0, .9)],
                                  ['{"APCTT-C02":"1"}', '[1,1,1]'], verify)
        self.assertEqual(result['predicted_trl'], 2)

    def test_c09_checks_successful_operation_period(self):
        def verify(pairs):
            self.assertIn('used successfully for a stated period', pairs[0][1])
            return [score(.985)]

        result, _ = self.run_case([page('Pump X operated successfully in service for almost two years.')],
                                  {'APCTT-C09': 'Proven successful deployment over a stated period.'}, lambda *_: [(0, .9)],
                                  ['{"APCTT-C09":"1"}', '[1,1,1]'], verify)
        self.assertEqual(result['predicted_trl'], 9)

    def test_wrong_length_prose_and_boolean_reviews_are_format_errors(self):
        for malformed in ('C04 should be rejected', '[1,0]', '[true,1,1]'):
            result, _ = self.run_case([page('Pump X was tested in a laboratory and delivered 50 liters per minute.')],
                                      {'APCTT-C04': 'Laboratory tests.'}, lambda *_: [(0, .9)],
                                      ['{"APCTT-C04":"1"}', malformed], lambda pairs: [score(.96)])
            self.assertEqual(result['error'], 'review_format_error')
            self.assertIsNone(result['predicted_trl'])

    def test_nli_contradiction_blocks_completed_work(self):
        result, calls = self.run_case([page('Pump X was never tested in its intended operational environment.')],
                                      {'APCTT-C07': 'Actual demonstration.'}, lambda *_: [(0, .9)],
                                      ['{"APCTT-C07":"1"}'], lambda pairs: [score(.92, .22)])
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(len(calls), 1)

    def test_decoding_grammar_has_exactly_two_binary_flags(self):
        class Tokenizer:
            def __call__(self, options, add_special_tokens=False):
                return {'input_ids': [list(map(ord, item)) for item in options]}

            def decode(self, tokens, skip_special_tokens=True):
                return ''.join(map(chr, tokens))

        eos = 1000
        allowed, limit = fact_token_constraint(Tokenizer(), 2, 2, eos)
        outputs = set()

        def visit(tokens):
            self.assertLess(len(tokens), limit)
            for token in allowed(0, [1, 2, *tokens]):
                if token == eos:
                    outputs.add(tuple(json.loads(''.join(map(chr, tokens)))))
                else:
                    visit([*tokens, token])

        visit([])
        self.assertEqual(outputs, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_fenced_json_is_accepted_and_bad_json_is_rejected(self):
        self.assertEqual(decode('```json\n[1,0]\n```', list, None), [1, 0])
        self.assertIsNone(decode('Report says [1]', list, None))

    def test_bullets_stay_separate_and_extensions_stay_on_the_page(self):
        pages = [page('– Pump X completed testing in the lab\r\n– Future work will test another prototype'),
                 page('Pump X completed a different laboratory trial.', 2)]
        items = list(sentence_windows(pages))
        self.assertEqual(len(items), 3)
        self.assertNotIn('Future work', items[0].quote)
        self.assertIn('Future work', items[0].extended_quote)
        self.assertFalse(items[1].extended_quote)

    def test_lexical_evidence_has_a_route_without_semantic_candidates(self):
        pages = [page('Pump X operated successfully in service for almost two years.')]
        result, calls = self.run_case(pages, {'APCTT-C09': 'Successful operation.'},
                                      lambda *_: [], ['[1,1,1]'], lambda pairs: [score(.99)])
        self.assertEqual(result['predicted_trl'], 9)
        self.assertEqual(result['trace']['candidate_pages'], [])
        self.assertEqual(result['trace']['validation_candidate_pages'], [1])
        self.assertEqual(len(calls), 1)

    def test_higher_criterion_can_replace_already_approved_lower_criterion(self):
        pages = [page('Pump X has a design concept for transporting oxygen.'),
                 page('Pump X operated successfully in service for almost two years.', 2)]
        criteria = {'APCTT-C02': 'A concrete concept.', 'APCTT-C09': 'Successful operation.'}
        result, _ = self.run_case(pages, criteria, lambda *_: [(0, 1), (1, .8)],
                                  ['{"APCTT-C09":"1"}', '[1,1,1]', '[1,1,1]'],
                                  lambda pairs: [score(.99 if 'two years' in quote else .1) for quote, _ in pairs])
        self.assertEqual(result['predicted_trl'], 9)
        self.assertEqual(result['evidence']['APCTT-C09']['page'], 2)

    def test_unrecognized_proposal_does_not_escape_source_index(self):
        pages = [page('Pump X was tested in a laboratory and delivered 50 liters per minute.')]
        result, _ = self.run_case(pages, {'APCTT-C04': 'Laboratory tests.'}, lambda *_: [(0, .8)],
                                  ['{"APCTT-C04":"99","APCTT-C09":"1"}', '[1,1,1]'],
                                  lambda pairs: [score(.99)])
        self.assertEqual(result['trace']['proposed_evidence'], {})
        self.assertEqual(result['trace']['validation_candidates'], {'APCTT-C04': ['1']})

    def test_page_budget(self):
        pages = [page('Pump X was tested in a laboratory and delivered 50 liters per minute.', i)
                 for i in range(1, 10)]
        sentences = list(sentence_windows(pages))
        selected = shortlist(sentences, [[(i, 1.0) for i in range(len(sentences))]])
        self.assertLessEqual(len({item.page for item in selected.values()}), 5)

    def test_expired_deadline(self):
        with self.assertRaises(TimeoutError):
            assess('Pump X', [], {}, None, None, None, time.monotonic() - 1)


if __name__ == '__main__':
    unittest.main()
