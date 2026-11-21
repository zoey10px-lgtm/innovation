import time
import unittest

from innovation.page_context import assess_pages
from innovation.pdf_parser import Page


class PageTest(unittest.TestCase):
    def test_full_context_single_call_and_verbatim_citation(self):
        pages = [Page('d', n, f'Definition of pump {n}. ' + 'Full contextual line. ' * 40 +
                      f'Pump {n} reached TRL 6.') for n in range(1, 6)]
        def retrieve(query, chunks):
            return [(i, 1) for i in range(len(chunks))]
        calls = []
        def generate(instruction, data, deadline):
            calls.append(data)
            for row in data['pages']:
                self.assertEqual(row['text'], pages[row['page'] - 1].text)
            return '{"trl":6,"page_key":1,"anchor":"Pump 1 reached TRL 6."}'
        r = assess_pages('pump 1', pages, {}, retrieve, generate, time.monotonic() + 30)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]['pages']), 4)
        self.assertEqual(r['evidence']['APCTT-C06']['quote'], 'Pump 1 reached TRL 6.')

    def test_fabricated_anchor_abstains(self):
        pages = [Page('d', 1, 'This is the source.')]
        r = assess_pages('item', pages, {}, lambda q, c: [(0,1)],
                         lambda *args: '{"trl":9,"page_key":1,"anchor":"never here"}', time.monotonic()+30)
        self.assertEqual(r['status'], 'abstained')
