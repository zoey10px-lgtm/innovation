"""Page-grounded PDF extraction for offline evaluation.

Preserves the exact PDF page index, and section-adjacent text windows.
"""
from __future__ import annotations
from dataclasses import dataclass
import re

from .engine import Evidence

SECTION_HEADING = re.compile(
    r"^[ \t]*(?:[1-9]\d?[.)]?[ \t]+)?"
    r"(references(?: cited)?|bibliography|literature cited|"
    r"appendix(?:[ \t]+[A-Z0-9])?(?:[.:][^\r\n]*)?|"
    r"results(?: and discussion)?|discussion|conclusions?(?: and recommendations)?|"
    r"introduction|(?:materials and )?methods|methodology)[ \t\r]*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Page:
    document_id: str
    number: int
    text: str


def read_pdf(path: str, document_id: str, remaining_ms=None, safety_ms: int = 18000, enable_ocr: bool = True) -> list[Page]:
    if remaining_ms is None:
        remaining_ms = lambda: 180000
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        result = []
        try:
            for idx in range(len(pdf)):
                if remaining_ms() < safety_ms:
                    break
                page = pdf[idx]
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_bounded() or ""
                finally:
                    textpage.close()
                    page.close()
                result.append(Page(document_id, idx + 1, text))
        finally:
            pdf.close()
        if enable_ocr and result and remaining_ms() >= 35000:
            from .ocr import selective_ocr
            corrected = selective_ocr(path, [p.text for p in result], remaining_ms=remaining_ms)
            result = [Page(document_id, p.page, p.text) for p in corrected]
        return result
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(path)
        result = []
        for idx, page in enumerate(reader.pages):
            if remaining_ms() < safety_ms:
                break
            result.append(Page(document_id, idx + 1, page.extract_text() or ""))
        return result


def evidence_sections(pages):
    section_by_document = {}
    for page in pages:
        section = section_by_document.get(page.document_id, "")
        text = page.text or ""
        start = 0
        for heading in SECTION_HEADING.finditer(text):
            if start < heading.start() and section != "bibliography":
                yield page, section, text[start:heading.start()]
            title = heading[1].casefold()
            if title.startswith(("references", "bibliography", "literature cited")):
                section = "bibliography"
            elif title.startswith(("result", "discussion", "conclusion")):
                section = "results"
            else:
                section = ""
            start = heading.start()
        if start < len(text) and section != "bibliography":
            yield page, section, text[start:]
        section_by_document[page.document_id] = section


def extract_windows(pages: list[Page], max_chars: int = 1000, min_chars: int = 20) -> list[Evidence]:
    """Exact-span chunks: every output is a verbatim substring of page text.

    Preserve whitespace and offsets. No paraphrasing or cleaned quotations.
    """
    evidence = []
    for page, section, text in evidence_sections(pages):
        offset = 0
        while offset < len(text):
            end = min(len(text), offset + max_chars)
            if end < len(text):
                window_start = offset + max_chars // 2
                sentence_end = text.rfind(". ", window_start, end)
                line_end = text.rfind("\n", window_start, end)
                if sentence_end != -1:
                    end = sentence_end + 2
                elif line_end != -1:
                    end = line_end + 1
            if end <= offset:
                end = min(len(text), offset + max_chars)
            segment = text[offset:end]
            if len(segment.strip()) >= min_chars:
                evidence.append(Evidence(page.number, segment, section=section, source_id=page.document_id))
            if end == len(text):
                break
            # Carry nearby lines into the next window when a finding straddles a boundary.
            overlap_start = text.find("\n", max(offset + 1, end - max_chars // 5), end - 1)
            offset = overlap_start + 1 if overlap_start != -1 else end
    return evidence
