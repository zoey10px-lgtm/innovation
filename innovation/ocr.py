"""Selective offline OCR fallback for image-only PDF pages.

Requires rapidocr_onnxruntime and pypdfium2 in the actual runtime.
OCR output is marked as OCR-derived; exact span grounding refers to the
extracted OCR text, not guaranteed perfect visual transcription.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class OCRPage:
    page: int
    text: str
    ocr_used: bool
    warning: str = ""

def needs_ocr(text: str, min_chars=75, max_bad_ratio=0.12) -> bool:
    if len(text.strip()) < min_chars:
        return True
    bad = sum(ch in ("\ufffd", "\x00", "\u25a1") for ch in text)
    return bad / max(1, len(text)) > max_bad_ratio

def selective_ocr(pdf_path: str, page_texts: list[str], dpi=160,
                  remaining_ms=None, min_budget_ms=35000) -> list[OCRPage]:
    import pypdfium2 as pdfium
    if remaining_ms is None:
        remaining_ms = lambda: 180000
    pdf = pdfium.PdfDocument(pdf_path)
    ocr = None
    out = []
    try:
        for i, original in enumerate(page_texts):
            if not needs_ocr(original):
                out.append(OCRPage(i+1, original, False))
                continue
            if remaining_ms() < min_budget_ms:
                out.append(OCRPage(i+1, original, False, "OCR skipped: time budget"))
                continue
            page = pdf[i]
            try:
                if not original.strip() and next(page.get_objects(), None) is None:
                    out.append(OCRPage(i+1, original, False))
                    continue
                if ocr is None:
                    from rapidocr_onnxruntime import RapidOCR
                    ocr = RapidOCR()
                bitmap = page.render(scale=dpi/72)
                image = bitmap.to_numpy()
                detected, _ = ocr(image)
                text = "\n".join(str(line[1]) for line in (detected or []) if len(line) > 1)
                out.append(OCRPage(i+1, text if len(text)>len(original) else original,
                                   len(text)>len(original)))
            except Exception as exc:
                out.append(OCRPage(i+1, original, False, f"OCR failed: {type(exc).__name__}"))
            finally:
                page.close()
    finally:
        pdf.close()
    return out
