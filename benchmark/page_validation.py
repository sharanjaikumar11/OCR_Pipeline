"""Staged validation policy; not yet wired into the running model adapters.

Page numbers are one-based. Blank declarations come from visual source review,
not OCR predictions or PDF text-layer absence. No pages are skipped.
"""

# Populate only from independently reviewed private source pages.
VERIFIED_BLANKS = {}


def validate_page(text, completed, *, filename, source_sha256, page_number):
    if type(page_number) is not int or page_number < 1:
        raise ValueError("Expected one-based integer page number")
    if not isinstance(text, str):
        raise TypeError("Adapter must return text")
    if not completed:
        error = RuntimeError(
            "Generation did not end naturally; output retained but not eligible for scoring"
        )
        error.partial_text = text
        raise error
    blank = page_number in VERIFIED_BLANKS.get((filename, source_sha256), set())
    if not text.strip() and not blank:
        raise RuntimeError("Empty output on a page not verified blank")
    return {
        "text": text,
        "verified_blank": blank,
        "nonempty_on_blank": blank and bool(text.strip()),
        "generation_completed": True,
    }
