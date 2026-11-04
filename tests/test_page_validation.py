import unittest
from unittest.mock import patch

from benchmark.page_validation import VERIFIED_BLANKS, validate_page


class PageValidationTests(unittest.TestCase):
    def setUp(self):
        fixture = patch.dict(VERIFIED_BLANKS, {("synthetic.pdf", "a" * 64): {2}}, clear=True)
        fixture.start()
        self.addCleanup(fixture.stop)

    def call(self, text="", completed=True, **overrides):
        (name, digest), pages = next(iter(VERIFIED_BLANKS.items()))
        args = dict(filename=name, source_sha256=digest, page_number=min(pages))
        args.update(overrides)
        return validate_page(text, completed, **args)

    def test_all_declared_blanks(self):
        for (name, digest), pages in VERIFIED_BLANKS.items():
            for page in pages:
                self.assertTrue(
                    self.call(filename=name, source_sha256=digest, page_number=page)[
                        "verified_blank"
                    ]
                )

    def test_whitespace_preserved(self):
        self.assertEqual(self.call(" \n")["text"], " \n")

    def test_nonblank_rejected(self):
        with self.assertRaises(RuntimeError):
            self.call(page_number=1)

    def test_wrong_hash_rejected(self):
        with self.assertRaises(RuntimeError):
            self.call(source_sha256="changed")

    def test_unknown_file_rejected(self):
        with self.assertRaises(RuntimeError):
            self.call(filename="other.pdf")

    def test_truncation_on_blank_rejected(self):
        with self.assertRaises(RuntimeError):
            self.call(completed=False)

    def test_partial_retained(self):
        with self.assertRaises(RuntimeError) as caught:
            self.call("partial", False)
        self.assertEqual(caught.exception.partial_text, "partial")

    def test_hallucination_on_blank_preserved(self):
        result = self.call("invented content")
        self.assertTrue(result["nonempty_on_blank"])
        self.assertEqual(result["text"], "invented content")

    def test_normal_text_accepted(self):
        self.assertEqual(self.call("Invoice", page_number=1)["text"], "Invoice")

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            self.call(None)

    def test_zero_based_page_rejected(self):
        with self.assertRaises(ValueError):
            self.call(page_number=0)


if __name__ == "__main__":
    unittest.main()
