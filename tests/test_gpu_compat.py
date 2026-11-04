"""CPU-only regression checks; no model loading or downloads."""

import sys
import types
import unittest

from benchmark.compat import checked_text, patch_mask


class CompatibilityTests(unittest.TestCase):
    def test_success(self):
        self.assertEqual(checked_text("invoice", True), "invoice")

    def test_empty(self):
        with self.assertRaises(RuntimeError):
            checked_text("", True)

    def test_truncation_retains_output(self):
        with self.assertRaises(RuntimeError) as caught:
            checked_text("partial", False)
        self.assertEqual(caught.exception.partial_text, "partial")

    def test_mask_rename_is_scoped(self):
        module = types.ModuleType("_compat_test_model")

        def mask(input_embeds):
            return input_embeds

        module.create_causal_mask = mask
        sys.modules[module.__name__] = module
        cls = type("Model", (), {"__module__": module.__name__})
        try:
            patch_mask(cls)
            self.assertEqual(module.create_causal_mask(inputs_embeds=42), 42)
            self.assertEqual(mask(input_embeds=7), 7)
        finally:
            del sys.modules[module.__name__]


if __name__ == "__main__":
    unittest.main()
