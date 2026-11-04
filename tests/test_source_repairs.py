"""Synthetic checks; no benchmark references or invoice fixtures."""

import copy
import unittest

from invoice_pipeline.core import HEADERS, ITEMS
from invoice_pipeline.source_repairs import repair, source_rows, transport_rows


def invoice(**fields):
    result = {key: "" for key in HEADERS}
    result["poLineArray"] = []
    result.update(fields)
    return {"invoiceDataArray": result}


def item(**fields):
    result = {key: "" for key in ITEMS}
    result.update(fields)
    return result


class SourceRepairsTest(unittest.TestCase):
    def test_legal_issuer_requires_source(self):
        d = invoice(supplierName="Example Brand - a division of Example Co Pty Ltd")
        self.assertEqual(
            repair(d, "Example Co Pty Ltd")[0]["invoiceDataArray"]["supplierName"],
            "Example Co Pty Ltd",
        )
        self.assertEqual(repair(d, "No supporting legal name")[0], d)
        d = invoice(supplierName="Example Supplies")
        self.assertEqual(
            repair(d, "Property of Example Supplies P/L until paid")[0]["invoiceDataArray"][
                "supplierName"
            ],
            "Example Supplies P/L",
        )
        self.assertEqual(repair(d, "Example Supplies P/L and Example Supplies Pty Ltd")[0], d)

    def test_transport_identity_guards(self):
        header = "<tr><td>Date</td><td>Consignment Note Number</td><td>Origin/Destination</td><td>Rating Codes</td></tr>"
        cells = [
            "12/03/26",
            "123456789",
            "City A/City B",
            "S",
            "42",
            "EXP",
            "",
            "2",
            "3.000",
            "0.1",
            "V",
            "10.00",
            "0.00",
            "1.00",
            "11.00",
            "1.10",
        ]
        text = (
            "<table>"
            + header
            + "<tr>"
            + "".join("<td>" + v + "</td>" for v in cells)
            + "</tr></table>"
        )
        d = invoice(
            poLineArray=[item(itemCode="42", itemDesc="City A/City B", exExtendedPrice="11.00")]
        )
        fixed = repair(d, text)[0]["invoiceDataArray"]["poLineArray"][0]
        self.assertEqual(fixed["itemCode"], "123456789")
        self.assertEqual(fixed["quantity"], "")  # No inferred numeric columns.
        self.assertEqual(transport_rows(text, d["invoiceDataArray"]["poLineArray"] * 2), [])
        d["invoiceDataArray"]["poLineArray"][0]["exExtendedPrice"] = "999.00"
        self.assertEqual(repair(d, text)[0], d)

    def test_tax_currency_and_immutability(self):
        data = invoice(taxNumber="ABN 12 345 678 901", currency="AUD")
        original = copy.deepcopy(data)
        fixed, log = repair(data, "Total $50.00")
        self.assertEqual(data, original)
        self.assertEqual(fixed["invoiceDataArray"]["currency"], "")
        self.assertEqual(fixed["invoiceDataArray"]["taxNumber"], "12 345 678 901")
        self.assertEqual(len(log), 2)

    def test_explicit_currency(self):
        for marker in ("AUD", "A$", "AU$", "Australian dollars"):
            self.assertEqual(
                repair(invoice(currency="AUD"), marker)[0]["invoiceDataArray"]["currency"], "AUD"
            )

    def test_date_and_conflicts(self):
        self.assertEqual(
            repair(invoice(), "Date: 17/03/26")[0]["invoiceDataArray"]["invoiceDate"], "2026-03-17"
        )
        for text in ("Date: 31/02/2026", "Date: 17/03/26\nDate: 18/03/26", "Due Date: 17/03/26"):
            self.assertEqual(repair(invoice(), text)[0]["invoiceDataArray"]["invoiceDate"], "")

    def test_undelivered_vs_free_goods(self):
        text = "<table><tr><td>Description</td><td>Delivered</td><td>B/order</td></tr><tr><td>Sample product</td><td>0</td><td>3</td></tr></table>"
        data = invoice(poLineArray=[item(itemDesc="Sample product", quantity="0")])
        self.assertEqual(repair(data, text)[0]["invoiceDataArray"]["poLineArray"], [])
        self.assertEqual(
            len(
                repair(data, text.replace("<td>0</td>", "<td>1</td>"))[0]["invoiceDataArray"][
                    "poLineArray"
                ]
            ),
            1,
        )

    def test_ambiguous_rows_unchanged(self):
        text = "<table><tr><td>Description</td><td>Qty</td></tr><tr><td>Widget</td><td>2</td></tr><tr><td>Widget</td><td>7</td></tr></table>"
        data = invoice(poLineArray=[item(itemDesc="Widget", quantity="3")])
        self.assertEqual(repair(data, text)[0], data)

    def test_ids_and_units(self):
        text = "ID: SVC-99\n<table><tr><td>Description</td><td>Per</td></tr><tr><td>Support</td><td>EA</td></tr></table>"
        data = invoice(poLineArray=[item(itemDesc="Support", location="SVC-99")])
        row = repair(data, text)[0]["invoiceDataArray"]["poLineArray"][0]
        self.assertEqual(
            (row["itemCode"], row["location"], row["unitOfMeasure"]), ("SVC-99", "", "each")
        )

    def test_rowspans_rejected(self):
        self.assertEqual(
            source_rows(
                '<table><tr><td>Description</td><td>Qty</td></tr><tr><td rowspan="2">A</td><td>3</td></tr></table>'
            ),
            [],
        )

    def test_no_arithmetic_or_added_rows(self):
        text = "<table><tr><td>Description</td><td>Qty</td><td>Price Ex</td></tr><tr><td>Other</td><td>2</td><td>10.00</td></tr></table>"
        data = invoice(poLineArray=[item(itemDesc="Original")])
        self.assertEqual(repair(data, text)[0], data)


if __name__ == "__main__":
    unittest.main()
