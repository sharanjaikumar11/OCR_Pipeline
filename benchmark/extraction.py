"""Conservative shared rules, independent of ground truth. Not a production extractor."""

import re
from html import unescape
from html.parser import HTMLParser

FIELDS = "supplierName invoiceNumber taxNumber poNumber bankBSB bankAccount invoiceAmountEx invoiceAmountInc invoiceDate dueDate currency paymentTerms freightAmount shipToName shipToAddress soldToName soldToAddress".split()
LINE_FIELDS = "itemCode itemDesc quantity exUnitPrice incUnitPrice exExtendedPrice incExtendedPrice unitOfMeasure poNumberDetail location".split()


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)


def plain_text(raw):
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", raw)
    text = re.sub(r"</(?:td|th)>", " ", text, flags=re.I)
    text = re.sub(r"</(?:tr|p|div)>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text)


def extract(raw):
    result = dict.fromkeys(FIELDS, "")
    lines = [re.sub(r"^[#* ]+", "", x).strip() for x in plain_text(raw).splitlines() if x.strip()]
    rules = {
        "invoiceNumber": r"(?:tax\s+invoice|invoice)(?:\s*(?:number|no\.?|#))?",
        "taxNumber": r"(?:A\.?B\.?N\.?|GSTIN|VAT\s*(?:number|no\.?)?)",
        "poNumber": r"(?:P\.?O\.?\s*(?:number|no\.?|#)|cust(?:omer)?\s+order\s*(?:no\.?)?)",
        "bankBSB": r"BSB",
        "bankAccount": r"(?:bank\s+account|account\s*(?:number|no\.?))",
        "invoiceAmountEx": r"(?:sub\s*total|total\s*(?:excl\.?|excluding)\s*(?:GST|tax))",
        "invoiceAmountInc": r"(?:total\s+invoice\s*(?:amt|amount|value)?|grand\s+total|total\s+AUD|total)",
        "invoiceDate": r"(?:invoice\s+date|date)",
        "dueDate": r"(?:payment\s+due\s+date|due\s+date)",
        "paymentTerms": r"(?:payment\s+terms|terms)",
        "freightAmount": r"(?:freight|shipping\s+charge)",
    }
    money = {"invoiceAmountEx", "invoiceAmountInc", "freightAmount"}
    for key, label in rules.items():
        for index, text in enumerate(lines):
            match = re.match(r"^" + label + r"\s*[:#]?\s+(.+)$", text, re.I)
            value = match.group(1).strip() if match else ""
            if (
                not value
                and re.fullmatch(label + r"\s*[:#]?", text, re.I)
                and index + 1 < len(lines)
            ):
                value = lines[index + 1]
            if key in money:
                amounts = re.findall(r"(?<!\w)[\d,]+\.\d{2}(?!\d)", value)
                value = amounts[-1].replace(",", "") if amounts else ""
            if value:
                result[key] = value
                break
    # Supplier is a documented first-non-heading-line heuristic, never a GT lookup.
    result["supplierName"] = next(
        (x for x in lines if not re.match(r"^(tax\s+)?invoice\b", x, re.I)), ""
    )
    currency = re.search(r"\b(AUD|INR|USD|EUR|GBP|NZD)\b", raw)
    result["currency"] = currency.group(1) if currency else ""
    for label, name, address in [
        ("ship to|delivery", "shipToName", "shipToAddress"),
        ("sold to|bill to", "soldToName", "soldToAddress"),
    ]:
        for index, text in enumerate(lines):
            if re.fullmatch(r"(?:" + label + r")\s*:?", text, re.I) and index + 1 < len(lines):
                result[name] = lines[index + 1]
                # Address boundaries vary; do not invent a fixed number of lines.
                break
    parser = TableReader()
    parser.feed(raw)
    rows = parser.rows or [
        [c.strip() for c in x.strip("|").split("|")] for x in raw.splitlines() if "|" in x
    ]
    aliases = {
        "item code": "itemCode",
        "item id": "itemCode",
        "part": "itemCode",
        "stock code": "itemCode",
        "description": "itemDesc",
        "qty": "quantity",
        "quantity": "quantity",
        "this shipment": "quantity",
        "unit price": "exUnitPrice",
        "amount": "exExtendedPrice",
        "extension": "exExtendedPrice",
        "unit": "unitOfMeasure",
        "uom": "unitOfMeasure",
        "po": "poNumberDetail",
        "location": "location",
    }
    columns, items = {}, []
    # Ambiguous table tax basis is not assumed from an unlabeled Unit Price/Amount header.
    explicit_ex = bool(re.search(r"\b(?:excl(?:uding)?|ex)\.?\s*(?:tax|GST)\b", raw, re.I))
    for row in rows:
        header = {i: aliases[c.lower()] for i, c in enumerate(row) if c.lower() in aliases}
        if "itemDesc" in header.values() and "quantity" in header.values():
            columns = header
            continue
        if not columns or all(re.fullmatch(r"[-: ]*", c) for c in row):
            continue
        item = dict.fromkeys(LINE_FIELDS, "")
        for i, key in columns.items():
            if i < len(row) and (explicit_ex or key not in ("exUnitPrice", "exExtendedPrice")):
                item[key] = row[i]
        if item["itemDesc"] and re.fullmatch(r"\d+(?:\.\d+)?", item["quantity"]):
            items.append(item)
    result["poLineArray"] = items
    return result
