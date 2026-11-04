"""Conservative, auditable extraction repairs using OCR evidence only.

No reference labels, invoice names, supplier lookup tables or network access.
Ambiguous row associations are left unchanged. This cannot recover lost OCR.
"""

import copy
import html
import re
from datetime import datetime
from difflib import SequenceMatcher
from html.parser import HTMLParser


def clean(s):
    return " ".join(html.unescape(s).split())


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.rows = None
        self.row = None
        self.cell = None
        self.span = 1
        self.unsafe = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.rows = []
            self.unsafe = False
        elif tag == "tr" and self.rows is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
            try:
                self.span = int(attrs.get("colspan", 1))
                if int(attrs.get("rowspan", 1)) != 1:
                    self.unsafe = True
                if not 1 <= self.span <= 100:
                    self.unsafe = True
                    self.span = 1
            except ValueError:
                self.unsafe = True
                self.span = 1
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, s):
        if self.cell is not None:
            self.cell.append(s)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.extend([clean("".join(self.cell))] + [""] * (self.span - 1))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.rows is not None:
            if not self.unsafe:
                self.tables.append(self.rows)
            self.rows = None


def heading(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def transport_rows(text, items):
    """Recover identity only from a complete, independently checked row sequence.

    OCR sometimes splits a combined service column and merges date/consignment.
    We do not infer the position of monetary/quantity columns in such tables.
    An existing amount is used only as a cross-check, never as an output label.
    No repairs if row count, source identity, amount or description checks fail.
    """
    from decimal import Decimal, InvalidOperation

    rows = []
    for table in re.findall(r"<table\b.*?</table>", text, re.S | re.I):
        if not all(
            label in table.lower()
            for label in ("consignment note number", "origin/destination", "rating codes")
        ):
            continue
        for raw in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S | re.I):
            cells = [
                clean(re.sub(r"<[^>]+>", " ", v))
                for v in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", raw, re.S | re.I)
            ]
            if len(cells) < 12:
                continue
            merged = re.fullmatch(r"(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{6,})", cells[0])
            if merged:
                cells = [merged[1], merged[2]] + cells[1:]
            if not re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", cells[0]):
                continue
            if not re.fullmatch(r"\d{6,}", cells[1]) or cells[3] not in ("S", "R", "T"):
                continue
            if "/" not in cells[2] or not re.fullmatch(r"\d+(?:\s+\d+)?", cells[4]):
                continue
            rows.append(cells)
    if not rows or len(rows) != len(items):
        return []
    for cells, item in zip(rows, items):
        desc = clean(item.get("itemDesc", "")).casefold()
        origin = cells[2].casefold()
        rating = cells[5].casefold()
        similar = max(
            SequenceMatcher(None, desc, v, autojunk=False).ratio() for v in (origin, rating)
        )
        if not desc or similar < 0.85:
            return []
        try:
            amount = Decimal(item.get("exExtendedPrice", "").replace(",", ""))
            tail = []
            for v in cells[-4:]:
                try:
                    tail.append(Decimal(v.replace(",", "")))
                except InvalidOperation:
                    pass
            if amount not in tail:
                return []
        except InvalidOperation:
            return []
    return rows


def source_rows(text):
    found = []
    tables = []
    for match in re.finditer(r"<table\b.*?</table>", text, re.S | re.I):
        parser = Tables()
        parser.feed(match.group())
        context = re.sub(
            r"<\|det\|>.*?<\|/det\|>", "", text[max(0, match.start() - 400) : match.start()]
        )
        ids = re.findall(r"(?im)^ID:\s*([\w-]+)\s*$", context)
        for table in parser.tables:
            tables.append((table, ids[-1] if ids else None))
    aliases = {
        "itemDesc": {"description", "code description", "recurring charges"},
        "itemCode": {"product code", "part number", "item code", "code", "item no your item"},
        "quantity": {"qty", "quantity", "delivered", "del", "qty supplied", "supplied"},
        "unitOfMeasure": {"uom", "per", "unit of measure", "unit"},
        "exUnitPrice": {"price ex", "unit price excl gst", "unit price ex gst"},
        "incUnitPrice": {"unit price incl gst", "unit price inc gst"},
        "exExtendedPrice": {"total ex", "total ex gst", "total excluding gst", "ex gst"},
        "incExtendedPrice": {"total inc", "total incl gst", "total inc gst"},
    }
    for table, initial_id in tables:
        mapping = None
        headers = None
        service_id = initial_id
        for cells in table:
            hs = [heading(x) for x in cells]
            if any(x in aliases["itemDesc"] for x in hs):
                headers = hs
                mapping = {}
                for key, options in aliases.items():
                    indexes = [i for i, h in enumerate(hs) if h in options]
                    if len(indexes) == 1:
                        mapping[key] = indexes[0]
                continue
            if not mapping or "itemDesc" not in mapping or len(cells) != len(headers):
                continue
            joined = " ".join(cells).strip()
            m = re.fullmatch(r"ID:\s*([\w-]+)", joined)
            if m:
                service_id = m.group(1)
                continue
            desc = cells[mapping["itemDesc"]]
            if not desc or re.match(
                r"(?i)^(?:sub\s*total|grand total|total|freight|handling)\b", desc
            ):
                continue
            row = {key: cells[i] for key, i in mapping.items()}
            if service_id:
                row["itemCode"] = service_id
                row["_service_id"] = service_id
            if "itemCode" in mapping and headers[mapping["itemCode"]] == "item no your item":
                row["itemCode"] = row["itemCode"].split()[0] if row["itemCode"] else ""
            if "unitOfMeasure" in row:
                u = row["unitOfMeasure"].lower()
                row["unitOfMeasure"] = (
                    "each"
                    if u in ("ea", "each")
                    else "hund"
                    if u in ("100", "hund", "hundred")
                    else ""
                )
            # A zero-delivery backorder is not a delivered free-of-charge item.
            row["_undelivered"] = False
            if "quantity" in mapping and headers[mapping["quantity"]] in (
                "delivered",
                "del",
                "qty supplied",
                "supplied",
            ):
                back = [
                    i for i, h in enumerate(headers) if h in ("b order", "bord", "back ordered")
                ]
                if row.get("quantity") in ("0", "0.00", "0.000") and back:
                    row["_undelivered"] = bool(re.search(r"[1-9]", cells[back[0]]))
            row["_evidence"] = dict(headers=headers, cells=cells)
            found.append(row)
    return found


def repair(data, text):
    """Return a copy plus evidence-bearing changes; never mutate input."""
    out = copy.deepcopy(data)
    obj = out["invoiceDataArray"]
    changes = []

    def change(target, key, value, path, evidence, rule):
        if target.get(key) != value:
            changes.append(
                dict(field=path, before=target.get(key), after=value, rule=rule, evidence=evidence)
            )
            target[key] = value

    tax = obj.get("taxNumber", "")
    stripped = re.sub(r"(?i)^\s*A\.?B\.?N\.?\s*:?\s*", "", tax)
    if re.fullmatch(r"[\d\s]{11,20}", stripped) and len(re.sub(r"\D", "", stripped)) == 11:
        change(obj, "taxNumber", stripped.strip(), "taxNumber", tax, "remove_tax_label")
    plain = html.unescape(re.sub(r"<[^>]*>", "\n", re.sub(r"<\|det\|>.*?<\|/det\|>", "", text)))
    # Resolve a stated legal issuer, not a supplier directory or guessed suffix.
    supplier = obj.get("supplierName", "").strip()
    legal = r"(?:Pty\.?\s+(?:Ltd\.?|Limited)|P/L)"
    issuer = re.split(r"(?i)\s+(?:a\s+)?division\s+of\s+", supplier)[-1]
    issuer = re.split(r"(?i)\s+trading\s+as\s+", issuer)[0].strip()
    if (
        issuer != supplier
        and re.search(legal + r"$", issuer, re.I)
        and clean(issuer).casefold() in clean(plain).casefold()
    ):
        change(obj, "supplierName", issuer, "supplierName", issuer, "explicit_legal_issuer")
    elif supplier and not re.search(legal + r"$", supplier, re.I):
        pattern = re.escape(supplier).replace(r"\ ", r"\s+") + r"\s+" + legal + r"(?!\w)"
        matches = {
            clean(m.group()).casefold(): clean(m.group()) for m in re.finditer(pattern, plain, re.I)
        }
        if len(matches) == 1:
            value = next(iter(matches.values()))
            change(
                obj, "supplierName", value, "supplierName", value, "source_supported_legal_suffix"
            )
    if obj.get("currency") == "AUD" and not re.search(
        r"\bAUD\b|\bAU\$|\bA\$|Australian [Dd]ollars", plain
    ):
        change(
            obj,
            "currency",
            "",
            "currency",
            "No explicit AUD marker in OCR",
            "explicit_currency_only",
        )
    # Australian day-first policy; two-digit years follow strptime's documented pivot.
    dates = set()
    for m in re.finditer(
        r"(?im)(?:^|\n)\s*(?:Invoice\s+Date|Issue\s+Date|Date)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2}(?:\d{2})?)\b",
        plain,
    ):
        raw = m.group(1)
        try:
            fmt = "%d/%m/%Y" if len(raw.split("/")[-1].split("-")[-1]) == 4 else "%d/%m/%y"
            dates.add(datetime.strptime(raw.replace("-", "/"), fmt).date().isoformat())
        except ValueError:
            pass
    if len(dates) == 1:
        change(
            obj, "invoiceDate", next(iter(dates)), "invoiceDate", sorted(dates), "AU_day_first_date"
        )
    transport = transport_rows(text, obj["poLineArray"])
    for i, (item, cells) in enumerate(zip(obj["poLineArray"], transport)):
        change(
            item,
            "itemCode",
            cells[1],
            f"poLineArray[{i}].itemCode",
            cells,
            "verified_transport_row_identity",
        )
    candidates = source_rows(text)
    remove = []
    for i, item in enumerate(obj["poLineArray"]):
        desc = clean(item.get("itemDesc", "")).casefold()
        code = item.get("itemCode", "")
        scored = []
        for row in candidates:
            sim = SequenceMatcher(
                None, desc, clean(row["itemDesc"]).casefold(), autojunk=False
            ).ratio()
            exact_code = bool(code and row.get("itemCode") and code.split()[0] == row["itemCode"])
            if sim >= 0.92 or (exact_code and sim >= 0.65):
                scored.append((sim + int(exact_code), row))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        # Repeated identical OCR tables are fine; conflicting candidate rows are not.
        contenders = [r for s, r in scored if scored[0][0] - s < 0.05]
        signatures = {str({k: v for k, v in r.items() if k != "_evidence"}) for r in contenders}
        if len(signatures) != 1:
            continue
        if best["_undelivered"]:
            remove.append(i)
            changes.append(
                dict(
                    field=f"poLineArray[{i}]",
                    before=item,
                    after=None,
                    rule="exclude_undelivered_backorder",
                    evidence=best["_evidence"],
                )
            )
            continue
        for key, value in best.items():
            if key.startswith("_") or not value and key != "unitOfMeasure":
                continue
            # Don't overwrite quantities/prices with malformed source strings.
            if key in (
                "quantity",
                "exUnitPrice",
                "incUnitPrice",
                "exExtendedPrice",
                "incExtendedPrice",
            ):
                value = value.replace("$", "").strip()
                if not re.fullmatch(r"-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", value):
                    continue
            change(
                item,
                key,
                value,
                f"poLineArray[{i}].{key}",
                best["_evidence"],
                "unique_source_table_row",
            )
        if best.get("_service_id") and item.get("location") == best["_service_id"]:
            change(
                item,
                "location",
                "",
                f"poLineArray[{i}].location",
                best["_service_id"],
                "service_identifier_not_location",
            )
    obj["poLineArray"] = [v for i, v in enumerate(obj["poLineArray"]) if i not in remove]
    return out, changes
