import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers must never see half-written checkpoints after interruption.
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dataset():
    records = read_json(ROOT / "data/ground_truth/ground_truth.json")
    pairs = read_json(ROOT / "invoice_subset_validation.json")["pairs"]
    folder = ROOT / "data/invoices"
    assert len(records) == len(pairs) == 100, "Expected exactly 100 references and mappings"
    assert len({p["filename"].casefold() for p in pairs}) == 100
    assert {p["ground_truth_array_index"] for p in pairs} == set(range(100))
    assert {p.name for p in folder.iterdir() if p.is_file()} == {p["filename"] for p in pairs}
    result = []
    for pair in pairs:
        path = folder / pair["filename"]
        assert path.resolve().parent == folder.resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pair["pdf_sha256"], path
        record = records[pair["ground_truth_array_index"]]["invoiceDataArray"]
        result.append((path, record))
    return result


def render(path, destination, dpi):
    """Render pixels only: never use the PDF text layer as model input."""
    from PIL import Image

    destination.mkdir(parents=True, exist_ok=True)
    pages = []
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        with pdfium.PdfDocument(str(path)) as document:
            for index in range(len(document)):
                page = document[index]
                bitmap = page.render(scale=dpi / 72)
                target = destination / f"{index:04}.png"
                bitmap.to_pil().convert("RGB").save(target)
                bitmap.close()
                page.close()
                pages.append(target)
    else:
        target = destination / "0000.png"
        with Image.open(path) as image:
            image.convert("RGB").save(target)
        pages.append(target)
    if not pages:
        raise ValueError(f"No pages in {path}")
    return pages
