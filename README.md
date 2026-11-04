# Invoice OCR benchmarking and extraction

Compare OCR engines, extract invoice fields from saved OCR using Gemini, and evaluate predictions against independently maintained references. Source-supported corrections are optional and retain the raw extraction.

## Setup

Use Python 3.10 or later. Create a virtual environment and install requirements-dev.txt for development. Install OCR dependencies from requirements.txt only for OCR execution; select CUDA builds for your hardware. PaddleOCR needs its own compatible PaddlePaddle environment. Tesseract and llama.cpp are external executables. Model weights are downloaded separately under models/ using their upstream licenses.

The Gemini REST adapter uses the Python standard library. Set GEMINI_API_KEY in your environment. The .env.example file documents names only; .env is not loaded automatically. Do not commit API keys or real invoices.

## Commands

```bash
python -m pip install -r requirements-dev.txt
python -m pre_commit install
python -m pre_commit run --all-files
python run_invoice_pipeline.py --help
python ocr_benchmark.py --help
python compare.py --help
```

The benchmark expects exactly 100 private PDFs in data/invoices, reference objects in data/ground_truth/ground_truth.json and an invoice_subset_validation.json manifest. Each pairs entry contains filename, ground_truth_array_index and pdf_sha256. These files are deliberately not distributed. Use only data you are authorized to process.

Saved OCR extraction accepts --ocr-dir, --source-dir, --out, --provider gemini and --allow-cloud-upload. The source PDF hashes must match the OCR records. Start with --dry-run. Enable --source-repairs to retain raw JSON and apply auditable source-text corrections. The configured model is gemini-2.5-flash; provider availability must be checked in your account.

## Architecture

- benchmark/: lazy OCR adapters, page rendering, metrics and common field extraction.
- invoice_pipeline/: schema, prompt, REST adapters and source-supported repairs.
- ocr_benchmark.py: isolated OCR workers and saved records.
- run_invoice_pipeline.py: OCR or cached text to structured JSON.
- compare.py: evaluate saved OCR records.
- tests/: synthetic unit tests without model downloads or live API calls.
- docs/: three approved aggregate project reports.

## Results and limitations

On the private development dataset, original Gemini extraction scored 72.97% strict field accuracy; optional corrections scored 80.27% strict and 83.43% normalized. The historical field-level evaluation is described in the reports; compare.py is the OCR baseline evaluator and does not reproduce that separate reviewed-reference protocol. The private data and historical runs are not included, so these figures cannot be independently reproduced from this repository alone. No claim of held-out production accuracy is made.

## Publication status

This is a clean publication copy. Original invoices, predictions, internal notes, model weights and prior Git history are excluded. Dependency lists for OCR are not a complete lockfile. Confirm organizational ownership and licensing before public distribution; no license grant is implied by this copy.
