# Invoice Extraction Pipeline Report

Prepared 3 November 2026

## Project overview

The project converts invoice PDFs into structured invoice data. Unlimited-OCR reads the invoice pages, and Gemini 2.5 Flash extracts the required fields from the saved OCR text. An additional correction stage checks selected values against evidence in that text.

The evaluation covers 100 invoices. The original Gemini output achieved 72.97% strict field accuracy. After source-supported corrections, the pipeline achieved 80.27% strict accuracy and 83.43% formatting-normalized accuracy on the same dataset. These results meet the current 80% target, but remaining errors still require review.

## Pipeline workflow

**Invoice PDF → Page rendering → Unlimited-OCR → Saved OCR text → Gemini 2.5 Flash → Structured JSON → Validation and source-supported corrections → Saved results and evaluation**

| Stage | What happens | Output |
|---|---|---|
| Invoice input | Original PDF invoices are collected and identified. | Source invoices |
| Page rendering | PDF pages are rendered as images at the benchmark resolution of 100 DPI. | Page images |
| OCR extraction | Unlimited-OCR reads the page images on the GPU server. | Text with layout markers and HTML tables |
| OCR storage | Extracted text is saved so downstream processing does not require another OCR run. | Saved OCR records |
| Field extraction | Gemini 2.5 Flash receives the saved OCR text and the invoice extraction instructions. | Structured JSON |
| Validation | The response is checked against the required schema, data types and supported field rules. | Validated output and review warnings |
| Evidence-based correction | Selected fields are corrected only when supported by the saved OCR text. | Corrected JSON and a change audit trail |
| Evaluation | Predicted fields are compared with the reviewed ground truth. | Accuracy results and error reports |

**Gemini receives OCR text, not the original PDF files, in this pipeline.** Therefore, a missing or incorrect OCR value can affect the final extraction even when Gemini follows the prompt correctly.

The correction stage has been evaluated separately on saved outputs. It is available through the optional `--source-repairs` setting; the original daily runner and original saved Gemini outputs remain unchanged.

## Information extracted

The output uses an `invoiceDataArray` object containing invoice-level fields and a `poLineArray` list containing the billed items.

| Group | Fields covered |
|---|---|
| Supplier and invoice identity | Supplier name, invoice number, tax number and purchase order number |
| Banking and amounts | Bank BSB, bank account, invoice amounts excluding and including tax, and freight amount |
| Dates and payment | Invoice date, due date, currency and payment terms |
| Customer and delivery | Sold-to name and address; ship-to name and address |
| Line items | Item code, description, quantity, unit prices, extended prices, unit of measure, line purchase order number and location |

The extraction rules map ABN to tax number and distinguish a customer purchase order from a supplier sales order. Currency is restricted to `AUD` or an empty value, with explicit source evidence required for AUD. Unit of measure is restricted to `each`, `hund` or empty. Unsupported values should remain empty rather than be invented.

## Evaluation method

The original PDFs were used to review the expected field values. The resulting reviewed reference was then compared with the saved extraction JSON for all 100 invoices.

| Measure | Meaning |
|---|---|
| Strict field accuracy | The predicted field must exactly match the reference value. |
| Formatting-normalized field accuracy | Defined Unicode, whitespace and numeric-format differences are accepted. Identifiers and dates remain exact. |
| Missing item | A reference line item has no corresponding predicted item after alignment. |
| Extra item | A predicted line item has no corresponding reference item after alignment. |

Accuracy is calculated as correct scored fields divided by all scored fields. Missing and extra item rows receive penalties for their schema fields. Eight explicitly uncertain reference fields are excluded. Empty fields also contribute to the score.

These are **field-extraction metrics**. They are different from OCR character error rate and word error rate, which compare full-page OCR text with a verified transcription.

## Evaluation results

| Measure | Original Gemini extraction | After source-supported corrections |
|---|---:|---:|
| Completed invoices | 100 of 100 | 100 of 100 |
| Strict field accuracy | 72.97% | **80.27%** |
| Formatting-normalized accuracy | 75.98% | **83.43%** |
| Strictly matching fields | 3,205 | 3,405 |
| Matching fields after normalization | 3,337 | 3,539 |
| Total scored fields | 4,392 | 4,242 |
| Strict mismatches | 1,187 | 837 |
| Remaining mismatches after normalization | 1,055 | 703 |
| Paired item rows | 217 | 229 |
| Missing item rows | 22 | 10 |
| Extra item rows | 31 | 16 |
| Uncertain fields excluded | 8 | 8 |

Strict accuracy improved by **7.30 percentage points**. Formatting-normalized accuracy improved by **7.45 percentage points**.

<!-- GEMINI_TIMINGS_START -->
## Gemini inference and request timing

The saved run used **Gemini 2.5 Flash** for all 100 invoices. The table below reports the measured elapsed time for each Gemini API request. This is the available measure of inference latency from the application: it includes network latency, provider processing and any retries or retry waits within that request. It is not a measurement of model computation alone.

| Timing measure | Measured result |
|---|---:|
| Invoices with recorded timings | 100 of 100 |
| Average Gemini request time per invoice | 3.650 seconds |
| Median Gemini request time per invoice | 3.283 seconds |
| Fastest recorded request | 2.497 seconds |
| Slowest recorded request | 15.559 seconds |
| Total Gemini request time for all 100 saved completed invoices | **364.984 seconds (6.08 minutes)** |
| Total recorded application processing time for those invoices | 367.054 seconds (6.12 minutes) |

The request total is the sum of the 100 saved completed-attempt timings. It excludes scheduled pauses between invoices, waiting for daily quota resets, earlier separate failed attempts, OCR processing, and the later offline correction and evaluation steps. Therefore, it is **not the total calendar duration of the batch**. Application processing time includes the request plus local work within each recorded attempt.

The corrected output reuses these original Gemini extractions. No new Gemini inference was performed to produce the correction-stage accuracy results.

<!-- GEMINI_TIMINGS_END -->
