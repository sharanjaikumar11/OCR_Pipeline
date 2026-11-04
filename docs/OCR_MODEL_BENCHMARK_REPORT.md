# OCR Model Benchmark Report

Recorded execution times from the original 100-invoice runs, before accuracy scoring. Later diagnostic retries are not included. All runs used 100-DPI page rendering.

| Model name | Parameters | Mode of run | Total invoice count | Completed invoice count | Failed rate | Average inference per successful invoice | Total processing time for 100 attempts including failures |
|---|---:|---|---:|---:|---:|---:|---:|
| EasyOCR | 24,551,811 (24.55M) | GPU | 100 | 100 | 0% (0/100) | 0.82 s | 87.51 s (1m 28s) |
| Tesseract | Not exposed | CPU | 100 | 100 | 0% (0/100) | 0.65 s | 70.57 s (1m 11s) |
| EasyOCR + TrOCR | 354,692,258 (354.69M) | GPU | 100 | 100 | 0% (0/100) | 9.83 s | 989.07 s (16m 29s) |
| PaddleOCR v5 | Not recorded | GPU | 100 | 100 | 0% (0/100) | 0.50 s | 54.76 s (0m 55s) |
| Unlimited-OCR | 3,336,106,240 (3.336B) | GPU | 100 | 100 | 0% (0/100) | 22.83 s | 2,288.24 s (38m 08s) |
| Falcon-OCR | 269,943,904 (269.94M) | GPU | 100 | 95 | 5% (5/100) | 18.92 s | 2,345.34 s (39m 05s) |
| PaddleOCR-VL | 958,588,736 (958.59M) | GPU | 100 | 91 | 9% (9/100) | 12.56 s | 1,629.15 s (27m 09s) |
| FireRed-OCR | 2,127,532,032 (2.128B) | GPU | 100 | 69 | 31% (31/100) | 34.31 s | 14,551.32 s (4h 02m 31s) |
| Nanonets-OCR2 | 3,754,622,976 (3.755B) | GPU | 100 | 50 | 50% (50/100) | 15.92 s | 1,838.52 s (30m 39s) |

**Combined processing time for the eight GPU model runs: 23,783.91 seconds — 6h 36m 24s.** These runs contain 800 invoice attempts: 705 completed and 95 failed. The CPU run is excluded from this combined total.

The per-invoice column averages recorded inference across successful invoices, including all their pages. The total column records the complete batch-processing loop, including failed attempts, page rendering and result saving. It excludes model loading, warm-up and accuracy scoring. Consequently, the total is not the per-invoice inference average multiplied by 100.
