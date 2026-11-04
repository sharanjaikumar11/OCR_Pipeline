# Why Unlimited-OCR Was Selected

## Selection rationale

Unlimited-OCR was chosen as the preferred OCR model for this invoice dataset because it combined complete processing coverage with the lowest recorded character and word error rates in the saved benchmark comparison. The decision prioritised transcription quality and complete invoice coverage, with processing speed considered after those requirements.

## Evaluation scope

The benchmark used 100 invoice PDFs. Unlimited-OCR processed every invoice, and text accuracy was evaluated against 95 approved full-text references. The accuracy percentages below therefore describe those 95 scored invoices; the completion and timing measurements cover all 100 invoices.

The recorded Unlimited-OCR configuration used GPU execution, 100-DPI page rendering and a maximum generation allowance of 4,096 new tokens. Rendering converted PDF pages into model inputs; no additional image-enhancement stage was used in this baseline.

## Measured results

| Measure | Unlimited-OCR result | Meaning |
| --- | --- | --- |
| Invoice completion | 100 out of 100 — 100% | Produced saved OCR output for the entire dataset |
| Character Error Rate (CER) | 15.48% | Character-level edit errors relative to the approved reference text; lower is better |
| Word Error Rate (WER) | 19.29% | Word-level edit errors relative to the approved reference text; lower is better |
| Token F1 | 94.18% | Supporting measure of token agreement between OCR and reference text; higher is better |
| Mean inference time | 22.83 seconds per invoice | Average recorded OCR inference time across the 100 invoices |
| Total inference time | 2,283.40 seconds — approximately 38.06 minutes | Recorded OCR inference time for the complete dataset |

Completion does not mean every extracted character is correct. CER and WER provide the separate measurements of transcription quality. Inference timing is not a measurement of every surrounding activity, such as document preparation or manual review.

## Why these results supported the choice

### 1. Strongest recorded character and word accuracy

Unlimited-OCR achieved the lowest recorded corpus CER and WER in the saved model comparison. These were the primary accuracy measures because invoice transcription must preserve identifiers, dates, descriptions and amounts with as few text edits as possible.

The reported values are full-text metrics. They do not independently establish the correctness of every financial field or line item.

### 2. Complete dataset coverage

The model produced saved OCR results for all 100 invoices. This met the benchmark's complete-coverage requirement and provided a consistent set of outputs for evaluation.

### 3. Supporting token agreement

The 94.18% token F1 provided additional evidence of agreement with the reference content. It complemented CER and WER rather than replacing them: token agreement alone does not establish correct reading order or exact transcription.

### 4. Measured processing cost

Unlimited-OCR averaged 22.83 seconds of inference per invoice on the recorded server configuration. It was selected for the combination of accuracy and coverage, not because it was the fastest model. Its measured runtime made the speed cost of that choice explicit.

## Decision rule

The saved benchmark applied the following order:

1. Require completion of all 100 invoices.
2. Prefer the lowest corpus CER.
3. Use WER as the next comparison criterion.
4. Use mean invoice inference time as the subsequent criterion.

Unlimited-OCR ranked first under this rule. This supports its selection as the preferred model for the tested invoice dataset and configuration, rather than a claim that it is universally the best OCR model.

## Evidence and interpretation

The figures come from the saved benchmark report dated 21 October 2026: the archived evaluation record.

That report identifies Unlimited-OCR as the recommended candidate. Its automated comparability-review flag remains pending, so the recommendation should be distinguished from an independently signed-off benchmark conclusion. The rationale above describes the evidence supporting the project choice without claiming that this outstanding review has been completed.
