"""Compare saved predictions only. Never loads an OCR model."""

import argparse
import csv
import hashlib
import statistics
from pathlib import Path

from benchmark.data import dataset, read_json, save_json
from benchmark.extraction import plain_text
from benchmark.metrics import compare_record, rates, text_counts
from benchmark.models import NAMES


def evaluate(out, scope="fields", text_reference=None, models=None, invoice=None):
    data = dataset()
    if invoice:
        data = [row for row in data if row[0].name == invoice]
        if len(data) != 1:
            raise ValueError("Invoice must be an exact filename from the dataset")
    texts = read_json(text_reference) if text_reference else {}
    if scope == "full-text":
        missing = [
            p.name
            for p, r in data
            if not isinstance(texts.get(p.name), str) or not texts[p.name].strip()
        ]
        if missing:
            raise ValueError(
                f"Full-text CER/WER needs verified transcriptions for {len(missing)} invoices; structured JSON is not a substitute"
            )
    run = read_json(out / "run.json") if (out / "run.json").exists() else {}
    cohort = models or run.get("models") or NAMES
    summaries = []
    for name in cohort:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        metadata = (
            read_json(folder / "metadata.json") if (folder / "metadata.json").exists() else {}
        )
        details = []
        total = text_counts("", "")
        times = []
        failures = 0
        for path, reference in data:
            pred_path = folder / (path.name + ".json")
            saved = read_json(pred_path) if pred_path.exists() else {"error": "Prediction missing"}
            if saved.get("filename", path.name) != path.name:
                raise ValueError("Prediction identity mismatch")
            if (
                pred_path.exists()
                and saved.get("source_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError(f"Prediction source hash mismatch: {pred_path}")
            error = saved.get("error")
            prediction = saved.get("invoiceDataArray", {}) if not error else {}
            comparison = compare_record(reference, prediction)
            counts = (
                text_counts(
                    texts[path.name], plain_text(saved.get("text", "")) if not error else ""
                )
                if scope == "full-text"
                else comparison["structured_counts"]
            )
            for key in total:
                total[key] += counts[key]
            failures += bool(error)
            if not error and saved.get("inference_seconds") is not None:
                times.append(saved["inference_seconds"])
            details.append(
                {
                    "filename": path.name,
                    "error": error,
                    "inference_seconds": saved.get("inference_seconds"),
                    "metric_scope": scope,
                    **rates(counts),
                    **comparison,
                }
            )
        fields = sum(x["scored_fields"] for x in details)
        correct = sum(x["correct_fields"] for x in details)
        ri = sum(x["reference_items"] for x in details)
        pi = sum(x["prediction_items"] for x in details)
        tp = sum(x["exact_items"] for x in details)
        row = {
            "model": name,
            "metric_scope": scope,
            **rates(total),
            "field_accuracy": correct / max(1, fields),
            "item_precision": tp / max(1, pi),
            "item_recall": tp / max(1, ri),
            "item_f1": 2 * tp / max(1, ri + pi),
            "invoices": len(data),
            "errors": failures,
            "mean_seconds_per_successful_invoice": statistics.mean(times) if times else None,
            "total_inference_seconds_successful": sum(times),
            "total_100_inference_seconds": sum(times)
            if len(times) == 100 and len(data) == 100
            else None,
            "device": metadata.get("device"),
            "parameters": metadata.get("parameters"),
            "status": metadata.get("status", "not_started"),
            "model_error": metadata.get("error"),
            "parameter_note": metadata.get("parameter_note"),
            "load_seconds": metadata.get("load_seconds"),
            "warmup_seconds": metadata.get("warmup_seconds"),
            "total_processing_seconds": metadata.get("total_processing_seconds"),
        }
        summaries.append(row)
        suffix = ".single" if invoice else ""
        save_json(folder / f"comparison{suffix}.json", details)
        with (folder / f"comparison{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["filename", "field", "ground_truth", "prediction", "exact"]
            )
            writer.writeheader()
            for detail in details:
                for field in detail["fields"]:
                    writer.writerow(
                        {
                            "filename": detail["filename"],
                            **{
                                k: field[k]
                                for k in ("field", "ground_truth", "prediction", "exact")
                            },
                        }
                    )
    if not summaries:
        raise ValueError("No prediction folders found")
    candidates = [
        s for s in summaries if s["errors"] == 0 and s["invoices"] == 100 and s["cer"] is not None
    ]
    candidates.sort(key=lambda s: (s["cer"], s["wer"], s["mean_seconds_per_successful_invoice"]))
    full_comparison = {s["model"] for s in candidates} == set(cohort)
    report = {
        "metric_scope": scope,
        "normalization": "NFKC + collapsed whitespace; case and punctuation retained",
        "selection_rule": "Lowest corpus CER, then WER, then mean latency; only complete error-free runs eligible.",
        "warning": "Field scope measures shared rule-based extraction as well as OCR, not full-page transcription accuracy."
        if scope == "fields"
        else None,
        "requested_models": cohort,
        "all_requested_models_complete": full_comparison,
        "final_ocr_winner": candidates[0]["model"]
        if full_comparison and scope == "full-text"
        else None,
        "best_complete_run_for_scope": candidates[0]["model"] if candidates else None,
        "models": summaries,
    }
    suffix = ".single" if invoice else ""
    save_json(out / f"summary{suffix}.json", report)
    with (out / f"summary{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(row["model"], "CER:", row["cer"], "WER:", row["wer"], "errors:", row["errors"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=NAMES)
    parser.add_argument("--invoice", help="Exact filename, including extension")
    parser.add_argument("--metric-scope", choices=["fields", "full-text"], required=True)
    parser.add_argument("--text-reference", type=Path)
    args = parser.parse_args()
    evaluate(args.out, args.metric_scope, args.text_reference, args.models, args.invoice)
