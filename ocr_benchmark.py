"""Run each selected OCR model in an isolated process, then compare predictions."""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmark.compat import REVISION
from benchmark.data import ROOT, dataset, read_json, render, save_json
from benchmark.extraction import extract
from benchmark.models import NAMES, load


def adapter_fingerprint():
    digest = hashlib.sha256()
    for name in ("models.py", "compat.py"):
        digest.update((ROOT / "benchmark" / name).read_bytes())
    return digest.hexdigest()


def reusable_record(source, name, path, args):
    """Reuse only matching successful records; never modify the original run."""
    metadata = read_json(source / "metadata.json")
    if getattr(args, "strict_reuse", False) and metadata.get("compatibility_revision") != REVISION:
        raise ValueError("Resume requires results from the current compatibility revision")
    if (
        getattr(args, "strict_reuse", False)
        and metadata.get("adapter_sha256") != adapter_fingerprint()
    ):
        raise ValueError("Adapter code changed since prior run; refusing reuse")
    for key, expected in [
        ("model", name),
        ("device", args.device),
        ("dpi", args.dpi),
        ("max_tokens", args.max_tokens),
    ]:
        if metadata.get(key) != expected:
            raise ValueError(f"Reuse configuration mismatch for {key}")
    saved = source / (path.name + ".json")
    if not saved.exists():
        return None
    record = read_json(saved)
    if record.get("error"):
        return None
    if record.get("filename") != path.name or record.get("model") != name:
        raise ValueError(f"Reuse identity mismatch: {saved}")
    if record.get("source_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"Reuse source hash mismatch: {saved}")
    if not isinstance(record.get("text"), str) or record.get("inference_seconds") is None:
        raise ValueError(f"Incomplete saved result: {saved}")
    return record


def worker(args):
    name = args.worker
    folder = args.out / name
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": name,
        "device": "cpu" if name == "tesseract" else args.device,
        "platform": platform.platform(),
        "python": sys.version,
        "dpi": args.dpi,
        "max_tokens": args.max_tokens,
        "packages": {},
        "invoices_expected": 100,
        "compatibility_revision": REVISION,
        "adapter_sha256": adapter_fingerprint(),
    }
    for package in [
        "torch",
        "transformers",
        "paddleocr",
        "paddlepaddle-gpu",
        "easyocr",
        "pypdfium2",
    ]:
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    engine = None
    started = time.perf_counter()
    try:
        data = dataset()
        source = getattr(args, "reuse_success_from", None)
        if source:
            remaining = []
            for path, reference in data:
                record = reusable_record(source, name, path, args)
                if record is None:
                    remaining.append((path, reference))
                else:
                    save_json(folder / (path.name + ".json"), record)
                    print(name, path.name, "reused successful result", flush=True)
            metadata.update(reused_successes=len(data) - len(remaining), reuse_source=str(source))
            prior = read_json(source / "metadata.json")
            for key in ("parameters", "parameter_note", "runtime_settings", "gpu"):
                if key in prior:
                    metadata[key] = prior[key]
            data = remaining
            if not data:
                return
        t = time.perf_counter()
        engine = load(name, args)
        metadata.update(
            load_seconds=time.perf_counter() - t,
            parameters=engine.parameters,
            parameter_note=engine.parameter_note,
        )
        metadata["runtime_settings"] = getattr(engine, "runtime_settings", {})
        if engine.device == "cuda" and name not in ("paddleocr", "nanonets_ocr2"):
            import torch

            metadata["gpu"] = torch.cuda.get_device_name(0)
        if name == "nanonets_ocr2":
            metadata["offload_note"] = (
                "Confirm actual GPU offload in llama-server.log; device records requested device."
            )
        save_json(folder / "metadata.json", metadata)
        # One page of the first invoice is warmed up, then excluded from measurements.
        with tempfile.TemporaryDirectory() as temp:
            first = render(data[0][0], Path(temp), args.dpi)[0]
            engine.sync()
            t = time.perf_counter()
            try:
                engine.predict(first)
            except Exception as exc:
                metadata["warmup_error"] = f"{type(exc).__name__}: {exc}"
            engine.sync()
            metadata["warmup_seconds"] = time.perf_counter() - t
        batch_start = time.perf_counter()
        for path, reference in data:
            record = {
                "filename": path.name,
                "model": name,
                "error": None,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            invoice_start = time.perf_counter()
            texts = []
            durations = []
            try:
                with tempfile.TemporaryDirectory() as temp:
                    t = time.perf_counter()
                    pages = render(path, Path(temp), args.dpi)
                    record["render_seconds"] = time.perf_counter() - t
                    texts = []
                    durations = []
                    for page in pages:
                        engine.sync()
                        t = time.perf_counter()
                        text = engine.predict(page)
                        engine.sync()
                        durations.append(time.perf_counter() - t)
                        if not isinstance(text, str):
                            raise TypeError("Adapter must return text")
                        texts.append(text)
                    record.update(
                        text="\n\n".join(texts),
                        page_count=len(pages),
                        page_seconds=durations,
                        inference_seconds=sum(durations),
                    )
                    # Ground truth is deliberately not passed to the extractor or model.
                    record["invoiceDataArray"] = extract(record["text"])
            except Exception as exc:
                record.update(
                    error=f"{type(exc).__name__}: {exc}",
                    text="\n\n".join(texts),
                    invoiceDataArray={},
                    completed_page_seconds=durations,
                    partial_page_text=getattr(exc, "partial_text", ""),
                )
            record["processing_seconds"] = time.perf_counter() - invoice_start
            save_json(folder / (path.name + ".json"), record)
            print(name, path.name, record["error"] or "saved", flush=True)
        metadata["total_processing_seconds"] = time.perf_counter() - batch_start
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        metadata["worker_wall_seconds"] = time.perf_counter() - started
        save_json(folder / "metadata.json", metadata)
        if engine is not None:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument(
        "--out", type=Path, required=True, help="New output directory; never overwrites a run"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--trocr", default="microsoft/trocr-base-printed", help="Cached HF ID or local directory"
    )
    parser.add_argument(
        "--llama-server", default="llama-server", help="Installed llama.cpp server executable"
    )
    parser.add_argument("--trust-model-code", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--metric-scope", choices=["fields", "full-text"], required=True)
    parser.add_argument("--text-reference", type=Path)
    parser.add_argument("--worker", choices=NAMES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--reuse-success-from",
        type=Path,
        help="Worker only: prior model folder with matching successful results",
    )
    parser.add_argument(
        "--strict-reuse",
        action="store_true",
        help="Require current compatibility revision for resume",
    )
    args = parser.parse_args()
    if args.reuse_success_from and not args.worker:
        parser.error("--reuse-success-from requires --worker")
    if args.dpi <= 0 or args.max_tokens <= 0:
        parser.error("DPI and token limit must be positive")
    args.out = args.out.resolve()
    args.models_dir = args.models_dir.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.worker:
        worker(args)
        return
    data = dataset()
    if args.metric_scope == "full-text":
        texts = read_json(args.text_reference) if args.text_reference else {}
        missing = [
            p.name
            for p, r in data
            if not isinstance(texts.get(p.name), str) or not texts[p.name].strip()
        ]
        if missing:
            parser.error(
                f"{len(missing)} verified full-text references missing. Supply --text-reference or explicitly use --metric-scope fields."
            )
    if args.out.exists():
        parser.error("Output directory exists; choose a new run name")
    args.out.mkdir(parents=True)
    save_json(
        args.out / "run.json",
        {
            "models": args.models,
            "device_requested": args.device,
            "metric_scope": args.metric_scope,
            "invoices": 100,
            "dpi": args.dpi,
            "max_tokens": args.max_tokens,
            "selection": "No winner before all 11 complete full-text evaluation",
            "text_reference": str(args.text_reference) if args.text_reference else None,
            "timing": "Page inference includes input preparation and synchronous generation. Rendering, loading and warmup recorded separately. CPU Tesseract is not GPU latency.",
        },
    )
    for name in dict.fromkeys(args.models):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            "--out",
            str(args.out),
            "--device",
            args.device,
            "--models-dir",
            str(args.models_dir),
            "--trocr",
            args.trocr,
            "--llama-server",
            args.llama_server,
            "--dpi",
            str(args.dpi),
            "--max-tokens",
            str(args.max_tokens),
            "--metric-scope",
            args.metric_scope,
        ]
        if args.trust_model_code:
            command.append("--trust-model-code")
        with (args.out / f"{name}.log").open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        if process.returncode:
            print(f"{name} worker failed; see its log", file=sys.stderr)
    from compare import evaluate

    evaluate(args.out, args.metric_scope, args.text_reference, args.models)


if __name__ == "__main__":
    main()
