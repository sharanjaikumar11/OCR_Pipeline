"""CLI: PDF/image or cached Unlimited-OCR -> selected LLM -> client-schema JSON."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from benchmark.data import ROOT, read_json, render, save_json
from invoice_pipeline import gemini
from invoice_pipeline.core import (
    PROMPT,
    decode_response,
    groq_request,
    payload,
    schema,
    source_warnings,
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def cached(path, source_dir):
    rec = read_json(path)
    if (
        rec.get("model") != "unlimited_ocr"
        or rec.get("error")
        or rec.get("diagnostic_only")
        or rec.get("scoring_eligible") is False
    ):
        raise ValueError("Not a successful baseline Unlimited-OCR record")
    name = rec.get("filename", "")
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Unsafe/missing source filename")
    source = source_dir / name
    if not source.is_file() or digest(source.read_bytes()) != rec.get("source_sha256"):
        raise ValueError("Missing source PDF/image or source hash mismatch")
    if not isinstance(rec.get("text"), str) or not rec["text"].strip():
        raise ValueError("Cached OCR is empty")
    return name, rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ocr-dir", type=Path)
    mode.add_argument("--input", type=Path, help="Single new PDF/image or directory")
    p.add_argument("--source-dir", type=Path, default=ROOT / "data/invoices")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--invoice", help="Exact filename filter")
    p.add_argument(
        "--exclude-invoice",
        action="append",
        default=[],
        help="Exclude an exact filename; repeat as needed",
    )
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--provider", choices=["gemini", "groq"], default="gemini")
    p.add_argument("--model", default=None)
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p.add_argument("--max-input-chars", type=int, default=24000)
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--allow-cloud-upload",
        action="store_true",
        help="Confirm permission to send invoice OCR to the selected provider",
    )
    p.add_argument("--trust-model-code", action="store_true")
    p.add_argument(
        "--source-repairs",
        action="store_true",
        help="Apply audited OCR-evidence corrections; retain raw LLM extraction",
    )
    args = p.parse_args()
    args.model = args.model or (
        "gemini-2.5-flash" if args.provider == "gemini" else "openai/gpt-oss-120b"
    )
    key_name = "GEMINI_API_KEY" if args.provider == "gemini" else "GROQ_API_KEY"
    if args.provider == "gemini" and args.model != "gemini-2.5-flash":
        p.error("Gemini adapter currently supports gemini-2.5-flash only")
    if (
        args.limit < 1
        or args.max_output_tokens < 1
        or args.max_input_chars < 1
        or args.interval < 0
    ):
        p.error("Invalid limit/token/interval arguments")
    if args.retry_errors and not args.resume:
        p.error("--retry-errors requires --resume")
    if not args.dry_run:
        if not args.allow_cloud_upload:
            p.error("Live extraction requires --allow-cloud-upload and company permission")
        if not os.environ.get(key_name):
            p.error(f"Set {key_name} locally; never paste it into chat")
    candidates = []
    if args.ocr_dir:
        for file in sorted(args.ocr_dir.glob("*.json")):
            # Metadata and comparison files are not invoice predictions.
            r = read_json(file)
            if isinstance(r, dict) and r.get("filename"):
                if not args.invoice or r["filename"] == args.invoice:
                    name, rec = cached(file, args.source_dir)
                    candidates.append((name, file, rec))
    else:
        files = [args.input] if args.input.is_file() else sorted(args.input.iterdir())
        for file in files:
            if file.is_file() and file.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg"):
                if not args.invoice or file.name == args.invoice:
                    candidates.append((file.name, file, None))
    candidates = sorted(
        (c for c in candidates if c[0] not in args.exclude_invoice), key=lambda c: c[0]
    )[: args.limit]
    if not candidates:
        p.error("No matching input invoices")
    if len({n for n, _, _ in candidates}) != len(candidates):
        p.error("Duplicate invoice names")
    fingerprint = dict(
        provider=args.provider,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=0 if args.provider == "gemini" else None,
        max_input_chars=args.max_input_chars,
        dry_run=args.dry_run,
        prompt_sha256=digest(PROMPT.encode()),
        schema=schema(),
        code_hashes={
            n: digest((ROOT / n).read_bytes())
            for n in (
                "run_invoice_pipeline.py",
                "invoice_pipeline/core.py",
                "invoice_pipeline/gemini.py",
                "benchmark/models.py",
                "benchmark/compat.py",
                "benchmark/data.py",
            )
        },
        inputs=[dict(filename=n, input_sha256=digest(f.read_bytes())) for n, f, _ in candidates],
    )
    if args.source_repairs:
        fingerprint["source_repairs_sha256"] = digest(
            (ROOT / "invoice_pipeline/source_repairs.py").read_bytes()
        )
    if args.out.exists():
        if not args.resume:
            p.error("Output exists; use a new folder or --resume")
        if read_json(args.out / "run.json")["config"] != fingerprint:
            p.error("Resume configuration or inputs changed; use a new output folder")
    else:
        args.out.mkdir(parents=True)
        save_json(
            args.out / "run.json",
            dict(config=fingerprint, created_utc=datetime.now(timezone.utc).isoformat()),
        )
    lock = args.out / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        p.error("Run locked; verify no process is active before removing a stale lock")
    os.close(fd)
    engine = None
    outcomes = []
    try:
        for index, (name, file, rec) in enumerate(candidates):
            folder = args.out / digest(name.encode())[:20]
            folder.mkdir(exist_ok=True)
            status_path = folder / "status.json"
            if status_path.exists():
                previous = read_json(status_path)
                if previous["status"] in ("completed", "dry_run") or not args.retry_errors:
                    outcomes.append(previous)
                    print(name, previous["status"], "(kept)", flush=True)
                    continue
            attempt = folder / (
                "attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
            )
            attempt.mkdir()
            state = dict(filename=name, status="started", attempt=str(attempt.resolve()))
            save_json(status_path, state)
            start = time.perf_counter()
            try:
                if rec is None and args.dry_run:
                    state.update(status="dry_run", note="New input: OCR and cloud calls skipped")
                else:
                    if rec is None:
                        if not args.trust_model_code:
                            raise ValueError("New OCR requires --trust-model-code on CUDA server")
                        if engine is None:
                            from benchmark.models import load

                            engine = load(
                                "unlimited_ocr",
                                SimpleNamespace(
                                    device="cuda",
                                    models_dir=ROOT / "models",
                                    max_tokens=4096,
                                    trust_model_code=True,
                                ),
                            )
                        page_records = []
                        with tempfile.TemporaryDirectory() as temp:
                            for num, page in enumerate(render(file, Path(temp), 100), 1):
                                engine.sync()
                                tick = time.perf_counter()
                                text = engine.predict(page)
                                engine.sync()
                                page_records.append(
                                    dict(page=num, text=text, seconds=time.perf_counter() - tick)
                                )
                        rec = dict(
                            filename=name,
                            model="unlimited_ocr",
                            error=None,
                            source_sha256=digest(file.read_bytes()),
                            text="\n\n".join(
                                f"PAGE {r['page']}\n{r['text']}" for r in page_records
                            ),
                            pages=page_records,
                        )
                    save_json(attempt / "ocr.json", rec)
                    body = (gemini.payload if args.provider == "gemini" else payload)(
                        rec["text"], args.model, args.max_output_tokens
                    )
                    input_chars = len(PROMPT) + len(rec["text"]) + len(json.dumps(schema()))
                    state.update(
                        input_characters=input_chars,
                        note="Character guard is not an exact API token count; legacy cached text has no invented page splits",
                    )
                    if input_chars > args.max_input_chars:
                        raise ValueError(
                            "Input exceeds configured character guard; not truncated. Review limit/context before retry"
                        )
                    if args.dry_run:
                        state["status"] = "dry_run"
                    else:
                        request_start = time.perf_counter()
                        try:
                            response = (
                                gemini.request(body, args.model)
                                if args.provider == "gemini"
                                else groq_request(body)
                            )
                        finally:
                            state["llm_request_seconds_including_retries"] = (
                                time.perf_counter() - request_start
                            )
                        save_json(attempt / "response.json", response)
                        data, warnings = (
                            gemini.decode if args.provider == "gemini" else decode_response
                        )(response)
                        if args.source_repairs:
                            from invoice_pipeline.core import validate
                            from invoice_pipeline.source_repairs import repair

                            save_json(attempt / "extraction.raw.json", data)
                            data, repairs = repair(data, rec["text"])
                            save_json(attempt / "repairs.json", repairs)
                            warnings = validate(data)
                            state["source_repairs"] = len(repairs)
                        warnings.extend(source_warnings(data, rec["text"]))
                        save_json(attempt / "extraction.json", data)
                        state.update(
                            status="completed",
                            needs_review=bool(warnings),
                            warnings=warnings,
                            provider=args.provider,
                            usage=response.get("usageMetadata")
                            if args.provider == "gemini"
                            else response.get("usage"),
                            returned_model=response.get("modelVersion")
                            if args.provider == "gemini"
                            else response.get("model"),
                        )
            except Exception as exc:
                # Avoid arbitrary third-party exception text leaking key/data into logs.
                safe = (
                    str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
                )
                key = os.environ.get(key_name, "")
                if key:
                    safe = safe.replace(key, "[REDACTED]")
                state.update(status="failed", error=safe[:500])
            finally:
                state["processing_seconds"] = time.perf_counter() - start
                save_json(status_path, state)
                outcomes.append(state)
                save_json(
                    args.out / "summary.json",
                    dict(
                        invoices=outcomes,
                        completed=sum(x["status"] == "completed" for x in outcomes),
                        failed=sum(x["status"] == "failed" for x in outcomes),
                        dry_run=args.dry_run,
                    ),
                )
            print(name, state["status"], state.get("error", ""), flush=True)
            if not args.dry_run and index + 1 < len(candidates):
                time.sleep(args.interval)
    finally:
        if engine is not None:
            engine.close()
        lock.unlink(missing_ok=True)
    save_json(
        args.out / "summary.json",
        dict(
            invoices=outcomes,
            completed=sum(x["status"] == "completed" for x in outcomes),
            failed=sum(x["status"] == "failed" for x in outcomes),
            dry_run=args.dry_run,
        ),
    )
    return int(any(x["status"] not in ("completed", "dry_run") for x in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
