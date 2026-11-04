"""Reject private artifacts, oversized files and recognizable credentials."""

import re
import sys
from pathlib import Path


def main(paths):
    failures = []
    private_dirs = {"data", "results", "models", "output", "local_backups"}
    documents = {
        "OCR_MODEL_BENCHMARK_REPORT.md",
        "INVOICE_PIPELINE_REPORT.md",
        "WHY_UNLIMITED_OCR_WAS_SELECTED.md",
    }
    patterns = [
        r"AIza[0-9A-Za-z_-]{35}",
        r"gsk_[0-9A-Za-z]{40,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    ]
    for name in paths:
        path = Path(name)
        if not path.is_file():
            continue
        if private_dirs.intersection(path.parts) or (
            path.name.startswith(".env") and path.name != ".env.example"
        ):
            failures.append(f"Private artifact: {name}")
        if "docs" in path.parts and path.name not in documents:
            failures.append(f"Unapproved report: {name}")
        if path.stat().st_size > 1_000_000:
            failures.append(f"File exceeds publication size limit: {name}")
        content = path.read_bytes().decode("utf-8", errors="replace")
        if any(re.search(pattern, content) for pattern in patterns):
            failures.append(f"Possible credential: {name}")
    for failure in failures:
        print(failure)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
