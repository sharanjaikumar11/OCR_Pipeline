"""Lazy model adapters; importing this module does not load any model."""

import base64
import json
import math
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from benchmark.compat import checked_text, patch_mask, patch_unlimited

REPOS = {
    "unlimited_ocr": "baidu/Unlimited-OCR",
    "firered_ocr": "FireRedTeam/FireRed-OCR",
    "falcon_ocr": "tiiuae/Falcon-OCR",
    "nanonets_ocr2": "mradermacher/Nanonets-OCR2-3B-GGUF",
    "paddleocr_vl": "PaddlePaddle/PaddleOCR-VL-1.6",
}
NAMES = ["paddleocr", "easyocr_trocr", "tesseract", "easyocr", *REPOS]


class Adapter:
    parameters = None
    parameter_note = "Not exposed by this runtime; not inferred from file size."

    def sync(self):
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()

    def close(self):
        pass


def ordered(entries):
    if not entries:
        return ""
    import statistics

    height = statistics.median(
        (max(1, max((p[1] for p in b)) - min((p[1] for p in b))) for b, t in entries)
    )
    entries = sorted(entries, key=lambda e: min((p[1] for p in e[0])))
    rows = []
    for box, text in entries:
        y = min((p[1] for p in box))
        if not rows or abs(y - rows[-1][0]) > 0.6 * height:
            rows.append((y, []))
        rows[-1][1].append((min((p[0] for p in box)), text))
    return "\n".join((" ".join((t for x, t in sorted(row))) for y, row in rows))


class Traditional(Adapter):
    def __init__(self, name, args):
        self.name, self.device = (name, "cpu" if name == "tesseract" else args.device)
        if name == "tesseract":
            import pytesseract

            pytesseract.get_tesseract_version()
            self.parameter_note = "Tesseract engine: neural parameter count not comparable/exposed."
        elif name == "paddleocr":
            import paddle
            from paddleocr import PaddleOCR

            if self.device == "cuda" and (not paddle.is_compiled_with_cuda()):
                raise RuntimeError("Install GPU PaddlePaddle for the board; refusing CPU fallback")
            self.new_api = hasattr(PaddleOCR, "predict")
            if self.new_api:
                cached = Path.home() / ".paddlex/official_models"
                options = {}
                for kind, model in [
                    ("text_detection", "PP-OCRv5_server_det"),
                    ("text_recognition", "en_PP-OCRv5_mobile_rec"),
                    ("textline_orientation", "PP-LCNet_x1_0_textline_ori"),
                ]:
                    for filename in ("inference.json", "inference.pdiparams"):
                        if not (cached / model / filename).is_file():
                            raise FileNotFoundError(
                                f"Required cached PaddleOCR file missing: {cached / model / filename}"
                            )
                    options[kind + "_model_name"] = model
                    options[kind + "_model_dir"] = str(cached / model)
                self.engine = PaddleOCR(
                    device="gpu:0" if self.device == "cuda" else "cpu",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    **options,
                )
            else:
                self.engine = PaddleOCR(
                    lang="en", use_gpu=self.device == "cuda", use_angle_cls=True, show_log=False
                )
        else:
            import easyocr

            self.engine = easyocr.Reader(
                ["en"], gpu=self.device == "cuda", verbose=False, download_enabled=False
            )
            self.parameters = sum((p.numel() for p in self.engine.detector.parameters()))
            if name == "easyocr":
                self.parameters += sum((p.numel() for p in self.engine.recognizer.parameters()))
            else:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                self.processor = TrOCRProcessor.from_pretrained(args.trocr, local_files_only=True)
                self.model = (
                    VisionEncoderDecoderModel.from_pretrained(args.trocr, local_files_only=True)
                    .to(self.device)
                    .eval()
                )
                self.parameters += sum((p.numel() for p in self.model.parameters()))
            self.parameter_note = "Runtime parameter count of active detector and recognizer; excludes unused EasyOCR recognizer in TrOCR hybrid."

    def sync(self):
        if self.device == "cuda" and self.name == "paddleocr":
            import paddle

            paddle.device.synchronize()
        else:
            super().sync()

    def predict(self, path):
        if self.name == "tesseract":
            import pytesseract
            from PIL import Image

            with Image.open(path) as image:
                return pytesseract.image_to_string(image, config="--psm 3")
        if self.name == "easyocr":
            return ordered([(b, t) for b, t, c in self.engine.readtext(str(path))])
        if self.name == "paddleocr":
            if self.new_api:
                entries = []
                for r in self.engine.predict(str(path)):
                    entries.extend(zip(r["rec_polys"], r["rec_texts"]))
                return ordered(entries)
            result = self.engine.ocr(str(path), cls=True)
            return ordered([(b, t[0]) for b, t in result[0] or []])
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            horizontal, free = self.engine.detect(np.array(image))
            boxes = [(x0, y0, x1, y1) for x0, x1, y0, y1 in horizontal[0]]
            boxes += [
                (
                    min((p[0] for p in b)),
                    min((p[1] for p in b)),
                    max((p[0] for p in b)),
                    max((p[1] for p in b)),
                )
                for b in free[0]
            ]
            entries = []
            for x0, y0, x1, y1 in boxes:
                x0, y0, x1, y1 = (
                    max(0, int(x0)),
                    max(0, int(y0)),
                    min(image.width, int(x1)),
                    min(image.height, int(y1)),
                )
                if x1 <= x0 or y1 <= y0:
                    continue
                pixels = self.processor(
                    images=image.crop((x0, y0, x1, y1)), return_tensors="pt"
                ).pixel_values.to(self.device)
                with torch.inference_mode():
                    ids = self.model.generate(pixels, max_new_tokens=256)
                entries.append(
                    (
                        [(x0, y0), (x1, y1)],
                        self.processor.batch_decode(ids, skip_special_tokens=True)[0],
                    )
                )
            return ordered(entries)


class TransformerOCR(Adapter):
    def __init__(self, name, args):
        import torch
        import transformers as tr

        self.name, self.device, self.limit = (name, args.device, args.max_tokens)
        path = str(args.models_dir / REPOS[name])
        if name in ("unlimited_ocr", "falcon_ocr", "paddleocr_vl") and (not args.trust_model_code):
            raise ValueError(
                "Review downloaded model code and supply --trust-model-code to enable it"
            )
        common = dict(local_files_only=True, trust_remote_code=args.trust_model_code)
        dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
            else torch.float16
            if self.device == "cuda"
            else torch.float32
        )
        if name in ("falcon_ocr", "paddleocr_vl"):
            dtype = torch.float32
        if name == "unlimited_ocr":
            dtype = torch.float16
        cls = {
            "unlimited_ocr": "AutoModel",
            "falcon_ocr": "AutoModelForCausalLM",
            "firered_ocr": "Qwen3VLForConditionalGeneration",
        }.get(name, "AutoModelForImageTextToText")
        if name == "unlimited_ocr" and self.device != "cuda":
            raise ValueError("Unlimited-OCR downloaded custom inference requires CUDA")
        model_class = getattr(tr, cls)
        if name == "paddleocr_vl":
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            model_class = get_class_from_dynamic_module(
                "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration",
                path,
                local_files_only=True,
            )
            patch_mask(model_class)
        options = (
            {"attn_implementation": "eager"} if name in ("unlimited_ocr", "paddleocr_vl") else {}
        )
        self.model = (
            model_class.from_pretrained(path, torch_dtype=dtype, **common, **options)
            .to(self.device)
            .eval()
        )
        self.runtime_settings = {
            "dtype": str(dtype),
            "attention": options.get("attn_implementation", "model default"),
            "use_cache": True,
        }
        self.parameters = sum((p.numel() for p in self.model.parameters()))
        self.parameter_note = "Loaded model parameter count, including vision components."
        if name == "unlimited_ocr":
            self.processor = tr.AutoTokenizer.from_pretrained(path, **common)
        elif name != "falcon_ocr":
            self.processor = tr.AutoProcessor.from_pretrained(path, **common)
        if name == "unlimited_ocr":
            patch_unlimited(self.model, self.processor, self.limit)
        if name == "falcon_ocr":
            self.processor = tr.PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
            self.model._tokenizer = self.processor

    def predict(self, path):
        import torch
        from PIL import Image

        with Image.open(path) as source, torch.inference_mode():
            image = source.convert("RGB")
            if self.name == "falcon_ocr":
                original = self.processor.decode
                statuses = []

                def decode(ids, *args, **kwargs):
                    stops = [
                        self.model.config.eos_id,
                        self.processor.convert_tokens_to_ids("<|end_of_query|>"),
                    ]
                    statuses.append(bool(len(ids) and int(ids[-1]) in stops))
                    return original(ids, *args, **kwargs)

                self.processor.decode = decode
                try:
                    result = self.model.generate(
                        image, category="plain", max_new_tokens=self.limit, compile=False
                    )[0]
                finally:
                    self.processor.decode = original
                return checked_text(result, bool(statuses) and statuses[-1])
            if self.name == "unlimited_ocr":
                with tempfile.TemporaryDirectory() as out:
                    result = self.model.infer(
                        self.processor,
                        prompt="<image>document parsing.",
                        image_file=str(path),
                        output_path=out,
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        eval_mode=True,
                        save_results=False,
                        max_length=self.limit,
                        no_repeat_ngram_size=35,
                        ngram_window=128,
                    )
                    if not isinstance(result, str):
                        raise TypeError("Unexpected Unlimited-OCR result type")
                    return checked_text(result, self.model._last_generation["ended_with_eos"])
            prompts = {
                "paddleocr_vl": "OCR:",
                "firered_ocr": "Transcribe this document faithfully as Markdown, with tables as HTML. Preserve reading order. Do not infer missing text or add explanations.",
            }
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompts[self.name]},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            limit = self.limit
            ids = self.model.generate(
                **inputs, max_new_tokens=limit, do_sample=False, use_cache=True
            )
            tail = ids[:, inputs["input_ids"].shape[-1] :]
            text = self.processor.batch_decode(tail, skip_special_tokens=True)[0]
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            checked_text(text, bool(tail.shape[-1] and tail[0, -1].item() in eos))
            return text


class Nanonets(Adapter):
    """Own a loopback-only llama.cpp process. No FastAPI or external OCR service."""

    def __init__(self, args):
        import socket

        from gguf import GGUFReader

        self.device = args.device
        folder = args.models_dir / REPOS["nanonets_ocr2"]
        weights = folder / "Nanonets-OCR2-3B.f16.gguf"
        projector = folder / "Nanonets-OCR2-3B.mmproj-f16.gguf"
        self.parameters = sum(
            (
                math.prod(t.shape)
                for file in (weights, projector)
                for t in GGUFReader(str(file)).tensors
            )
        )
        self.parameters = int(self.parameters)
        self.parameter_note = "Stored tensor elements in GGUF language model plus vision projector; may differ from tied runtime counts."
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.limit = args.max_tokens
        context = getattr(args, "llama_context", 4096)
        if context <= self.limit:
            raise ValueError(
                "llama context must exceed output limit to leave room for image/prompt tokens"
            )
        self.runtime_settings = {"context_tokens": context, "max_output_tokens": self.limit}
        self.log = open(args.out / "llama-server.log", "w", encoding="utf-8")
        command = [
            args.llama_server,
            "-m",
            str(weights),
            "--mmproj",
            str(projector),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(context),
            "--parallel",
            "1",
            "-ngl",
            "99" if args.device == "cuda" else "0",
        ]
        if args.device == "cpu":
            command.append("--no-mmproj-offload")
        self.process = subprocess.Popen(
            command,
            stdout=self.log,
            stderr=self.log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for _ in range(300):
                if self.process.poll() is not None:
                    raise RuntimeError("llama-server exited; inspect its log")
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError("llama-server startup timed out")
        except BaseException:
            self.close()
            raise

    def sync(self):
        pass

    def predict(self, path):
        image = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image}},
                        {
                            "type": "text",
                            "text": "Extract all text from this document in reading order. Preserve tables and do not invent content.",
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": self.limit,
            "cache_prompt": False,
        }
        request = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.load(response)
        choice = result["choices"][0]
        return checked_text(choice["message"]["content"], choice.get("finish_reason") == "stop")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def load(name, args):
    if args.device == "cuda" and name not in ("tesseract", "paddleocr", "nanonets_ocr2"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing implicit CPU fallback")
    if name == "nanonets_ocr2":
        return Nanonets(args)
    if name in REPOS:
        return TransformerOCR(name, args)
    return Traditional(name, args)
