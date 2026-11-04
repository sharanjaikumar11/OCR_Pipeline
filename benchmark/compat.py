"""Scoped compatibility fixes verified on the Quadro RTX 5000.

Never edits downloaded model files or global torch functions.
"""

import inspect
import sys
import textwrap
import types

REVISION = "quadro-tested-v1"


def patch_mask(model_class):
    module = sys.modules[model_class.__module__]
    original = module.create_causal_mask
    if "input_embeds" not in inspect.signature(original).parameters:
        return

    def compatible(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original(*args, **kwargs)

    module.create_causal_mask = compatible


def patch_unlimited(model, tokenizer, limit):
    import torch

    embeddings = model.model.vision_model.embeddings
    with torch.no_grad():
        embeddings.position_ids.copy_(
            torch.arange(
                embeddings.position_ids.shape[-1],
                device=embeddings.position_ids.device,
                dtype=embeddings.position_ids.dtype,
            ).unsqueeze(0)
        )
    method = model.infer.__func__
    source = textwrap.dedent(inspect.getsource(method))
    if "torch.bfloat16" not in source:
        raise RuntimeError("Unlimited source changed; compatibility patch needs review")
    namespace = dict(method.__globals__)
    exec(
        compile(
            source.replace("torch.bfloat16", "torch.float16"), "<unlimited-fp16-compat>", "exec"
        ),
        namespace,
    )
    model.infer = types.MethodType(namespace["infer"], model)
    original = model.generate

    def generate(*args, **kwargs):
        kwargs.pop("max_length", None)
        kwargs.update(
            max_new_tokens=limit,
            attention_mask=torch.ones_like(kwargs["input_ids"]),
            pad_token_id=tokenizer.eos_token_id,
        )
        result = original(*args, **kwargs)
        tail = result[0, kwargs["input_ids"].shape[1] :]
        model._last_generation = {
            "generated_tokens": int(tail.numel()),
            "ended_with_eos": bool(tail.numel() and tail[-1].item() == tokenizer.eos_token_id),
        }
        return result

    model.generate = generate


def checked_text(text, completed):
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Empty or invalid OCR output")
    if not completed:
        error = RuntimeError(
            "Generation did not end naturally; output retained but not eligible for scoring"
        )
        error.partial_text = text
        raise error
    return text
