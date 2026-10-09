"""Pinned LLaDA-MoE loading and the notebook's Fixed8 equivalence gate."""
import gc
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .config import MODEL_ID, MODEL_REVISION, NUM_MOE_LAYERS, NUM_EXPERTS, NATIVE_TOPK, MASK_ID, GEN_LENGTH, BLOCK_LENGTH
from .routing import VariableKController, install_variable_k_patch


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    cfg = model.config
    checks = {
        "num_hidden_layers": NUM_MOE_LAYERS,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": NATIVE_TOPK,
    }
    for key, expected in checks.items():
        if getattr(cfg, key, None) != expected:
            raise RuntimeError(f"Unexpected model config {key}={getattr(cfg, key, None)!r}; expected {expected}")
    if sum(cfg.moe_layer_freq) != NUM_MOE_LAYERS or cfg.moe_router_score_function != "softmax" or bool(cfg.norm_topk_prob):
        raise RuntimeError("Checkpoint MoE configuration differs from the notebook assumptions")
    return model, tokenizer


def input_device(model):
    return next(model.get_input_embeddings().parameters()).device


def build_model_input(context: str, tokenizer):
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": context}], add_generation_prompt=True, tokenize=False)
    ids = tokenizer(rendered, return_tensors="pt")["input_ids"]
    return rendered, ids


def fixed8_patch_equivalence(model, prompt_ids: torch.Tensor, controller: VariableKController) -> dict:
    """Require patched uniform-K logits and decisions to match native routing."""
    prompt_ids = prompt_ids.to(input_device(model))
    prompt_len = int(prompt_ids.shape[1])
    total_len = prompt_len + GEN_LENGTH
    x = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=prompt_ids.device)
    x[:, :prompt_len] = prompt_ids
    active = slice(prompt_len, prompt_len + BLOCK_LENGTH)
    with torch.no_grad():
        native = model(x).logits[:, active, :].detach().float().cpu()
    install_variable_k_patch(model, controller)
    controller.begin_forward(torch.full((total_len,), NATIVE_TOPK, dtype=torch.long))
    with torch.no_grad():
        patched = model(x).logits[:, active, :].detach().float().cpu()
    pairs = controller.end_forward()
    native_pred, patched_pred = native.argmax(-1), patched.argmax(-1)
    native_conf = F.softmax(native.double(), dim=-1).max(-1).values
    patched_conf = F.softmax(patched.double(), dim=-1).max(-1).values
    result = {
        "max_abs_logit_diff": float((native - patched).abs().max()),
        "argmax_match_fraction": float((native_pred == patched_pred).float().mean()),
        "max_confidence_diff": float((native_conf - patched_conf).abs().max()),
        "direct_pairs": pairs["direct_pairs"],
        "expected_pairs": NUM_MOE_LAYERS * NATIVE_TOPK * total_len,
    }
    if result["argmax_match_fraction"] != 1.0 or result["max_confidence_diff"] > 1e-6:
        raise RuntimeError(f"Variable-K patch failed Fixed8 equivalence: {result}")
    if result["direct_pairs"] != result["expected_pairs"]:
        raise RuntimeError(f"Variable-K pair-count gate failed: {result}")
    del native, patched, native_pred, patched_pred, native_conf, patched_conf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
