"""Frozen experiment settings copied from the supplied replication notebook."""

MODEL_ID = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
MODEL_REVISION = "783d3467f108d28ac0a78d3e41af16ab05cabd8d"
MASK_ID = 156895

GEN_LENGTH = 256
BLOCK_LENGTH = 32
NUM_BLOCKS = GEN_LENGTH // BLOCK_LENGTH
NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234
THETA = 0.90

NUM_MOE_LAYERS = 16
NUM_EXPERTS = 64
NATIVE_TOPK = 8

K_BASE = 8
DELTA_ROLE = 4
DELTA_CUR = 2
Q_OUTER = 0.25
W_CF = 0.30
V_REF = 0.01
TAU = 0.05

MODE_RRB = "rrb_only"
MODE_FGER = "fger_only"
MODE_REFLEX = "reflex"
RUN_MODES = [MODE_RRB, MODE_FGER, MODE_REFLEX]
RUN_ORDER = [MODE_RRB, MODE_REFLEX, MODE_FGER]

STOP_SEQUENCES = ["Question:", "</s>", "<|im_end|>"]
GRADER_VERSION = "canonical_numeric_v1.0.2"
FROZEN_GRADER_SHA256 = "f3e3f5963c5d2b5b22d7b74c657fb1bc524dec7ab15aaa6e0b8e6e0e09926787"

PAPER_TARGETS = [
    {"mode": "fixed8", "paper_gsm8k_acc_pct": 76.5, "paper_pairs_m": 32.2, "paper_avgk_across_tasks": 8.0},
    {"mode": MODE_RRB, "paper_gsm8k_acc_pct": 76.0, "paper_pairs_m": 27.6, "paper_avgk_across_tasks": 6.6},
    {"mode": MODE_FGER, "paper_gsm8k_acc_pct": 77.9, "paper_pairs_m": 32.6, "paper_avgk_across_tasks": 8.0},
    {"mode": MODE_REFLEX, "paper_gsm8k_acc_pct": 76.4, "paper_pairs_m": 27.7, "paper_avgk_across_tasks": 6.6},
]
