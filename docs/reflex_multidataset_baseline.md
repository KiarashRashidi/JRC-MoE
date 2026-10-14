# REFLEX baseline across four datasets

This extension runs the existing `reflex` routing mode once for each of 40 prompts: 10 GSM8K, 10 MATH-500, 10 MBPP, and 10 HumanEval. The historical paper reproduction remains the original 50 GSM8K prompts × 3 modes and is not overwritten.

## Protocol

- Checkpoint: `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`, revision `783d3467f108d28ac0a78d3e41af16ab05cabd8`.
- Mode: REFLEX only. No RRB-only or FGER-only rollouts are included in this 40-example baseline.
- Selection seed: `20261014`. Sampling is deterministic from stable task IDs. The GSM8K candidates are the existing 50-row Fixed8 diagnostic set, sampled without using correctness values. The selected IDs and hashes are saved before model inference.
- Input prompts use the existing model chat template. GSM8K reuses the original seeded 5-shot harness contexts. MBPP uses its pinned `lm_eval==0.4.8` task prompt and three-shot examples. HumanEval uses the pinned task's zero-shot function prompt. MATH-500 uses a zero-shot `Problem: ... Solution:` prompt.
- Output lengths follow the per-task config: 256 tokens for GSM8K and MBPP, 512 for MATH-500, and 1024 for HumanEval. Each is a multiple of the 32-token REFLEX block size. Therefore counts and timing are reported per dataset; they are not a same-token-budget comparison across datasets.
- Graders: frozen canonical numeric GSM8K grader; Minerva-style `math_verify` equivalence for MATH-500; official `lm_eval` code-evaluation `pass@1` checks for MBPP and HumanEval. The math and code grader smoke checks must pass before loading the model.
- The math grading extra is installable with `pip install -e '.[math-eval]'` and includes `math-verify` as well as the pinned `lm_eval[math]` dependencies.
- MBPP and HumanEval scoring executes generated Python in the `lm_eval` `code_eval` metric. This is executable code from model outputs, so run it only in an isolated, disposable account or container. The metric requires `HF_ALLOW_CODE_EVAL=1`; the preflight itself runs only a harmless synthetic addition function. If the evaluator or its cache cannot load, the run stops before GPU inference.
- Fixed8 patch parity is rechecked before the first REFLEX rollout. Full prompts, references, per-example generations, traces, grader outputs, errors, dataset/task revisions, and configuration hashes are saved for independent review.
- Output is resumable by dataset ID. A completed result is skipped only if both its result row and trace file exist. Failed attempts are retained in `failures.jsonl`.

## Commands on `research-gpu`

```bash
source /home/new_drive/rashidi/venvs/reflex-llada/bin/activate
export PYTHONPATH=src
export HF_HOME=/home/new_drive/rashidi/hf_cache
export HF_HUB_CACHE=/home/new_drive/rashidi/hf_cache/hub
export HF_ALLOW_CODE_EVAL=1
export JRC_MOE_GSM8K_DIR=/home/new_drive/rashidi/JRC-MoE-followup/data/gsm8k
pip install -e '.[math-eval]'
python -m jrc_moe.multidataset_baseline --stage prepare \
  --config configs/reflex_multidataset_40.yaml \
  --output-dir outputs/reflex_multidataset_40
python -m jrc_moe.multidataset_baseline --stage run \
  --config configs/reflex_multidataset_40.yaml \
  --output-dir outputs/reflex_multidataset_40
```

The preparation command must produce exactly 40 unique examples with ten per dataset. The run command must pass grader smoke checks and Fixed8 parity before inference. A missing dataset cache, incompatible task version, unavailable grader, parity failure, or resource-limit breach is a stop condition, not a reason to silently substitute data or scoring.

## Output files

- `cases.jsonl`: frozen prompts, references, selected IDs, grader names, and per-task generation lengths.
- `manifest.json`: dataset/task source versions, selected-case hashes, config/code identity, parity, and completion state.
- `results.jsonl`: completed per-example REFLEX outputs and metrics.
- `traces/`: decoder traces for each completed example.
- `failures.jsonl`: attempts that did not produce a valid saved result.
- `summary.csv`: per-dataset accuracy/pass@1, expert work, decoder forwards, and runtime.

With ten examples, accuracy is coarse: each example changes a dataset score by ten percentage points. These results are a cross-domain smoke baseline, not statistically precise benchmark estimates.
