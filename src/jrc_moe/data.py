"""Load the validated Fixed8 inputs and reconstruct the notebook's GSM8K contexts."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from lm_eval.api.task import ConfigurableTask
import yaml
from .config import FEWSHOT_SEED, NUM_FEWSHOT


SCHEMA_ALIASES = {
    "prompt_length": ("prompt_length", "prompt_token_count"),
    "total_pairs": ("total_pairs", "selected_expert_token_pairs"),
    "pairs_m": ("pairs_m", "selected_expert_token_pairs_m"),
}
REQUIRED_COLUMNS = {
    "test_idx", "diagnostic_ordinal", "prompt_length", "total_nfe",
    "total_pairs", "pairs_m", "prompt_token_count", "clean_generated_text",
}


def load_fixed8(path: str | Path) -> pd.DataFrame:
    """Read and validate the exact historical 50-row Fixed8 diagnostic."""
    frame = pd.read_csv(path)
    for canonical, candidates in SCHEMA_ALIASES.items():
        if canonical not in frame:
            source = next((name for name in candidates if name in frame), None)
            if source is None:
                raise ValueError(f"Missing {canonical}; tried {candidates}. Columns: {list(frame)}")
            frame[canonical] = frame[source]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Fixed8 diagnostic is missing columns: {sorted(missing)}")
    for left, right in (("prompt_length", "prompt_token_count"),
                        ("total_pairs", "selected_expert_token_pairs")):
        if right in frame and not np.array_equal(frame[left].astype("int64"), frame[right].astype("int64")):
            raise ValueError(f"Fixed8 aliases disagree: {left} and {right}")
    frame = frame.drop_duplicates("test_idx", keep="last").copy()
    if len(frame) != 50:
        raise ValueError(f"Expected exactly 50 Fixed8 rows; found {len(frame)}")
    frame["test_idx"] = frame["test_idx"].astype(int)
    return frame


def reconstruct_gsm8k_records(indices: list[int]) -> dict[int, dict]:
    """Rebuild lm-eval's seeded 5-shot contexts used by the source notebook."""
    local_root = os.environ.get("JRC_MOE_GSM8K_DIR")
    if local_root:
        root = Path(local_root).expanduser().resolve()
        train = root / "main" / "train-00000-of-00001.parquet"
        test = root / "main" / "test-00000-of-00001.parquet"
        if not train.is_file() or not test.is_file():
            raise FileNotFoundError(f"Expected local GSM8K parquet files under {root}/main")
        task_yaml = Path(__import__("lm_eval.tasks", fromlist=["__file__"]).__file__).parent / "gsm8k" / "gsm8k.yaml"
        config = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
        config["dataset_path"] = "parquet"
        config["dataset_name"] = None
        config["dataset_kwargs"] = {"data_files": {"train": str(train), "test": str(test)}}
        task = ConfigurableTask(config=config)
    else:
        # Preserve the standard online lm-eval path when no local mirror is set.
        from lm_eval.tasks import TaskManager, get_task_dict
        task = get_task_dict(["gsm8k"], task_manager=TaskManager())["gsm8k"]
    docs = list(task.test_docs())
    task.set_fewshot_seed(FEWSHOT_SEED)
    target = set(map(int, indices))
    records = {}
    for idx, doc in enumerate(docs[:max(target) + 1]):
        # ConfigurableTask's few-shot RNG is stateful. The notebook asks for
        # every preceding context, even when it only retains selected rows.
        context = task.fewshot_context(doc, num_fewshot=NUM_FEWSHOT)
        if idx in target:
            records[idx] = {
                "test_idx": idx,
                "question": doc["question"],
                "gold_full_answer": doc["answer"],
                "harness_context": context,
            }
    if set(records) != target:
        raise RuntimeError(f"Could not reconstruct GSM8K contexts: {sorted(target - set(records))}")
    return records
