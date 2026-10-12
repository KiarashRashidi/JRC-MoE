import unittest

import pandas as pd

from jrc_moe.config import BLOCK_LENGTH, NUM_BLOCKS
from jrc_moe.generation import validate_trace


class TraceValidationTests(unittest.TestCase):
    def test_fixed8_trace_accepts_only_native_topk_budget(self):
        rows = []
        for block_idx in range(NUM_BLOCKS):
            rows.append({
                "block_idx": block_idx,
                "local_nfe": 1,
                "num_committed": BLOCK_LENGTH,
                "masked_after": 0,
                "masked_before": BLOCK_LENGTH,
                "forward_mean_k": 8.0,
                "k4_count": 0,
                "k6_count": 0,
                "k8_count": BLOCK_LENGTH,
                "k10_count": 0,
                "k12_count": 0,
                "k14_count": 0,
            })
        validate_trace(pd.DataFrame(rows), "fixed8")

        rows[0]["k8_count"] = BLOCK_LENGTH - 1
        rows[0]["k10_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "unexpected observed K set"):
            validate_trace(pd.DataFrame(rows), "fixed8")


if __name__ == "__main__":
    unittest.main()
