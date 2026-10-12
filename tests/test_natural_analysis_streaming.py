import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jrc_moe.natural_analysis import _iter_trajectory_records


class StreamingAnalysisTests(unittest.TestCase):
    def test_loads_one_trajectory_at_a_time(self):
        with tempfile.TemporaryDirectory() as temp:
            items = []
            for prompt_id in ("p1", "p2"):
                raw = Path(temp) / prompt_id
                raw.mkdir()
                (raw / "prompt_output.json").write_text(json.dumps({"prompt_id": prompt_id}))
                for name in ("events.jsonl.gz", "positions.jsonl.gz", "relations.jsonl.gz"):
                    (raw / name).touch()
                index = {"raw_path": str(raw), "prompt_id": prompt_id, "mode": "reflex"}
                items.append((("run", prompt_id, "reflex", 1), index))

            reads = []
            def read_jsonl(path):
                reads.append(Path(path))
                return [Path(path).name]

            with patch("jrc_moe.natural_analysis._read_jsonl_gz", side_effect=read_jsonl):
                records = _iter_trajectory_records(items)
                first = next(records)
                self.assertEqual(first["output"]["prompt_id"], "p1")
                self.assertEqual(len(reads), 3)
                second = next(records)
                self.assertEqual(second["output"]["prompt_id"], "p2")
                self.assertEqual(len(reads), 6)
                with self.assertRaises(StopIteration):
                    next(records)


if __name__ == "__main__":
    unittest.main()
