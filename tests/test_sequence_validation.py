import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from merge_ballot_logs import is_valid_preference_sequence, load_digit_runs


class TestSequenceValidation(unittest.TestCase):
    def test_accepts_contiguous_sequence_starting_at_one(self):
        self.assertTrue(is_valid_preference_sequence([1, 2, 3, 4]))

    def test_rejects_duplicate_preferences(self):
        self.assertFalse(is_valid_preference_sequence([1, 2, 2]))

    def test_rejects_gap_in_preferences(self):
        self.assertFalse(is_valid_preference_sequence([1, 2, 4]))

    def test_rejects_sequence_missing_one_and_two(self):
        self.assertFalse(is_valid_preference_sequence([3, 4]))

    def test_rejects_empty_preferences(self):
        self.assertFalse(is_valid_preference_sequence([]))

    def test_load_digit_runs_marks_duplicate_preferences_invalid(self):
        with TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "ballot_12345"
            results_dir.mkdir(parents=True)
            (results_dir / "results.json").write_text(
                json.dumps(
                    [
                        {"row": 1, "digit": 1},
                        {"row": 2, "digit": 2},
                        {"row": 3, "digit": 2},
                    ]
                ),
                encoding="utf-8",
            )

            records = load_digit_runs(temp_dir)

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["sequence_ok"])
        self.assertEqual(records[0]["numbers_found"], 3)


if __name__ == "__main__":
    unittest.main()
