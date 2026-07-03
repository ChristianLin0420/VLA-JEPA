import tempfile
import unittest
from pathlib import Path

from scripts.offline_eval.cache_tokens import (
    assert_selection_cached,
    load_episode_selection,
)


class LoadEpisodeSelectionTest(unittest.TestCase):
    def test_parses_lines_whitespace_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.txt"
            path.write_text("3\n1\n\n  2 3\n")
            self.assertEqual(load_episode_selection(path), {1, 2, 3})

    def test_empty_file_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.txt"
            path.write_text("\n")
            with self.assertRaisesRegex(ValueError, "empty episode selection"):
                load_episode_selection(path)

    def test_non_integer_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.txt"
            path.write_text("1\nep2\n")
            with self.assertRaises(ValueError):
                load_episode_selection(path)


class AssertSelectionCachedTest(unittest.TestCase):
    def test_passes_when_every_requested_episode_cached(self):
        assert_selection_cached({1, 2}, {1, 2, 5})

    def test_missing_episodes_fail_loudly_and_are_listed(self):
        with self.assertRaisesRegex(SystemExit, r"2 selected episodes not cached.*1, 3"):
            assert_selection_cached({1, 2, 3}, {2})


if __name__ == "__main__":
    unittest.main()
