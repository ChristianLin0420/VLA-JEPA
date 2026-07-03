import unittest

import numpy as np
import pandas as pd

from scripts.analysis.droid_reach_summary import (
    DEPTH_LABELS,
    depth_label,
    summarize,
)

# Constructed effects: bypass - live = +0.02 everywhere; shuffled - live =
# +0.01 below the d=12 training horizon and 0 beyond it.
BYPASS_DELTA, SHUFFLED_EARLY_DELTA = 0.02, 0.01


def synthetic_frame(num_episodes=3, num_decisions=160):
    rows = []
    for episode in range(num_episodes):
        for decision in range(num_decisions):
            base = 0.5 + 0.001 * episode + 0.0001 * decision
            shared = {"dataset": "droid_lerobot", "suite": "droid",
                      "episode": episode, "decision": decision,
                      "J": -1, "lam": 1.0, "tf_loss": 0.0, "donor_episode": None}
            rows.append({**shared, "condition": "live", "mse": base,
                         "working_norm": 10.0 + 0.1 * decision})
            rows.append({**shared, "condition": "bypass", "mse": base + BYPASS_DELTA,
                         "working_norm": float("nan")})
            rows.append({**shared, "condition": "shuffled",
                         "mse": base + (SHUFFLED_EARLY_DELTA if decision < 12 else 0.0),
                         "working_norm": 10.0 + 0.1 * decision,
                         "donor_episode": (episode + 1) % num_episodes})
    return pd.DataFrame(rows)


class DepthLabelTest(unittest.TestCase):
    def test_bin_edges(self):
        cases = {0: "0-11", 11: "0-11", 12: "12-24", 24: "12-24", 25: "25-49",
                 49: "25-49", 50: "50-99", 99: "50-99", 100: "100-150+",
                 149: "100-150+", 300: "100-150+"}
        self.assertEqual(list(depth_label(list(cases))), list(cases.values()))


class SummarizeTest(unittest.TestCase):
    def setUp(self):
        self.rows = summarize(synthetic_frame(), n_boot=200, seed=0)
        self.by_key = {(r["condition"], r["depth_bin"]): r for r in self.rows}

    def test_full_grid_of_rows(self):
        self.assertEqual(
            [(r["condition"], r["depth_bin"]) for r in self.rows],
            [(c, b) for c in ("bypass", "shuffled") for b in DEPTH_LABELS],
        )

    def test_paired_dmse_and_counts(self):
        for label in DEPTH_LABELS:
            self.assertAlmostEqual(
                self.by_key[("bypass", label)]["dmse_mean"], BYPASS_DELTA, places=10)
        self.assertAlmostEqual(
            self.by_key[("shuffled", "0-11")]["dmse_mean"], SHUFFLED_EARLY_DELTA, places=10)
        self.assertAlmostEqual(
            self.by_key[("shuffled", "100-150+")]["dmse_mean"], 0.0, places=10)
        row = self.by_key[("bypass", "0-11")]
        self.assertEqual((row["pairs"], row["episodes"]), (3 * 12, 3))
        row = self.by_key[("bypass", "100-150+")]
        self.assertEqual((row["pairs"], row["episodes"]), (3 * 60, 3))

    def test_bootstrap_ci_brackets_the_mean_and_is_seeded(self):
        for row in self.rows:
            self.assertLessEqual(row["dmse_ci_lo"], row["dmse_mean"])
            self.assertLessEqual(row["dmse_mean"], row["dmse_ci_hi"])
        again = summarize(synthetic_frame(), n_boot=200, seed=0)
        self.assertEqual(self.rows, again)

    def test_working_norm_stats_come_from_live_rows(self):
        norms = 10.0 + 0.1 * np.arange(50, 100)  # identical across episodes
        row = self.by_key[("bypass", "50-99")]
        self.assertAlmostEqual(row["working_norm_mean"], float(norms.mean()), places=10)
        self.assertAlmostEqual(
            row["working_norm_p95"], float(np.percentile(np.tile(norms, 3), 95)), places=10)

    def test_missing_condition_is_loud(self):
        frame = synthetic_frame()
        with self.assertRaisesRegex(ValueError, "no paired"):
            summarize(frame[frame["condition"] != "shuffled"], n_boot=10, seed=0)


if __name__ == "__main__":
    unittest.main()
