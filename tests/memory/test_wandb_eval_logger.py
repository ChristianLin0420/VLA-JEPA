import types
import unittest

import pandas as pd

from scripts.analysis.log_eval_to_wandb import (
    build_payload,
    decision_index_means,
    episode_table,
    log_to_wandb,
    offline_burnin_curve,
    offline_lambda_curve,
    offline_paired_dmse,
    sanitize_run_id,
    step_group,
    suite_summary,
)


def _episode(suite, episode_idx, success, **extra):
    record = {
        "suite": suite, "task_id": 0, "episode_idx": episode_idx,
        "memory_mode": "live", "memory_params": {}, "success": success,
        "ckpt": "/x/ck.pt", "ckpt_sha": "abc", "git_sha": "def",
        "blackout": {"start_decision": -1, "num_decisions": 0},
    }
    record.update(extra)
    return record


def _decision(decision_index, injection_ratio, working_norm, cf=None):
    return {
        "episode_idx": 0, "decision_index": decision_index,
        "injection_ratio": injection_ratio, "working_norm": working_norm,
        "cf_delta_action_l2": cf, "update_gate_mean": None,
    }


def _offline_frame(rows):
    return pd.DataFrame(
        rows, columns=["suite", "episode", "decision", "condition", "J", "lam", "mse"]
    )


class RunIdTest(unittest.TestCase):
    def test_clean_names_pass_through(self):
        self.assertEqual(
            sanitize_run_id("VLA-JEPA-memv1-live-step_34729"),
            "VLA-JEPA-memv1-live-step_34729",
        )

    def test_bad_chars_collapse_to_dashes(self):
        self.assertEqual(sanitize_run_id("a b/c:d"), "a-b-c-d")
        self.assertEqual(sanitize_run_id("//weird name//"), "weird-name")

    def test_length_cap_and_empty_rejection(self):
        self.assertEqual(len(sanitize_run_id("x" * 200)), 128)
        with self.assertRaisesRegex(ValueError, "run id"):
            sanitize_run_id("///")

    def test_step_group_parsing(self):
        self.assertEqual(step_group("VLA-JEPA-memv1-live-step_34729"), "step_34729")
        self.assertEqual(step_group("VLA-JEPA-cotrain-step2000"), "step_2000")
        self.assertEqual(step_group("no-tag-here"), "no-tag-here")


class EpisodeAggregationTest(unittest.TestCase):
    def test_suite_summary_per_suite_and_pooled(self):
        summary = suite_summary({
            "libero_goal": [_episode("libero_goal", i, i < 3) for i in range(4)],
            "libero_10": [_episode("libero_10", i, i == 0) for i in range(2)],
        })
        self.assertEqual(summary["success_rate/libero_goal"], 0.75)
        self.assertEqual(summary["success_rate/libero_10"], 0.5)
        self.assertEqual(summary["episodes/pooled"], 6)
        self.assertAlmostEqual(summary["success_rate/pooled"], 4 / 6)

    def test_suite_summary_empty(self):
        self.assertEqual(suite_summary({}), {})

    def test_episode_table_union_and_json_cells(self):
        episodes = [
            _episode("libero_goal", 0, True),
            _episode("libero_goal", 1, False, donor_source="libero_10--0--ep1"),
        ]
        columns, rows = episode_table(episodes)
        self.assertIn("donor_source", columns)
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0][columns.index("donor_source")])
        self.assertEqual(rows[0][columns.index("memory_params")], "{}")
        self.assertIn("start_decision", rows[0][columns.index("blackout")])

    def test_decision_index_means_skips_missing_values(self):
        columns, rows = decision_index_means([
            _decision(0, 0.5, 1.0),
            _decision(0, 0.7, None),
            _decision(1, 0.4, 2.0),
        ])
        self.assertEqual(columns, ["decision_index", "injection_ratio", "working_norm", "count"])
        self.assertEqual(rows[0][0], 0)
        self.assertAlmostEqual(rows[0][1], 0.6)
        self.assertEqual(rows[0][2:], [1.0, 2])
        self.assertEqual(rows[1], [1, 0.4, 2.0, 1])

    def test_build_payload_config_trials_and_histograms(self):
        arms = {"libero_goal": {
            "episodes": [_episode("libero_goal", i, False) for i in range(10)],
            "decisions": [_decision(0, 0.5, 1.0), _decision(1, 0.3, 2.0, cf=0.1)],
        }}
        payload = build_payload("VLA-JEPA-memv1-live-step_34729", arms, None)
        self.assertEqual(payload["config"]["trials"], 10)
        self.assertEqual(payload["config"]["memory_mode"], "live")
        self.assertEqual(sorted(payload["histograms"]),
                         ["cf_delta_action_l2", "injection_ratio", "working_norm"])
        self.assertEqual(payload["histograms"]["cf_delta_action_l2"], [0.1])
        self.assertIsNone(payload["offline"])


class OfflineAggregationTest(unittest.TestCase):
    def test_paired_dmse_vs_live(self):
        frame = _offline_frame([
            ("libero_goal", 0, 0, "live", -1, 1.0, 1.0),
            ("libero_goal", 0, 1, "live", -1, 1.0, 2.0),
            ("libero_goal", 0, 0, "bypass", -1, 1.0, 1.5),
            ("libero_goal", 0, 1, "bypass", -1, 1.0, 3.5),
            # lambda-sweep + burn-in rows must not leak into the base battery
            ("libero_goal", 0, 0, "live", -1, 4.0, 99.0),
            ("libero_goal", 0, 0, "burnin_forward", 8, 1.0, 99.0),
            # unpaired decision (no live partner) is dropped
            ("libero_goal", 0, 2, "bypass", -1, 1.0, 99.0),
        ])
        rows = offline_paired_dmse(frame, conditions=("bypass",))
        self.assertEqual(len(rows), 2)  # per-suite + pooled
        self.assertEqual(rows[0]["suite"], "libero_goal")
        self.assertAlmostEqual(rows[0]["dmse"], 1.0)
        self.assertEqual(rows[0]["pairs"], 2)
        self.assertEqual(rows[1]["suite"], "pooled")
        self.assertAlmostEqual(rows[1]["dmse"], 1.0)

    def test_paired_dmse_dedupes_duplicate_live_rows(self):
        # base battery + lam sweep both emit live at lam == 1 -> mean, not double count
        frame = _offline_frame([
            ("libero_goal", 0, 0, "live", -1, 1.0, 1.0),
            ("libero_goal", 0, 0, "live", -1, 1.0, 3.0),
            ("libero_goal", 0, 0, "prior", -1, 1.0, 5.0),
        ])
        rows = offline_paired_dmse(frame, conditions=("prior",))
        self.assertAlmostEqual(rows[1]["dmse"], 3.0)

    def test_burnin_curve(self):
        frame = _offline_frame([
            ("libero_goal", 0, 0, "burnin_forward", 1, 1.0, 2.0),
            ("libero_goal", 0, 1, "burnin_forward", 1, 1.0, 4.0),
            ("libero_goal", 0, 0, "burnin_reversed", 1, 1.0, 6.0),
            ("libero_goal", 0, 0, "burnin_forward", 8, 1.0, 1.0),
            ("libero_goal", 0, 0, "live", -1, 1.0, 99.0),
        ])
        rows = offline_burnin_curve(frame)
        self.assertEqual(
            rows,
            [
                {"J": 1, "order": "forward", "mse": 3.0, "rows": 2},
                {"J": 1, "order": "reversed", "mse": 6.0, "rows": 1},
                {"J": 8, "order": "forward", "mse": 1.0, "rows": 1},
            ],
        )

    def test_lambda_curve(self):
        frame = _offline_frame([
            ("libero_goal", 0, 0, "live", -1, 0.0, 2.0),
            ("libero_goal", 0, 1, "live", -1, 0.0, 4.0),
            ("libero_goal", 0, 0, "shuffled", -1, 2.0, 5.0),
            ("libero_goal", 0, 0, "burnin_forward", 8, 1.0, 99.0),  # excluded (J != -1)
            ("libero_goal", 0, 0, "bypass", -1, 1.0, 99.0),  # excluded condition
        ])
        rows = offline_lambda_curve(frame)
        self.assertEqual(
            rows,
            [
                {"condition": "live", "lam": 0.0, "mse": 3.0, "rows": 2},
                {"condition": "shuffled", "lam": 2.0, "mse": 5.0, "rows": 1},
            ],
        )


class _StubRun:
    def __init__(self):
        self.logged = {}
        self.summary = {}
        self.finished = None

    def log(self, payload):
        self.logged.update(payload)

    def finish(self, exit_code=0):
        self.finished = exit_code


def _stub_wandb(run):
    stub = types.SimpleNamespace()
    stub.init_kwargs = None

    def init(**kwargs):
        stub.init_kwargs = kwargs
        return run

    stub.init = init
    stub.Table = lambda columns, data: ("table", tuple(columns), len(data))
    stub.Histogram = lambda values: ("hist", len(values))
    stub.plot = types.SimpleNamespace(
        line=lambda table, x, y, title=None: ("line", x, y)
    )
    return stub


class WandbLoggingStubTest(unittest.TestCase):
    def test_log_to_wandb_resumable_run_and_keys(self):
        arms = {"libero_goal": {
            "episodes": [_episode("libero_goal", 0, True)],
            "decisions": [_decision(0, 0.5, 1.0)],
        }}
        payload = build_payload("VLA-JEPA-memv1-live-step_34729", arms, None)
        args = types.SimpleNamespace(
            ckpt_name="VLA-JEPA-memv1-live-step_34729",
            project="vla-jepa", entity="crlc112358",
        )
        run = _StubRun()
        stub = _stub_wandb(run)
        log_to_wandb(args, "VLA-JEPA-memv1-live-step_34729", "step_34729", payload, wandb=stub)
        self.assertEqual(stub.init_kwargs["id"], "VLA-JEPA-memv1-live-step_34729")
        self.assertEqual(stub.init_kwargs["resume"], "allow")
        self.assertEqual(stub.init_kwargs["group"], "step_34729")
        self.assertEqual(stub.init_kwargs["job_type"], "eval")
        self.assertIn("eval/episodes", run.logged)
        self.assertIn("decisions/injection_ratio", run.logged)
        self.assertIn("decisions/working_norm_by_index", run.logged)
        self.assertEqual(run.summary["success_rate/pooled"], 1.0)
        self.assertEqual(run.finished, 0)


if __name__ == "__main__":
    unittest.main()
