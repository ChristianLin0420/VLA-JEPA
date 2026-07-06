import unittest

import numpy as np

from scripts.data.g0_demand_audit import audit, build_argparser, build_design, phase_bucket


def make_episodes(*, coupling: float, num_episodes: int = 24, length: int = 60,
                  action_dim: int = 7, layout_dim: int = 8, num_tasks: int = 2, seed: int = 0):
    """Synthetic corpus: action = f(task, phase) + coupling * W @ layout + noise."""

    rng = np.random.default_rng(seed)
    mixing = rng.standard_normal((layout_dim, action_dim))
    episodes = []
    for episode in range(num_episodes):
        task = episode % num_tasks
        layout = rng.standard_normal(layout_dim)
        phase = np.linspace(0.0, 1.0, length)[:, None]
        task_component = (task + 1) * np.sin(np.pi * phase + task) * np.ones((1, action_dim))
        actions = (
            task_component
            + coupling * (layout @ mixing)[None, :]
            + 0.05 * rng.standard_normal((length, action_dim))
        )
        episodes.append({"task": f"task {task}", "layout": layout, "actions": actions})
    return episodes


class PhaseBucketTest(unittest.TestCase):
    def test_bounds_and_monotonicity(self):
        self.assertEqual(phase_bucket(0, 50, 10), 0)
        self.assertEqual(phase_bucket(49, 50, 10), 9)
        self.assertEqual(phase_bucket(0, 1, 10), 0)
        buckets = [phase_bucket(t, 50, 10) for t in range(50)]
        self.assertEqual(buckets, sorted(buckets))
        self.assertEqual(set(buckets), set(range(10)))


class BuildDesignTest(unittest.TestCase):
    def test_shapes_and_leak_free_start(self):
        episodes = make_episodes(coupling=0.0, num_episodes=4, length=30)
        cells, layout, targets, groups = build_design(
            episodes, num_buckets=10, stride=7, start=7
        )
        # lattice 7, 14, 21, 28 per episode; t < chunk never becomes a target
        self.assertEqual(targets.shape, (16, 7))
        self.assertEqual(cells.shape, (16, 20))
        # layout is task-interacted: 2 tasks x 8 layout dims
        self.assertEqual(layout.shape, (16, 16))
        self.assertTrue(np.all((layout != 0).sum(axis=1) <= 8))
        self.assertEqual(set(groups.tolist()), {0, 1, 2, 3})
        self.assertTrue(np.all(cells.sum(axis=1) == 1.0))


class AuditTest(unittest.TestCase):
    def test_layout_coupling_opens_the_gap(self):
        result = audit(make_episodes(coupling=1.0))
        self.assertGreater(result["gap"], 0.2)
        self.assertGreater(result["r2_plus_layout"], result["r2_task_phase"])

    def test_uncoupled_corpus_has_null_gap(self):
        result = audit(make_episodes(coupling=0.0))
        self.assertLess(abs(result["gap"]), 0.05)
        # (task, phase) alone already explains the task-conditional trajectory
        self.assertGreater(result["r2_task_phase"], 0.5)

    def test_deterministic(self):
        episodes = make_episodes(coupling=0.5)
        self.assertEqual(audit(episodes), audit(episodes))


class CliTest(unittest.TestCase):
    def test_help_builds_without_heavy_imports(self):
        self.assertIn("G0 demand audit", build_argparser().format_help())


if __name__ == "__main__":
    unittest.main()
