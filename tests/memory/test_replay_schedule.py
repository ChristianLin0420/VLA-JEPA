import math
import unittest

import numpy as np
import torch

from scripts.offline_eval.replay_engine import (
    RECORD_COLUMNS,
    PlanRow,
    SmokeFusion,
    SmokeHead,
    SmokeMemory,
    build_plan,
    build_state_dump,
    burn_in_indices,
    make_smoke_cache,
    parse_conditions,
    replay_episode,
    stable_seed,
    teacher_forced_states,
)


def _expected_working(memory, cache, indices):
    """Reference stub-write chain: order-sensitive affine encode of the writes."""

    working = torch.ones(1, memory.num_slots, memory.memory_dim)
    for index in indices:
        summary = (
            cache["action_tokens"][index : index + 1]
            .to(torch.float32)
            .mean(dim=1)[:, : memory.memory_dim]
        )
        working = 0.5 * working + summary[:, None, :]
    return working


def _run(plan, num_decisions=5, donor_decisions=3, episode_seed=123, noise_draws=4):
    memory, fusion, head = SmokeMemory(), SmokeFusion(), SmokeHead()
    cache = make_smoke_cache(0, num_decisions)
    donor_cache = make_smoke_cache(1, donor_decisions)
    states, _ = teacher_forced_states(memory, cache["action_tokens"])
    donor_states, _ = teacher_forced_states(memory, donor_cache["action_tokens"])
    records = replay_episode(
        memory, fusion, head, cache, plan, states,
        episode_seed=episode_seed,
        noise_draws=noise_draws,
        donor_states=donor_states,
    )
    return records, memory, fusion, head, cache, donor_cache, donor_states


class BurnInIndicesTest(unittest.TestCase):
    def test_window_and_orders(self):
        self.assertEqual(burn_in_indices(6, 4, "forward"), [2, 3, 4, 5])
        self.assertEqual(burn_in_indices(6, 4, "reversed"), [5, 4, 3, 2])
        self.assertEqual(burn_in_indices(2, 4, "forward"), [0, 1])
        self.assertEqual(burn_in_indices(3, 0, "forward"), [])

    def test_shuffled_is_a_deterministic_permutation(self):
        first = burn_in_indices(8, 6, "shuffled", np.random.default_rng(3))
        second = burn_in_indices(8, 6, "shuffled", np.random.default_rng(3))
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(2, 8)))
        with self.assertRaisesRegex(ValueError, "rng"):
            burn_in_indices(8, 6, "shuffled")
        with self.assertRaisesRegex(ValueError, "order"):
            burn_in_indices(8, 6, "backwards")


class PlanParsingTest(unittest.TestCase):
    def test_conditions_and_dedup(self):
        self.assertEqual(
            parse_conditions("live,bypass,frozen_after=8"),
            [PlanRow("live"), PlanRow("bypass"), PlanRow("frozen_after", J=8)],
        )
        with self.assertRaisesRegex(ValueError, "unknown condition"):
            parse_conditions("foreign")
        plan = build_plan("live,prior", "0,1", "forward,reversed", "1,2", "live")
        self.assertEqual(plan.count(PlanRow("live")), 1)  # lam=1 row deduped
        self.assertIn(PlanRow("burnin_reversed", J=1), plan)
        self.assertIn(PlanRow("live", lam=2.0), plan)
        with self.assertRaisesRegex(ValueError, "live/shuffled"):
            build_plan("live", "", "forward", "2", "bypass")


class ConditionScheduleTest(unittest.TestCase):
    """Which state feeds the read at decision d, per condition."""

    def test_states_fed_per_condition(self):
        plan = [
            PlanRow("live"),
            PlanRow("prior"),
            PlanRow("bypass"),
            PlanRow("frozen_after", J=2),
            PlanRow("shuffled"),
        ]
        records, memory, fusion, _, cache, _, donor_states = _run(plan, num_decisions=5)

        expected = []
        for row in plan:
            for decision in range(5):
                if row.condition == "live":
                    expected.append(_expected_working(memory, cache, range(decision)))
                elif row.condition == "prior":
                    expected.append(_expected_working(memory, cache, ()))
                elif row.condition == "frozen_after":
                    expected.append(
                        _expected_working(memory, cache, range(min(decision, row.J)))
                    )
                elif row.condition == "shuffled":
                    expected.append(
                        donor_states[min(decision, len(donor_states) - 1)].working
                    )
        self.assertEqual(len(fusion.calls), len(expected))  # bypass never fused
        for (observed, _), reference in zip(fusion.calls, expected):
            self.assertTrue(torch.equal(observed, reference))
        self.assertEqual(len(records), len(plan) * 5)

    def test_bypass_semantics_and_live_equals_prior_at_decision_zero(self):
        plan = [PlanRow("live"), PlanRow("prior"), PlanRow("bypass")]
        records, *_ = _run(plan)
        by_key = {(r["condition"], r["decision"]): r for r in records}
        self.assertEqual(by_key[("live", 0)]["mse"], by_key[("prior", 0)]["mse"])
        self.assertEqual(by_key[("live", 0)]["tf_loss"], by_key[("prior", 0)]["tf_loss"])
        for decision in range(5):
            self.assertTrue(math.isnan(by_key[("bypass", decision)]["working_norm"]))
            self.assertFalse(math.isnan(by_key[("live", decision)]["working_norm"]))


class BurnInScheduleTest(unittest.TestCase):
    def test_orders_permute_the_write_window(self):
        episode_seed = 123
        plan = [
            PlanRow("prior"),
            PlanRow("burnin_forward", J=3),
            PlanRow("burnin_reversed", J=3),
            PlanRow("burnin_shuffled", J=3),
            PlanRow("burnin_forward", J=0),
        ]
        _, memory, fusion, _, cache, _, _ = _run(plan, num_decisions=5, episode_seed=episode_seed)

        calls = [tokens for tokens, _ in fusion.calls]
        prior, forward = calls[0:5], calls[5:10]
        reverse, shuffled = calls[10:15], calls[15:20]
        zero_window = calls[20:25]
        for decision in range(5):
            window = list(range(max(0, decision - 3), decision))
            self.assertTrue(
                torch.equal(forward[decision], _expected_working(memory, cache, window))
            )
            self.assertTrue(
                torch.equal(reverse[decision], _expected_working(memory, cache, window[::-1]))
            )
            rng = np.random.default_rng(
                stable_seed("burnin-shuffle-v1", episode_seed, decision, 3)
            )
            permuted = [window[i] for i in rng.permutation(len(window))]
            self.assertTrue(
                torch.equal(shuffled[decision], _expected_working(memory, cache, permuted))
            )
            # J=0 burn-in must be exactly the prior condition.
            self.assertTrue(torch.equal(zero_window[decision], prior[decision]))
        # The orders genuinely differ once the window has >= 2 writes.
        self.assertFalse(torch.equal(forward[4], reverse[4]))


class SharedNoiseTest(unittest.TestCase):
    def test_noise_draws_are_bit_identical_across_conditions(self):
        plan = [PlanRow("live"), PlanRow("prior"), PlanRow("bypass"), PlanRow("live", lam=2.0)]
        _, _, _, head, *_ = _run(plan, num_decisions=5, noise_draws=4)

        self.assertEqual(len(head.predict_calls), len(plan) * 5)
        self.assertEqual(len(head.forward_calls), len(plan) * 5)
        for decision in range(5):
            _, base_noise = head.predict_calls[decision]
            base_t, base_tf_noise = head.forward_calls[decision]
            self.assertEqual(tuple(base_noise.shape), (4, 7, 7))
            for row_index in range(1, len(plan)):
                _, other_noise = head.predict_calls[row_index * 5 + decision]
                other_t, other_tf_noise = head.forward_calls[row_index * 5 + decision]
                self.assertTrue(torch.equal(base_noise, other_noise))
                self.assertTrue(torch.equal(base_t, other_t))
                self.assertTrue(torch.equal(base_tf_noise, other_tf_noise))


class LambdaSweepTest(unittest.TestCase):
    def test_residual_scale_is_set_per_row_and_restored(self):
        plan = [PlanRow("live"), PlanRow("live", lam=0.0), PlanRow("live", lam=2.0)]
        _, _, fusion, _, _, _, _ = _run(plan, num_decisions=3)
        scales = [scale for _, scale in fusion.calls]
        self.assertEqual(scales, [1.0] * 3 + [0.0] * 3 + [2.0] * 3)
        self.assertEqual(fusion.residual_scale, 1.0)

    def test_missing_residual_scale_hook_fails_loudly(self):
        memory, fusion, head = SmokeMemory(), SmokeFusion(), SmokeHead()
        del fusion.residual_scale
        cache = make_smoke_cache(0, 2)
        states, _ = teacher_forced_states(memory, cache["action_tokens"])
        with self.assertRaisesRegex(RuntimeError, "residual_scale"):
            replay_episode(
                memory, fusion, head, cache, [PlanRow("live", lam=2.0)], states,
                episode_seed=1, noise_draws=2,
            )


class RecordContractTest(unittest.TestCase):
    def test_schema_and_determinism(self):
        plan = build_plan("live,bypass,prior,shuffled,frozen_after=2", "0,2", "forward", "2", "live")
        first, *_ = _run(plan, episode_seed=99)
        second, *_ = _run(plan, episode_seed=99)

        def _nan_safe(record):
            return {
                key: "nan" if isinstance(value, float) and math.isnan(value) else value
                for key, value in record.items()
            }

        self.assertEqual([_nan_safe(r) for r in first], [_nan_safe(r) for r in second])
        for record in first:
            self.assertEqual(tuple(record.keys()), RECORD_COLUMNS)
        base = [r for r in first if r["condition"] == "live" and r["lam"] == 1.0]
        self.assertTrue(all(r["J"] == -1 for r in base))
        frozen = [r for r in first if r["condition"] == "frozen_after"]
        self.assertTrue(all(r["J"] == 2 for r in frozen))

    def test_shuffled_skipped_without_donor(self):
        memory, fusion, head = SmokeMemory(), SmokeFusion(), SmokeHead()
        cache = make_smoke_cache(0, 3)
        states, _ = teacher_forced_states(memory, cache["action_tokens"])
        records = replay_episode(
            memory, fusion, head, cache,
            [PlanRow("live"), PlanRow("shuffled")], states,
            episode_seed=1, noise_draws=2, donor_states=None,
        )
        self.assertEqual({r["condition"] for r in records}, {"live"})


class StateDumpCaptureTest(unittest.TestCase):
    def test_capture_collects_h3_diagnostics(self):
        memory = SmokeMemory()
        cache = make_smoke_cache(0, 4)
        states, write_diags = teacher_forced_states(
            memory, cache["action_tokens"], capture=True
        )
        self.assertEqual(len(write_diags), 4)
        dump = build_state_dump(memory, cache, states, write_diags)
        self.assertEqual(dump["states"].shape, (4, memory.num_slots, memory.memory_dim))
        self.assertEqual(dump["token_mean"].shape, (4, cache["action_tokens"].shape[-1]))
        self.assertEqual(dump["write_update_gate_mean"].shape, (4,))
        self.assertEqual(dump["read_attention"].shape[0], 4)
        self.assertEqual(dump["progress"][0], 0.0)
        self.assertEqual(dump["progress"][-1], 1.0)

    def test_missing_h3_hook_fails_loudly(self):
        class _NoDiagnosticsMemory(SmokeMemory):
            """Pre-H3 module: write() never populates last_write_diagnostics."""

            def write(self, source_tokens, state, update_mask=None):
                capture, self.capture_diagnostics = self.capture_diagnostics, False
                try:
                    return super().write(source_tokens, state, update_mask)
                finally:
                    self.capture_diagnostics = capture

        memory = _NoDiagnosticsMemory()
        cache = make_smoke_cache(0, 2)
        with self.assertRaisesRegex(RuntimeError, "last_write_diagnostics"):
            teacher_forced_states(memory, cache["action_tokens"], capture=True)


if __name__ == "__main__":
    unittest.main()
