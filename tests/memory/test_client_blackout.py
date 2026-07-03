import json
import unittest

import numpy as np

from examples.LIBERO.blackout import (
    EPISODE_RECORD_KEYS,
    corrupt_views,
    decision_record,
    episode_record,
    in_blackout,
    memory_params_from_env,
)


class BlackoutScheduleTest(unittest.TestCase):
    def test_window_covers_exactly_d0_to_d0_plus_D(self):
        active = [d for d in range(16) if in_blackout(d, 8, 4)]
        self.assertEqual(active, [8, 9, 10, 11])

    def test_disabled_sentinels(self):
        self.assertFalse(in_blackout(0, -1, 4))
        self.assertFalse(in_blackout(0, 0, 0))
        self.assertTrue(in_blackout(0, 0, 1))

    def test_env_step_alignment_at_chunk_seven(self):
        # decisions [2, 4) at chunk 7 => env steps [14, 28)
        blacked = [s for s in range(70) if in_blackout(s // 7, 2, 2)]
        self.assertEqual(blacked, list(range(14, 28)))


class BlackoutFillTest(unittest.TestCase):
    def setUp(self):
        self.img = np.full((4, 4, 3), 100, dtype=np.uint8)
        self.wrist = np.full((4, 4, 3), 200, dtype=np.uint8)
        self.last_img = np.full((4, 4, 3), 1, dtype=np.uint8)
        self.last_wrist = np.full((4, 4, 3), 2, dtype=np.uint8)

    def test_black_fill_zeroes_both_views(self):
        img, wrist = corrupt_views(
            self.img, self.wrist, "black", "both", self.last_img, self.last_wrist
        )
        self.assertTrue((img == 0).all())
        self.assertTrue((wrist == 0).all())
        self.assertEqual(img.dtype, np.uint8)

    def test_agentview_only_leaves_wrist_untouched(self):
        img, wrist = corrupt_views(
            self.img, self.wrist, "black", "agentview", self.last_img, self.last_wrist
        )
        self.assertTrue((img == 0).all())
        self.assertIs(wrist, self.wrist)

    def test_freeze_repeats_last_clean_frame(self):
        img, wrist = corrupt_views(
            self.img, self.wrist, "freeze", "both", self.last_img, self.last_wrist
        )
        np.testing.assert_array_equal(img, self.last_img)
        np.testing.assert_array_equal(wrist, self.last_wrist)

    def test_freeze_without_history_falls_back_to_black(self):
        img, wrist = corrupt_views(self.img, self.wrist, "freeze", "both", None, None)
        self.assertTrue((img == 0).all())
        self.assertTrue((wrist == 0).all())

    def test_freeze_returns_copies_not_aliases(self):
        img, _ = corrupt_views(
            self.img, self.wrist, "freeze", "both", self.last_img, self.last_wrist
        )
        img[...] = 7
        self.assertTrue((self.last_img == 1).all())

    def test_rejects_unknown_fill_and_views(self):
        with self.assertRaises(ValueError):
            corrupt_views(self.img, self.wrist, "noise", "both", None, None)
        with self.assertRaises(ValueError):
            corrupt_views(self.img, self.wrist, "black", "wrist", None, None)


class EpisodeRecordTest(unittest.TestCase):
    def _record(self, **overrides):
        kwargs = dict(
            suite="libero_10",
            task_id=3,
            task_description="put the bowl on the plate",
            episode_idx=np.int64(4),
            memory_mode="live",
            memory_params={"MEMORY_RESET_K": "8"},
            episode_seed=5,
            success=np.bool_(True),
            num_env_steps=np.int64(140),
            num_decisions=20,
            ckpt="ckpt/VLA-JEPA-memv1-live-step_34729.pt",
            git_sha="abc123",
        )
        kwargs.update(overrides)
        return episode_record(**kwargs)

    def test_schema_has_contract_keys_and_is_json_native(self):
        record = self._record()
        for key in EPISODE_RECORD_KEYS:
            self.assertIn(key, record)
        round_trip = json.loads(json.dumps(record))
        self.assertEqual(round_trip, record)
        self.assertIs(record["success"], True)
        self.assertEqual(record["num_env_steps"], 140)
        self.assertEqual(record["episode_idx"], 4)

    def test_extras_are_merged(self):
        blackout = {"start_decision": 8, "num_decisions": 4, "fill": "black",
                    "views": "both", "suppress_write": False}
        record = self._record(extras={"blackout": blackout})
        self.assertEqual(record["blackout"], blackout)

    def test_memory_params_from_env_filters_known_keys(self):
        env = {"MEMORY_RESET_K": "8", "MEMORY_MODE": "live", "PATH": "/usr/bin"}
        self.assertEqual(memory_params_from_env(env), {"MEMORY_RESET_K": "8"})

    def test_decision_record_merges_memory_extras(self):
        extras = {"mode": "live", "decision_index": 5, "injection_ratio": 0.02}
        record = decision_record(
            episode_idx=3, d=5, memory_extras=extras, blackout_active=True
        )
        self.assertEqual(record["episode_idx"], 3)
        self.assertEqual(record["d"], 5)
        self.assertEqual(record["injection_ratio"], 0.02)
        self.assertIs(record["blackout_active"], True)

    def test_decision_record_without_memory_extras(self):
        record = decision_record(episode_idx=0, d=0)
        self.assertEqual(record, {"episode_idx": 0, "d": 0, "blackout_active": False})


if __name__ == "__main__":
    unittest.main()
