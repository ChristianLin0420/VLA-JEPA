import dataclasses
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

import starVLA.training.train_vlajepa_cotrain as cotrain
from starVLA.model.modules.memory import (
    RecurrentMemory,
    ResidualMemoryFusion,
    SparseKeyMemoryFusion,
)


def _unit(rows):
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def _bare_trainer(rec_weight=0.0, nce_weight=0.0, segment_length=2):
    trainer = object.__new__(cotrain.VLAMTrainer)
    trainer.rec_loss_weight = rec_weight
    trainer.nce_loss_weight = nce_weight
    trainer._nce_queue = cotrain.EpisodicNCEQueue() if nce_weight else None
    trainer.segment_length = segment_length
    return trainer


def _fake_segment(dataset_id, episode_id, lang):
    return {
        "dataset_id": dataset_id,
        "episode_id": episode_id,
        "steps": [None, {"lang": lang}],
    }


class _StubSegmentModel(nn.Module):
    """Minimal read -> fuse -> write unroll exposing the meter contract.

    ``schema2=True`` mirrors the memv2 topology: keyed memory plus a
    SparseKeyMemoryFusion that consumes the pre-write MemoryState directly
    (the read output feeds nothing downstream), with an opened content gate.
    Segments may carry a per-step boolean ``mask_plan``; ``rec_loss`` is then
    computed from the fused tokens of the masked steps only, mirroring the
    real masked-step threading.  Without a mask_plan every step contributes
    (the legacy stub behaviour).  ``read_state_overrides`` maps step index to
    a full replacement MemoryState for explicit foreign counterfactuals.
    """

    def __init__(self, steps=3, schema2=False):
        super().__init__()
        torch.manual_seed(0)
        self.schema2 = schema2
        if schema2:
            self.memory_module = RecurrentMemory(
                source_dim=6, memory_dim=4, num_slots=2, num_heads=2,
                update_gate_init=0.3, use_keys=True, key_dim=4,
            )
            self.policy_memory_fusion = SparseKeyMemoryFusion(
                consumer_dim=6, memory_dim=4, key_dim=4, num_slots=2,
                content_gate_init=0.5,
            )
        else:
            self.memory_module = RecurrentMemory(
                source_dim=6, memory_dim=4, num_slots=2, num_heads=2, update_gate_init=0.3
            )
            self.policy_memory_fusion = ResidualMemoryFusion(
                consumer_dim=6, memory_dim=4, bottleneck_dim=4, num_heads=2, gate_init=0.8
            )
        self.steps = steps
        self.force_bypass = False
        self.read_state_overrides = {}
        self.read_states = []

    def forward(self, segments):
        tokens = torch.stack([segment["tokens"] for segment in segments])
        mask_plan = segments[0].get("mask_plan")
        if mask_plan is None:
            mask_plan = [True] * self.steps
        state = self.memory_module.init_state(len(segments), torch.device("cpu"))
        fused_steps = []
        rec_terms = []
        self.read_states = []
        for t in range(self.steps):
            source = tokens[:, t]
            self.read_states.append(state.detach())
            read_state = self.read_state_overrides.get(t)
            if read_state is None:
                read_state = state
            read = self.memory_module.read(source, read_state)
            fused = self.policy_memory_fusion(
                source,
                read_state if self.schema2 else read.tokens,
                bypass=self.force_bypass,
            )
            fused_steps.append(fused)
            if mask_plan[t]:
                rec_terms.append(fused.square().mean())
            state = self.memory_module.write(source, state)
        losses = {"action_loss": torch.stack(fused_steps).abs().mean()}
        if rec_terms:
            losses["rec_loss"] = torch.stack(rec_terms).mean()
        return losses


def _stub_batch(batch_size=2, steps=3, tokens_per_step=2, mask_plan=None):
    generator = torch.Generator().manual_seed(11)
    return [
        {
            "tokens": torch.randn(steps, tokens_per_step, 6, generator=generator),
            **({"mask_plan": list(mask_plan)} if mask_plan is not None else {}),
        }
        for _ in range(batch_size)
    ]


class EpisodicNCEQueueTest(unittest.TestCase):
    def test_loss_matches_manual_softmax_and_own_episode_is_masked(self):
        queue = cotrain.EpisodicNCEQueue(size=8, temperature=0.5)
        task = torch.tensor([1], dtype=torch.int64)

        p0 = _unit([[1.0, 0.0, 0.0, 0.0]])
        loss0, diag0 = queue.loss(p0.clone(), p0, task, torch.tensor([100]))
        # The queue holds only the anchor's own episode: no valid negatives.
        self.assertAlmostEqual(float(loss0), 0.0, places=6)
        self.assertEqual(diag0, {})

        a1 = _unit([[0.8, 0.6, 0.0, 0.0]])
        p1 = _unit([[0.0, 1.0, 0.0, 0.0]])
        loss1, diag1 = queue.loss(a1, p1, task, torch.tensor([200]))
        # Negatives = {p0}; own p1 (episode 200) is masked out.
        manual = F.cross_entropy(
            torch.tensor([[0.6 / 0.5, 0.8 / 0.5]]), torch.tensor([0])
        )
        self.assertAlmostEqual(float(loss1), float(manual), places=5)
        self.assertEqual(diag1["nce/acc"], 0.0)
        self.assertEqual(diag1["nce/same_task_acc"], 0.0)

        # A same-episode near-duplicate of the anchor must not enter the
        # denominator: anchor == p0 with episode 100 sees only p1 and the
        # freshly enqueued p2 as negatives.
        p2 = _unit([[0.0, 0.0, 1.0, 0.0]])
        loss2, diag2 = queue.loss(p0.clone(), p2, task, torch.tensor([100]))
        manual2 = F.cross_entropy(
            torch.tensor([[0.0 / 0.5, 0.0 / 0.5]]), torch.tensor([0])
        )
        self.assertAlmostEqual(float(loss2), float(manual2), places=5)
        self.assertEqual(diag2["nce/acc"], 0.0)

    def test_same_task_accuracy_splits_from_overall_accuracy(self):
        queue = cotrain.EpisodicNCEQueue(size=8, temperature=0.07)
        # Seed one same-task and one cross-task entry.
        queue._enqueue(
            _unit([[0.5, 0.866, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([7, 8]),
            torch.tensor([1, 2]),
        )
        anchor = _unit([[1.0, 0.0, 0.0, 0.0]])
        positive = _unit([[0.9, 0.436, 0.0, 0.0]])
        loss, diag = queue.loss(anchor, positive, torch.tensor([7]), torch.tensor([3]))
        self.assertGreater(float(loss), 0.0)
        # Positive similarity beats the same-task negative but loses to the
        # cross-task duplicate of the anchor.
        self.assertEqual(diag["nce/same_task_acc"], 1.0)
        self.assertEqual(diag["nce/acc"], 0.0)

    def test_fifo_queue_is_bounded_and_keeps_newest(self):
        queue = cotrain.EpisodicNCEQueue(size=4, temperature=0.07)
        chunks = []
        for call in range(3):
            embeddings = _unit(torch.randn(2, 4, generator=torch.Generator().manual_seed(call)).tolist())
            chunks.append(embeddings)
            queue.loss(
                embeddings.clone(),
                embeddings,
                torch.tensor([call, call]),
                torch.tensor([10 + call, 10 + call]),
            )
        self.assertEqual(queue.embeddings.shape, (4, 4))
        torch.testing.assert_close(queue.embeddings, torch.cat(chunks[1:]))
        self.assertEqual(queue.episodes.tolist(), [11, 11, 12, 12])

    def test_negatives_carry_no_grad_while_anchor_and_positive_do(self):
        queue = cotrain.EpisodicNCEQueue(size=8, temperature=0.07)
        queue._enqueue(_unit([[0.0, 1.0, 0.0, 0.0]]), torch.tensor([1]), torch.tensor([50]))
        anchor = _unit([[1.0, 0.0, 0.0, 0.0]]).requires_grad_()
        positive = _unit([[0.9, 0.4, 0.0, 0.1]]).requires_grad_()
        loss, _ = queue.loss(anchor, positive, torch.tensor([1]), torch.tensor([60]))
        loss.backward()
        self.assertIsNotNone(anchor.grad)
        self.assertIsNotNone(positive.grad)
        self.assertFalse(queue.embeddings.requires_grad)


class LossAggregationTest(unittest.TestCase):
    def test_default_weights_reproduce_memv1_sum(self):
        trainer = _bare_trainer()
        losses = {
            "action_loss": torch.tensor(1.3, requires_grad=True),
            "wm_loss": torch.tensor(0.7, requires_grad=True),
        }
        total, nce_loss, metrics = trainer._aggregate_vla_losses(losses, None, None, [])
        self.assertTrue(torch.equal(total, sum(losses.values())))
        self.assertIsNone(nce_loss)
        self.assertEqual(metrics, {})

        # Schema-2 extras with default-zero weights change nothing.
        losses["rec_loss"] = torch.tensor(2.9, requires_grad=True)
        anchors = _unit([[1.0, 0.0], [0.0, 1.0]])
        total_two, nce_loss, metrics = trainer._aggregate_vla_losses(
            losses, anchors, anchors, [_fake_segment("d", 1, "t")]
        )
        self.assertTrue(torch.equal(total_two, losses["action_loss"] + losses["wm_loss"]))
        self.assertIsNone(nce_loss)
        self.assertEqual(metrics, {})

    def test_configured_weights_scale_rec_and_nce_terms(self):
        trainer = _bare_trainer(rec_weight=0.5, nce_weight=0.2)
        losses = {
            "action_loss": torch.tensor(1.0),
            "wm_loss": torch.tensor(0.5),
            "rec_loss": torch.tensor(2.0),
        }
        anchors = _unit([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        positives = _unit([[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]])
        segments = [_fake_segment("d", 1, "task-a"), _fake_segment("d", 2, "task-a")]

        expected_queue = cotrain.EpisodicNCEQueue()
        task_ids, episode_ids = cotrain.VLAMTrainer._segment_metadata_ids(
            segments, 2, anchors.device
        )
        expected_nce, _ = expected_queue.loss(anchors, positives, task_ids, episode_ids)

        total, nce_loss, _ = trainer._aggregate_vla_losses(
            dict(losses), anchors, positives, segments
        )
        self.assertAlmostEqual(float(nce_loss), float(expected_nce), places=6)
        expected_total = 1.0 + 0.5 + 0.5 * 2.0 + 0.2 * float(expected_nce)
        self.assertAlmostEqual(float(total), expected_total, places=5)

    def test_metadata_ids_tile_step_major_and_reject_misalignment(self):
        segments = [_fake_segment("d", 1, "task-a"), _fake_segment("d", 2, "task-b")]
        task_ids, episode_ids = cotrain.VLAMTrainer._segment_metadata_ids(
            segments, 6, torch.device("cpu")
        )
        self.assertEqual(task_ids.shape, (6,))
        self.assertEqual(task_ids[0::2].unique().numel(), 1)
        self.assertEqual(task_ids[1::2].unique().numel(), 1)
        self.assertNotEqual(int(task_ids[0]), int(task_ids[1]))
        self.assertEqual(episode_ids[0::2].unique().numel(), 1)
        self.assertNotEqual(int(episode_ids[0]), int(episode_ids[1]))
        with self.assertRaisesRegex(ValueError, "cannot align"):
            cotrain.VLAMTrainer._segment_metadata_ids(segments, 5, torch.device("cpu"))
        same = [_fake_segment("d", 1, "task-a"), _fake_segment("d", 1, "task-a")]
        task_same, episode_same = cotrain.VLAMTrainer._segment_metadata_ids(
            same, 2, torch.device("cpu")
        )
        self.assertEqual(int(task_same[0]), int(task_same[1]))
        self.assertEqual(int(episode_same[0]), int(episode_same[1]))


class Memv2MetersTest(unittest.TestCase):
    def test_bypass_and_foreign_deltas_match_explicit_counterfactuals(self):
        model = _StubSegmentModel()
        trainer = _bare_trainer(segment_length=2)
        trainer._unwrapped = model
        batch = _stub_batch(batch_size=2)

        with torch.no_grad():
            live = model(batch)
            recorded = list(model.read_states)

            model.force_bypass = True
            expected_bypass = model(batch)
            model.force_bypass = False

            model.read_state_overrides = {
                t: dataclasses.replace(
                    recorded[t], working=recorded[t].working.roll(1, dims=0)
                )
                for t in (1, 2)
            }
            expected_foreign = model(batch)
            model.read_state_overrides = {}

        meters = trainer._memv2_meters(batch, live)
        self.assertEqual(
            set(meters),
            {
                "meters/delta_bypass_rec",
                "meters/delta_bypass_act",
                "meters/delta_foreign_rec",
                "meters/delta_foreign_act",
            },
        )
        for mode, expected in (("bypass", expected_bypass), ("foreign", expected_foreign)):
            for key, name in (("rec_loss", "rec"), ("action_loss", "act")):
                self.assertAlmostEqual(
                    meters[f"meters/delta_{mode}_{name}"],
                    float(expected[key]) - float(live[key]),
                    places=6,
                )
        self.assertNotEqual(meters["meters/delta_bypass_rec"], 0.0)
        self.assertNotEqual(meters["meters/delta_foreign_rec"], 0.0)

        # Wrappers must not leak past the meter computation.
        self.assertNotIn("read", vars(model.memory_module))
        self.assertNotIn("forward", vars(model.policy_memory_fusion))

    def test_identical_batch_rows_make_the_foreign_swap_a_no_op(self):
        model = _StubSegmentModel()
        trainer = _bare_trainer(segment_length=2)
        trainer._unwrapped = model
        row = _stub_batch(batch_size=1)[0]
        batch = [row, {"tokens": row["tokens"].clone()}]
        with torch.no_grad():
            live = model(batch)
        meters = trainer._memv2_meters(batch, live)
        self.assertAlmostEqual(meters["meters/delta_foreign_rec"], 0.0, places=6)
        self.assertAlmostEqual(meters["meters/delta_foreign_act"], 0.0, places=6)

    def test_single_row_world_size_one_skips_the_foreign_meter(self):
        model = _StubSegmentModel()
        trainer = _bare_trainer(segment_length=2)
        trainer._unwrapped = model
        batch = _stub_batch(batch_size=1)
        with torch.no_grad():
            live = model(batch)
        meters = trainer._memv2_meters(batch, live)
        self.assertIn("meters/delta_bypass_rec", meters)
        self.assertIn("meters/delta_bypass_act", meters)
        self.assertNotIn("meters/delta_foreign_rec", meters)

    def test_meters_require_memory_modules(self):
        trainer = _bare_trainer(segment_length=2)
        trainer._unwrapped = nn.Linear(2, 2)
        self.assertEqual(trainer._memv2_meters([], {}), {})

    def test_schema2_masked_step_fires_bypass_and_foreign_rec_deltas(self):
        """memv2.1 regression: rec deltas must be live on masked segments.

        The stage-1 bug: the foreign swap was applied only inside memory.read
        while the schema-2 fusion consumes the state directly, so
        delta_foreign_rec was identically zero.
        """
        model = _StubSegmentModel(schema2=True)
        trainer = _bare_trainer(segment_length=2)
        trainer._unwrapped = model
        batch = _stub_batch(batch_size=2, mask_plan=[False, False, True])

        with torch.no_grad():
            live = model(batch)
            recorded = list(model.read_states)

            model.force_bypass = True
            expected_bypass = model(batch)
            model.force_bypass = False

            model.read_state_overrides = {
                t: dataclasses.replace(
                    recorded[t],
                    working=recorded[t].working.roll(1, dims=0),
                    keys=recorded[t].keys.roll(1, dims=0),
                )
                for t in (1, 2)
            }
            expected_foreign = model(batch)
            model.read_state_overrides = {}

        meters = trainer._memv2_meters(batch, live)
        self.assertIn("rec_loss", live)
        for mode, expected in (("bypass", expected_bypass), ("foreign", expected_foreign)):
            delta = meters[f"meters/delta_{mode}_rec"]
            self.assertNotEqual(delta, 0.0)
            self.assertAlmostEqual(
                delta, float(expected["rec_loss"]) - float(live["rec_loss"]), places=6
            )
        self.assertNotIn("read", vars(model.memory_module))
        self.assertNotIn("forward", vars(model.policy_memory_fusion))

    def test_schema2_unmasked_batch_emits_action_deltas_only(self):
        model = _StubSegmentModel(schema2=True)
        trainer = _bare_trainer(segment_length=2)
        trainer._unwrapped = model
        batch = _stub_batch(batch_size=2, mask_plan=[False, False, False])
        with torch.no_grad():
            live = model(batch)
        self.assertNotIn("rec_loss", live)
        meters = trainer._memv2_meters(batch, live)
        self.assertEqual(
            set(meters),
            {"meters/delta_bypass_act", "meters/delta_foreign_act"},
        )


class MaskScheduleConfigTest(unittest.TestCase):
    def _trainer(self, mask_rate, dataset, run_len=None):
        trainer = object.__new__(cotrain.VLAMTrainer)
        trainer.memory_mask_rate = mask_rate
        trainer.mask_ramp_steps = 100
        trainer.total_batch_size = 8
        vla_data = {"memory_mask_max_per_segment": 2}
        if run_len is not None:
            vla_data["memory_mask_run_len"] = run_len
        trainer.config = OmegaConf.create({"datasets": {"vla_data": vla_data}})
        trainer.vla_train_dataloader = SimpleNamespace(dataset=dataset)
        return trainer

    def test_static_attributes_reach_the_segment_dataset(self):
        dataset = SimpleNamespace(sample_segment=lambda index: None)
        self._trainer(0.25, dataset)._configure_mask_schedule()
        self.assertEqual(dataset.memory_mask_rate, 0.25)
        self.assertEqual(dataset.memory_mask_max_per_segment, 2)
        self.assertEqual(dataset.memory_mask_run_len, 1)
        self.assertEqual(dataset.memory_mask_ramp_samples, 800)

    def test_run_len_reaches_the_segment_dataset(self):
        dataset = SimpleNamespace(sample_segment=lambda index: None)
        self._trainer(0.25, dataset, run_len=2)._configure_mask_schedule()
        self.assertEqual(dataset.memory_mask_run_len, 2)

    def test_run_len_beyond_cap_fails_loudly(self):
        dataset = SimpleNamespace(sample_segment=lambda index: None)
        with self.assertRaisesRegex(ValueError, "memory_mask_run_len"):
            self._trainer(0.25, dataset, run_len=3)._configure_mask_schedule()

    def test_zero_rate_is_a_memv1_no_op(self):
        dataset = SimpleNamespace()
        self._trainer(0.0, dataset)._configure_mask_schedule()
        self.assertFalse(hasattr(dataset, "memory_mask_rate"))

    def test_masking_without_segment_dataset_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "contiguous-segment"):
            self._trainer(0.25, SimpleNamespace())._configure_mask_schedule()


if __name__ == "__main__":
    unittest.main()
