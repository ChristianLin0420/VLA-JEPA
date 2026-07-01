import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


class _ExpandedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.memory_module = nn.Linear(4, 2)


class CheckpointMigrationTest(unittest.TestCase):
    def _checkpoint(self, state_dict):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "model.pt"
        torch.save(state_dict, path)
        self.addCleanup(directory.cleanup)
        return path

    def test_only_allowlisted_missing_prefix_is_accepted(self):
        model = _ExpandedModel()
        checkpoint = self._checkpoint(
            {
                "backbone.weight": model.backbone.weight.detach().clone(),
                "backbone.bias": model.backbone.bias.detach().clone(),
            }
        )
        loaded = TrainerUtils.load_pretrained_backbones(
            model,
            checkpoint,
            allowed_missing_prefixes=("memory_module.",),
        )
        self.assertIs(loaded, model)

    def test_disallowed_missing_key_is_rejected(self):
        model = _ExpandedModel()
        checkpoint = self._checkpoint({"backbone.weight": model.backbone.weight.detach().clone()})
        with self.assertRaisesRegex(RuntimeError, "disallowed missing"):
            TrainerUtils.load_pretrained_backbones(
                model,
                checkpoint,
                allowed_missing_prefixes=("memory_module.",),
            )

    def test_unexpected_key_is_always_rejected(self):
        model = _ExpandedModel()
        state = model.state_dict()
        state["not_allowed.weight"] = torch.ones(1)
        checkpoint = self._checkpoint(state)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            TrainerUtils.load_pretrained_backbones(
                model,
                checkpoint,
                allowed_missing_prefixes=("memory_module.",),
            )

    def test_optimizer_groups_are_disjoint_and_exclude_frozen_modules(self):
        model = _ExpandedModel()
        config = OmegaConf.create(
            {
                "trainer": {
                    "freeze_modules": "backbone",
                    "learning_rate": {
                        "base": 1.0e-5,
                        "memory_module": 1.0e-4,
                    },
                }
            }
        )
        groups = build_param_lr_groups(model, config)
        grouped = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.memory_module.parameters()})


if __name__ == "__main__":
    unittest.main()
