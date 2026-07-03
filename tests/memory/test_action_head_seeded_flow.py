import unittest

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


def _tiny_head_config():
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 16,
                    "add_pos_embed": True,
                    "max_seq_len": 32,
                    "action_dim": 4,
                    "future_action_window_size": 2,
                    "num_inference_timesteps": 2,
                    "state_dim": None,
                    "num_target_vision_tokens": 3,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "diffusion_model_cfg": {
                        "num_layers": 1,
                        "output_dim": 16,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": False,
                        "norm_type": "ada_norm",
                        "positional_embeddings": None,
                        "cross_attention_dim": 16,
                    },
                }
            }
        }
    )


class SeededFlowLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.head = FlowmatchingActionHead(_tiny_head_config())
        self.head.eval()
        self.vl_embs = torch.randn(2, 6, 16)
        self.actions = torch.randn(2, 3, 4)

    def test_same_t_and_noise_reproduce_the_loss(self):
        t = torch.rand(2)
        noise = torch.randn(2, 3, 4)
        with torch.no_grad():
            first = self.head(self.vl_embs, self.actions, t=t, noise=noise)
            second = self.head(self.vl_embs, self.actions, t=t, noise=noise)
        self.assertTrue(torch.equal(first, second))

    def test_different_noise_changes_the_loss(self):
        t = torch.rand(2)
        with torch.no_grad():
            first = self.head(self.vl_embs, self.actions, t=t, noise=torch.randn(2, 3, 4))
            second = self.head(self.vl_embs, self.actions, t=t, noise=torch.randn(2, 3, 4))
        self.assertFalse(torch.equal(first, second))

    def test_default_sampling_path_still_runs(self):
        with torch.no_grad():
            loss = self.head(self.vl_embs, self.actions)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_shape_validation_is_explicit(self):
        with self.assertRaises(ValueError):
            self.head(self.vl_embs, self.actions, noise=torch.randn(2, 3, 5))
        with self.assertRaises(ValueError):
            self.head(self.vl_embs, self.actions, t=torch.rand(3))


if __name__ == "__main__":
    unittest.main()
