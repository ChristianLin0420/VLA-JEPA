import unittest

import torch

from starVLA.model.framework.VLA_JEPA import VLA_JEPA


class QwenTokenSelectionTest(unittest.TestCase):
    def test_selection_preserves_batch_rows(self):
        hidden = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
        input_ids = torch.tensor([[1, 8, 2, 8, 3, 0], [8, 4, 8, 5, 6, 0]])
        selected = VLA_JEPA._select_token_rows(
            hidden, input_ids, [8], expected_count=2, label="marker"
        )
        torch.testing.assert_close(selected[0], hidden[0, [1, 3]])
        torch.testing.assert_close(selected[1], hidden[1, [0, 2]])

    def test_malformed_row_fails_before_reshape(self):
        hidden = torch.zeros(2, 5, 4)
        input_ids = torch.tensor([[8, 8, 0, 0, 0], [8, 0, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "got \[2, 1\]"):
            VLA_JEPA._select_token_rows(
                hidden, input_ids, [8], expected_count=2, label="marker"
            )


if __name__ == "__main__":
    unittest.main()
