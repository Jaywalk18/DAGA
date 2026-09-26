"""Regression checks for sharing one attention encoding per image."""

import unittest

import torch

from core.daga import DAGA


class AttentionCacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DAGA(
            feature_dim=8,
            guidance_dim=4,
            mlp_ratio=0.25,
            daga_layers=[0, 1, 2, 3],
        )
        self.calls = []
        self.hook = self.model.shared_extractor.register_forward_hook(
            lambda *_: self.calls.append(1)
        )

    def tearDown(self):
        self.hook.remove()

    def test_one_extraction_per_map_and_fresh_for_next_image(self):
        self.model.eval()
        first_map = torch.randn(2, 4, 4)
        next_map = torch.randn(2, 4, 4)
        with torch.no_grad():
            for layer in range(4):
                self.model.encode_attention(first_map, layer)
            first_features = self.model._cached_shared_features.clone()
            self.assertEqual(len(self.calls), 1)

            self.model.encode_attention(next_map, 0)
            next_features = self.model._cached_shared_features.clone()
            self.assertEqual(len(self.calls), 2)
            self.assertFalse(torch.allclose(first_features, next_features))

            next_map.add_(1)
            self.model.encode_attention(next_map, 1)
            self.assertEqual(len(self.calls), 3)

    def test_training_shares_graph_across_layer_heads(self):
        self.model.train()
        attention_map = torch.randn(2, 4, 4)
        guidance = [
            self.model.encode_attention(attention_map, layer)[0]
            for layer in range(4)
        ]
        self.assertEqual(len(self.calls), 1)
        sum(value.square().sum() for value in guidance).backward()
        self.assertIsNotNone(self.model.shared_extractor.conv1.weight.grad)


if __name__ == "__main__":
    unittest.main()
