from types import SimpleNamespace
import unittest

import numpy as np
import torch

import utils
from models import hgnn_module, st_block


def configure(**overrides):
    values = {
        "device": "cpu",
        "eps": 1e-6,
        "khop": 2,
        "dynamic_top_k": 1,
        "random_walk_steps": 2,
        "laplacian_k": 2,
        "d_model": 8,
        "d_token": 4,
        "embed_dim": 3,
        "num_heads": 1,
        "dropout": 0.0,
        "num_layers": 1,
        "dim_forward": 16,
        "pred_k": 2,
        "use_line_hyperedge": True,
        "use_khop_hyperedge": True,
        "use_dynamic_hyperedge": True,
        "use_flow_aware": True,
        "use_spatial_embedding": True,
    }
    values.update(overrides)
    cfg = SimpleNamespace(**values)
    utils.cfg = cfg
    hgnn_module.cfg = cfg
    st_block.cfg = cfg
    return cfg


class FSTHGTContractTest(unittest.TestCase):
    def setUp(self):
        configure()

    def test_static_hypergraph_line_and_khop_edges(self):
        line_matrix = np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 1],
            ]
        )
        adjacency = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ]
        )

        static_h = utils.construct_static_hypergraph(line_matrix, adjacency)

        self.assertEqual(static_h.shape, (4, 3))
        self.assertTrue(torch.equal(static_h[:, :2], torch.tensor(line_matrix).float()))
        self.assertEqual(static_h[1, 2].item(), 1.0)

    def test_dynamic_hypergraph_excludes_self_from_topk_then_adds_center(self):
        attention = torch.tensor(
            [
                [
                    [10.0, 2.0, 1.0],
                    [3.0, 10.0, 2.0],
                    [1.0, 4.0, 10.0],
                ]
            ]
        )

        dynamic_h = utils.construct_dynamic_hypergraph(attention)

        self.assertEqual(dynamic_h.shape, (1, 3, 3))
        self.assertTrue(torch.equal(dynamic_h[0].diag(), torch.ones(3)))
        self.assertEqual(dynamic_h[0, 1, 0].item(), 1.0)
        self.assertEqual(dynamic_h[0, 0, 1].item(), 1.0)
        self.assertEqual(dynamic_h[0, 1, 2].item(), 1.0)

    def test_spatial_encodings_follow_paper_shapes(self):
        adjacency = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        )

        rw = utils.random_walk_encoding(adjacency, 2)
        lap = utils.laplacian_encoding(adjacency, 2)

        self.assertEqual(rw.shape, (3, 2))
        self.assertTrue(torch.allclose(rw[:, 0], torch.ones(3).float()))
        self.assertEqual(lap.shape, (3, 2))

    def test_fsthgt_forward_predicts_future_steps(self):
        configure()
        model = st_block.FSTHGT()
        src = torch.randn(2, 4, 3, 2)
        static_h = torch.ones(2, 4, 3, 2)
        embedding = torch.randn(2, 4, 3, 4)
        popu_flow = torch.rand(2, 4, 3) + 1.0
        target = torch.randn(2, 2, 3, 2)

        output, truth = model(
            src=src,
            src_static_h=static_h,
            src_embedding=embedding,
            src_popu_flow=popu_flow,
            tgt_y=target,
        )

        self.assertEqual(output.shape, target.shape)
        self.assertIs(truth, target)

    def test_temporal_attention_uses_spatial_only_for_qk_not_v(self):
        cfg = configure()
        layer = st_block.FSTHGTLayer()

        self.assertEqual(layer.temporal_attn.embed_dim, cfg.d_model)
        self.assertEqual(layer.temporal_attn.kdim, cfg.d_model + cfg.embed_dim)
        self.assertEqual(layer.temporal_attn.vdim, cfg.d_model)
        self.assertEqual(layer.temporal_query_proj.in_features, cfg.d_model + cfg.embed_dim)
        self.assertEqual(layer.temporal_query_proj.out_features, cfg.d_model)
        self.assertEqual(layer.ffn[0].in_features, cfg.d_model)

    def test_dynamic_hypergraph_constructed_once_per_forward(self):
        configure(num_layers=3)
        model = st_block.FSTHGT()

        counter = {"count": 0}
        original_dynamic_hypergraph = model._dynamic_hypergraph

        def wrapped_dynamic_hypergraph(x):
            counter["count"] += 1
            return original_dynamic_hypergraph(x)

        model._dynamic_hypergraph = wrapped_dynamic_hypergraph

        src = torch.randn(2, 4, 3, 2)
        static_h = torch.ones(2, 4, 3, 2)
        embedding = torch.randn(2, 4, 3, 4)
        popu_flow = torch.rand(2, 4, 3) + 1.0
        target = torch.randn(2, 2, 3, 2)

        model(
            src=src,
            src_static_h=static_h,
            src_embedding=embedding,
            src_popu_flow=popu_flow,
            tgt_y=target,
        )

        self.assertEqual(counter["count"], 1)


if __name__ == "__main__":
    unittest.main()
