import logging
from types import SimpleNamespace

import torch
from torch import einsum, nn
from torch.nn import Parameter

import config
from utils import calculate_edge_population

logging.basicConfig(level=logging.INFO)
cfg = SimpleNamespace(**vars(config))


class FAHGConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, bias=True) -> None:
        super().__init__()
        self.drop_out = dropout
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter("bias", None)

        self.bn = nn.LayerNorm(out_ch)
        self.gelu = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def _calculate_flow_terms(self, popu_flow, hypergraph):
        node_total_flow = popu_flow.squeeze(-1).clamp_min(cfg.eps)
        if not cfg.use_flow_aware:
            node_total_flow = torch.ones_like(node_total_flow)
        edge_mean_flow = calculate_edge_population(
            node_total_flow.unsqueeze(-1), hypergraph
        ).squeeze(-1)
        return node_total_flow, edge_mean_flow

    def forward(self, x: torch.Tensor, H, popu_flow):
        assert len(x.shape) == 3, "the input of HyperConv should be N * V * C"
        if H.shape[-1] == 0:
            return self.gelu(einsum("nvc,co->nvo", x, self.theta))
        node_total_flow, edge_mean_flow = self._calculate_flow_terms(popu_flow, H)
        y = einsum("nvc,co->nvo", x, self.theta)
        edge_to_node_weight = H * edge_mean_flow.unsqueeze(1)
        Dv = torch.diag_embed(
            1.0 / edge_to_node_weight.sum(2).clamp_min(cfg.eps),
            dim1=-2,
            dim2=-1,
        )
        node_to_edge_weight = H * node_total_flow.unsqueeze(-1)
        HDv = einsum("nkv,nve->nke", Dv, edge_to_node_weight)
        De = torch.diag_embed(
            1.0 / node_to_edge_weight.sum(1).clamp_min(cfg.eps),
            dim1=-2,
            dim2=-1,
        )
        HDe = einsum("nve,nek->nvk", node_to_edge_weight, De)

        e_fts = einsum("nvc,nve->nec", y, HDe)

        y = einsum("nec,nve->nvc", e_fts, HDv)

        y = y + self.bias.unsqueeze(0).unsqueeze(0)
        return self.gelu(y)
