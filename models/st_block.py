from types import SimpleNamespace

import torch
from torch import nn

import config
from models.utils import MLPRegressor
from utils import construct_dynamic_hypergraph

from .hgnn_module import FAHGConv

cfg = SimpleNamespace(**vars(config))


class FSTHGTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        temporal_dim = cfg.d_model + cfg.embed_dim
        self.hgnn = FAHGConv(cfg.d_model, cfg.d_model, cfg.dropout)
        self.spatial_proj = nn.Linear(cfg.d_token, cfg.embed_dim)
        self.temporal_query_proj = nn.Linear(temporal_dim, cfg.d_model)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
            kdim=temporal_dim,
            vdim=cfg.d_model,
        )

        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_forward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_forward, cfg.d_model),
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, spatial_embedding, static_h, dynamic_h, popu_flow):
        B, T, N, _ = x.shape
        h = torch.cat(
            [static_h.reshape(B * T, N, -1), dynamic_h],
            dim=-1,
        )
        x_h = self.hgnn(
            x.reshape(B * T, N, -1),
            h,
            popu_flow.reshape(B * T, N, -1),
        ).reshape(B, T, N, -1)
        x = self.norm(x + x_h)

        if cfg.use_spatial_embedding:
            spatial = self.spatial_proj(spatial_embedding)
        else:
            spatial = torch.zeros(
                B, T, N, cfg.embed_dim, device=x.device, dtype=x.dtype
            )

        qk_input = torch.cat([x, spatial], dim=-1)
        qk_input = qk_input.permute(0, 2, 1, 3).reshape(B * N, T, -1)

        v_input = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)

        q_input = self.temporal_query_proj(qk_input)

        temporal_output, _ = self.temporal_attn(
            q_input,
            qk_input,
            v_input,
        )

        temporal_output = temporal_output.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        x = self.norm(x + self.dropout(self.ffn(temporal_output)))
        return x


class FSTHGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(2, cfg.d_model)
        self.dynamic_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dynamic_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.layers = nn.ModuleList([FSTHGTLayer() for _ in range(cfg.num_layers)])
        self.output_mlp = MLPRegressor(
            cfg.d_model, cfg.dim_forward, cfg.pred_k * 2, cfg.dropout
        )

    def _dynamic_hypergraph(self, x):
        B, T, N, D = x.shape
        if not cfg.use_dynamic_hyperedge:
            return torch.zeros(B * T, N, 0, device=x.device, dtype=x.dtype)
        x_t = x.reshape(B * T, N, D)
        q = self.dynamic_q(x_t)
        k = self.dynamic_k(x_t)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        return construct_dynamic_hypergraph(scores)

    def forward(
        self,
        src,
        src_static_h,
        src_embedding,
        src_popu_flow,
        src_y=None,
        tgt_y=None,
        epoch=None,
    ):
        x = self.input_proj(src)
        dynamic_h = self._dynamic_hypergraph(x)

        for layer in self.layers:
            x = layer(x, src_embedding, src_static_h, dynamic_h, src_popu_flow)

        context = x[:, -1]
        output = self.output_mlp(context).reshape(
            src.shape[0], src.shape[2], cfg.pred_k, 2
        )
        output = output.permute(0, 2, 1, 3)
        return output, tgt_y
