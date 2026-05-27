# FSTHGT

FSTHGT is a PyTorch implementation of **Flow-Aware Spatial-Temporal Hypergraph Transformer for Metro Passenger Flow Prediction**.

The code follows the paper-level method components:

- ST-hybrid hypergraph construction with line-based static hyperedges and intersection-centered k-hop static hyperedges.
- Dynamic hypergraph construction from time-step spatial self-attention, using top-k related stations for each center station and always including the center station in its own dynamic hyperedge.
- FAHGConv with node total flow, edge mean flow, node-to-edge flow weighting, and edge-to-node flow weighting.
- Spatial information embedding from random walk encoding and normalized Laplacian encoding.
- Stacked FAHGConv and temporal self-attention layers with an MLP output head.
- MAE loss for training and MAE, MAPE, RMSE for evaluation.

## Structure

- `main.py`: entry point.
- `config.py`: configuration schema. Fill values from your own experiment launcher before training.
- `engine.py`: chronological train/validation/test training and evaluation loop.
- `utils.py`: dataset parsing, preprocessing, spatial embedding, and hypergraph construction.
- `models/`: FSTHGT layers, FAHGConv, output utilities, and loss utilities.

## Configuration

The release package intentionally does not include dataset paths, device IDs, or private machine settings. Set the fields in `config.py` or override them from an external launcher before running training.

Dataset directories are expected to provide the metro `.geo`, `.rel`, `.dyna`, `.pkl`, and line membership `.xlsx` files used by the preprocessing pipeline.

For the paper setting, use chronological split ratios `0.7/0.1/0.2`, `window_size = 4`, `pred_k = 4`, Adam, MAE loss, early stopping, and `d_token = 2 + random_walk_steps + laplacian_k`.

The implementation exposes ablation switches matching the paper variants: `use_line_hyperedge`, `use_khop_hyperedge`, `use_dynamic_hyperedge`, `use_flow_aware`, and `use_spatial_embedding`.

## Minimal Dependencies

- Python 3
- PyTorch
- NumPy
- pandas
- openpyxl

## Static Check

```bash
python -m py_compile FSTHGT_release/*.py FSTHGT_release/models/*.py
python FSTHGT_release/test_fsthgt_contracts.py
```
