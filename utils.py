import os
import pickle
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import config

cfg = SimpleNamespace(**vars(config))
device = torch.device(cfg.device)


def require_config(*names):
    missing = [name for name in names if getattr(cfg, name, None) is None]
    if missing:
        raise ValueError(f"Missing FSTHGT configuration values: {', '.join(missing)}")


def read_line_matrix(line_matrix_dir):
    file = pd.read_excel(line_matrix_dir)

    dict_data = file.set_index(file.columns[0]).to_dict()[file.columns[1]]

    unique_values = sorted(
        set(line for lines in dict_data.values() for line in str(lines).split(","))
    )
    value_to_number = {value: i for i, value in enumerate(unique_values)}

    num_nodes = len(dict_data.keys())
    num_lines = len(unique_values)

    line_matrix = np.zeros((num_nodes, num_lines), dtype=int)

    for i, (node, lines) in enumerate(dict_data.items()):
        line_ids = [value_to_number[line] for line in str(lines).split(",")]
        for line_id in line_ids:
            line_matrix[i, line_id] = 1

    return line_matrix


def extract_day_hour_minute(line: str):
    time_str = line[2]
    time_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    day = time_obj.day
    hour = time_obj.hour
    minute = time_obj.minute
    return day, hour, minute


def read_week_day_pkl(file_dir):
    with open(file_dir, "rb") as f:
        week_day_dict = pickle.load(f)
    return week_day_dict


def read_file(file_dir):
    with open(file_dir, "r") as f:
        lines = f.readlines()[1:]
        datas = []
        for line in lines:
            data = line.strip().split(",")
            datas.append(data)
    return datas


def read_dataset(base_dir):
    listed_files = os.listdir(base_dir)
    for file in listed_files:
        if file.endswith(".geo"):
            geo_file = os.path.join(base_dir, file)
        elif file.endswith(".rel"):
            rel_file = os.path.join(base_dir, file)
        elif file.endswith(".dyna"):
            dyna_file = os.path.join(base_dir, file)
        elif file.endswith(".pkl"):
            week_day_dict = os.path.join(base_dir, file)
        elif file.endswith(".xlsx"):
            line_file = os.path.join(base_dir, file)

    geo_file = read_file(geo_file)
    rel_file = read_file(rel_file)
    dyna_file = read_file(dyna_file)
    week_day_dict = read_week_day_pkl(week_day_dict)
    line_file = read_line_matrix(line_file)
    adj_matrix = np.zeros((len(geo_file), len(geo_file)))

    for rel_line in rel_file:
        start_node = int(rel_line[2])
        end_node = int(rel_line[3])
        adj_matrix[start_node, end_node] = float(rel_line[4])

    if len(dyna_file) % len(geo_file) != 0:
        raise ValueError("The .dyna record count must be divisible by station count.")
    time_len = len(dyna_file) // len(geo_file)
    timestamp_matrix = np.zeros((time_len, len(geo_file), 2))
    time_encoder_matrix = np.zeros((time_len, len(geo_file), 3))
    for t in range(time_len):
        day, hour, minute = extract_day_hour_minute(dyna_file[t])
        weekday = week_day_dict[day]

        for n in range(len(geo_file)):
            timestamp_matrix[t, n, 0] = float(dyna_file[n * time_len + t][4])
            timestamp_matrix[t, n, 1] = float(dyna_file[n * time_len + t][5])
            time_encoder_matrix[t, n, 0] = weekday
            time_encoder_matrix[t, n, 1] = hour
            time_encoder_matrix[t, n, 2] = minute
    return adj_matrix, timestamp_matrix, time_encoder_matrix, line_file


def identify_intersections(node_line_matrix: np.array):
    return np.where(np.sum(node_line_matrix, axis=-1) > 1)[0]


def construct_static_hypergraph(node_line_matrix: np.array, adj_matrix: np.array):
    require_config("khop")
    hyperedges = []
    if cfg.use_line_hyperedge:
        hyperedges.append(node_line_matrix.astype(float))

    intersections = identify_intersections(node_line_matrix)
    if cfg.use_khop_hyperedge and len(intersections) > 0:
        binary_adj = (np.asarray(adj_matrix) > 0).astype(float)
        reachability = np.linalg.matrix_power(binary_adj, cfg.khop)
        intersection_h = np.zeros((node_line_matrix.shape[0], len(intersections)))
        for j, node in enumerate(intersections):
            indices = np.where(reachability[node] != 0)[0]
            intersection_h[indices, j] = 1.0
            intersection_h[node, j] = 1.0
        hyperedges.append(intersection_h)

    if not hyperedges:
        return torch.zeros((node_line_matrix.shape[0], 0), dtype=torch.float32)

    static_h = np.concatenate(hyperedges, axis=-1)
    return torch.from_numpy(static_h).to(torch.float32)


def construct_dynamic_hypergraph(attention_matrix: torch.Tensor):
    """
    Args:
        attention_matrix: (batch, nodes, nodes) spatial self-attention matrix.
    Returns:
        dynamic_h: (batch, nodes, nodes) dynamic incidence matrix where each
        centered hyperedge contains its center station and top-k related nodes.
    """
    batch_count, N, _ = attention_matrix.shape
    K = N
    require_config("dynamic_top_k")
    top_k = min(cfg.dynamic_top_k, max(N - 1, 1))
    center = torch.arange(N, device=attention_matrix.device)
    attention_matrix = attention_matrix.clone()
    attention_matrix[:, center, center] = float("-inf")
    _, topn_indices = torch.topk(attention_matrix, k=top_k, dim=-1)

    dynamic_h = torch.zeros(batch_count, N, N, device=attention_matrix.device)
    batch_idx = (
        torch.arange(batch_count, device=attention_matrix.device)
        .view(-1, 1, 1)
        .expand(batch_count, K, top_k)
        .flatten()
    )

    hyperedge_idx = (
        torch.arange(K, device=attention_matrix.device)
        .view(1, -1, 1)
        .expand(batch_count, K, top_k)
        .flatten()
    )

    node_idx = topn_indices.flatten()

    dynamic_h[batch_idx, node_idx, hyperedge_idx] = 1.0
    dynamic_h[:, center, center] = 1.0
    return dynamic_h


def construct_hypergraph(static_h, dynamic_h):
    static_h = static_h.unsqueeze(0).expand(dynamic_h.shape[0], -1, -1)
    return torch.cat([static_h, dynamic_h], dim=-1)


def calculate_node_population(popu_flow):
    return torch.sum(popu_flow, dim=-1)


def calculate_edge_population(popu_flow: torch.Tensor, hypergraph: torch.Tensor):
    incidence = hypergraph.unsqueeze(-1)
    edge_flow_sum = torch.sum(popu_flow.unsqueeze(2) * incidence, dim=1)
    edge_size = incidence.sum(dim=1).clamp_min(1.0)
    return edge_flow_sum / edge_size


def random_walk_encoding(adj_matrix, steps):
    adjacency = np.asarray(adj_matrix, dtype=float)
    degree = adjacency.sum(axis=0)
    inv_degree = np.zeros_like(degree)
    inv_degree[degree > 0] = 1.0 / degree[degree > 0]
    random_walk = adjacency @ np.diag(inv_degree)
    powers = []
    current = np.eye(adjacency.shape[0])
    for _ in range(steps):
        powers.append(np.diag(current))
        current = current @ random_walk
    return torch.from_numpy(np.stack(powers, axis=-1)).to(torch.float32)


def laplacian_encoding(adj_matrix, k):
    adjacency = np.asarray(adj_matrix, dtype=float)
    degree = adjacency.sum(axis=-1)
    inv_sqrt_degree = np.zeros_like(degree)
    inv_sqrt_degree[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    normalized_adj = (
        np.diag(inv_sqrt_degree) @ adjacency @ np.diag(inv_sqrt_degree)
    )
    laplacian = np.eye(adjacency.shape[0]) - normalized_adj
    _, eigenvectors = np.linalg.eigh(laplacian)
    start = 1 if eigenvectors.shape[1] > 1 else 0
    end = min(start + k, eigenvectors.shape[1])
    embedding = eigenvectors[:, start:end]
    if embedding.shape[1] < k:
        padding = np.zeros((embedding.shape[0], k - embedding.shape[1]))
        embedding = np.concatenate([embedding, padding], axis=-1)
    return torch.from_numpy(embedding).to(torch.float32)


def time_encoder(time_encoder_matrix):
    require_config("time_step_in_day")
    day = time_encoder_matrix[:, :, 0]
    time_embedding = np.zeros(
        (time_encoder_matrix.shape[0], time_encoder_matrix.shape[1], 2)
    )

    time_embedding[:, :, 1] = day / 6

    time_step_in_day = 0
    for i in range(time_encoder_matrix.shape[0]):
        time_embedding[i, :, 0] = time_step_in_day
        time_step_in_day = time_step_in_day + 1
        if time_step_in_day == cfg.time_step_in_day:
            time_step_in_day = 0
    time_embedding[..., 0] = time_embedding[..., 0] / (cfg.time_step_in_day - 1)
    return torch.from_numpy(time_embedding)


def slide_window(features, embeddings, y, popu_flow, static_h):
    """
    Args:
        features: traffic features with shape (time, node, feature).
        embeddings: spatial-temporal embeddings.
        y: normalized prediction target.
        popu_flow: node total flow sequence.
        static_h: static hypergraph incidence matrix.
    """
    require_config("window_size", "pred_k")
    window_size = cfg.window_size
    pred_k = cfg.pred_k
    time_steps = features.shape[0]

    for t in range(time_steps - window_size - pred_k):
        start_src_step = t
        end_src_step = t + window_size
        start_tgt_step = t + window_size
        end_tgt_step = t + window_size + pred_k
        src_set = {
            "src": features[start_src_step:end_src_step].to(device),
            "src_static_h": static_h[start_src_step:end_src_step].to(device),
            "src_embedding": embeddings[start_src_step:end_src_step].to(device),
            "src_popu_flow": popu_flow[start_src_step:end_src_step].to(device),
            "src_y": y[start_src_step:end_src_step].to(device),
        }
        tgt_set = {
            "tgt_y": y[start_tgt_step:end_tgt_step].to(device),
        }
        yield t, src_set, tgt_set


def read_preprocess_dataset():
    require_config(
        "dataset",
        "random_walk_steps",
        "laplacian_k",
        "time_step_in_day",
    )
    dataset_name = cfg.dataset
    adj_matrix, features, time_encoder_matrix, line_matrix = read_dataset(
        dataset_name
    )
    y_popu = torch.from_numpy(np.sum(features, axis=-1))
    time_embedding = time_encoder(time_encoder_matrix)
    static_h = construct_static_hypergraph(line_matrix, adj_matrix)
    static_h = static_h.expand(features.shape[0], -1, -1)
    random_walk_embedding = random_walk_encoding(adj_matrix, cfg.random_walk_steps)
    laplacian_embedding = laplacian_encoding(adj_matrix, cfg.laplacian_k)
    spatial_embedding = torch.cat([random_walk_embedding, laplacian_embedding], dim=-1)
    spatial_embedding = spatial_embedding.expand(features.shape[0], -1, -1)
    embeddings = torch.cat([time_embedding, spatial_embedding], dim=-1)

    y, mean, std = scaler(features)
    features = (features - mean) / (std + cfg.eps)
    y = torch.from_numpy(features)

    return (
        torch.from_numpy(features).to(torch.float32),
        embeddings.to(torch.float32),
        y.to(torch.float32),
        static_h.to(torch.float32),
        y_popu.to(torch.float32),
        std,
        mean,
    )


def scaler(y):
    mean = y.mean()
    std = y.std()
    return (y - mean) / (std + cfg.eps), mean, std


def minmax_scale(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def slide_window_b(features, embeddings, y, popu_flow, static_h):
    require_config("window_size", "pred_k", "batch")
    window_size = cfg.window_size
    pred_k = cfg.pred_k

    batch = cfg.batch
    batch_src = {
        key: []
        for key in ["src", "src_static_h", "src_embedding", "src_popu_flow", "src_y"]
    }
    batch_tgt = {"tgt_y": []}
    for t in range(features.shape[0]):
        src_features = features[t][:window_size].to(device)
        src_emb = embeddings[t][:window_size].to(device)
        src_popu = popu_flow[t][:window_size].to(device)
        src_y_vals = y[t][:window_size].to(device)
        src_static_h = static_h[t][:window_size].to(device)
        tgt_y_vals = y[t][window_size : window_size + pred_k].to(device)

        batch_src["src"].append(src_features)
        batch_src["src_static_h"].append(src_static_h)
        batch_src["src_embedding"].append(src_emb)
        batch_src["src_popu_flow"].append(src_popu)
        batch_src["src_y"].append(src_y_vals + cfg.eps)

        batch_tgt["tgt_y"].append(tgt_y_vals + cfg.eps)
        if len(batch_src["src"]) == batch:
            batched_src = {k: torch.stack(v, dim=0) for k, v in batch_src.items()}
            batched_tgt = {k: torch.stack(v, dim=0) for k, v in batch_tgt.items()}
            yield batched_src, batched_tgt

            batch_src = {key: [] for key in batch_src.keys()}
            batch_tgt = {key: [] for key in batch_tgt.keys()}
    if len(batch_src["src"]) > 0:
        batched_src = {k: torch.stack(v, dim=0) for k, v in batch_src.items()}
        batched_tgt = {k: torch.stack(v, dim=0) for k, v in batch_tgt.items()}
        yield batched_src, batched_tgt


def slice_dataset(features, embeddings, y, popu_flow, static_h, shuffle=False):
    require_config("window_size", "pred_k")
    window_size = cfg.window_size
    pred_k = cfg.pred_k

    features_list = []
    embeddings_list = []
    y_list = []
    popu_flow_list = []
    static_h_list = []

    for i in range(0, features.shape[0] - window_size - pred_k, 1):
        features_list.append(features[i : i + window_size + pred_k])
        embeddings_list.append(embeddings[i : i + window_size + pred_k])
        y_list.append(y[i : i + window_size + pred_k])
        popu_flow_list.append(popu_flow[i : i + window_size + pred_k])
        static_h_list.append(static_h[i : i + window_size + pred_k])

    features_array = torch.stack(features_list, dim=0)
    embeddings_array = torch.stack(embeddings_list, dim=0)
    y_array = torch.stack(y_list, dim=0)
    popu_flow_array = torch.stack(popu_flow_list, dim=0)
    static_h_array = torch.stack(static_h_list, dim=0)
    indices = torch.randperm(len(features_list)) if shuffle else torch.arange(len(features_list))

    return (
        features_array[indices],
        embeddings_array[indices],
        y_array[indices],
        popu_flow_array[indices],
        static_h_array[indices],
    )
