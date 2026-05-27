import logging
from types import SimpleNamespace

import torch

import config
from models.st_block import FSTHGT
from models.utils import Criterion
from utils import read_preprocess_dataset, slice_dataset, slide_window_b

cfg = SimpleNamespace(**vars(config))
device = torch.device(cfg.device)


def require_config(*names):
    missing = [name for name in names if getattr(cfg, name, None) is None]
    if missing:
        raise ValueError(f"Missing FSTHGT configuration values: {', '.join(missing)}")


def split_dataset(features, embeddings, y, static_h, popu_flow):
    require_config("trainset_ratio", "valset_ratio", "testset_ratio")
    ratio_sum = cfg.trainset_ratio + cfg.valset_ratio + cfg.testset_ratio
    if abs(ratio_sum - 1.0) > cfg.eps:
        raise ValueError("Dataset split ratios must sum to 1.")

    train_len = int(features.shape[0] * cfg.trainset_ratio)
    val_len = int(features.shape[0] * cfg.valset_ratio)
    test_start = train_len + val_len
    train_set = (
        features[:train_len],
        embeddings[:train_len],
        y[:train_len],
        popu_flow[:train_len],
        static_h[:train_len],
    )
    val_set = (
        features[train_len:test_start],
        embeddings[train_len:test_start],
        y[train_len:test_start],
        popu_flow[train_len:test_start],
        static_h[train_len:test_start],
    )
    test_set = (
        features[test_start:],
        embeddings[test_start:],
        y[test_start:],
        popu_flow[test_start:],
        static_h[test_start:],
    )
    return train_set, val_set, test_set


def evaluate_result(output: torch.Tensor, target: torch.Tensor, std, mean):
    output = output * std + mean
    target = target * std + mean
    rmse = torch.sqrt(torch.nn.functional.mse_loss(output, target))
    mae = torch.nn.functional.l1_loss(output, target)
    mape = torch.mean(torch.abs((output - target)) / (target + cfg.eps))
    return (
        mae.cpu().detach().numpy(),
        mape.cpu().detach().numpy(),
        rmse.cpu().detach().numpy(),
    )


def train_one_epoch(dataset, model, optimizer, criterion, std, mean, epoch):
    model.train()

    total_out = []
    total_true = []
    total_loss = 0.0
    total_step = 0
    for src_set, tgt_set in slide_window_b(*dataset):
        batch_data = {**src_set, **tgt_set, "epoch": epoch}
        output, y = model(**batch_data)
        optimizer.zero_grad()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss = loss.item() + total_loss
        total_out.append(output)
        total_true.append(y)
        total_step = total_step + 1

    total_out = torch.cat(total_out, dim=0)
    total_true = torch.cat(total_true, dim=0)
    mae, mape, rmse = evaluate_result(total_out, total_true, std, mean)
    return total_loss / total_step, mae, mape, rmse


def evaluate_one_epoch(dataset, model, criterion, std, mean, epoch):
    model.eval()

    total_out = []
    total_true = []
    total_loss = 0.0
    total_step = 0
    with torch.no_grad():
        for src_set, tgt_set in slide_window_b(*dataset):
            batch_data = {**src_set, **tgt_set, "epoch": epoch}
            output, y = model(**batch_data)

            loss = criterion(output, y)
            total_loss = loss.item() + total_loss
            total_out.append(output)
            total_true.append(y)
            total_step = total_step + 1

    total_out = torch.cat(total_out, dim=0)
    total_true = torch.cat(total_true, dim=0)
    mae, mape, rmse = evaluate_result(total_out, total_true, std, mean)
    return total_loss / total_step, mae, mape, rmse


def train():
    require_config("epochs", "learning_rate", "patience")
    features, embeddings, y, static_h, popu_flow, std, mean = read_preprocess_dataset()
    train_set, val_set, test_set = split_dataset(
        features, embeddings, y, static_h, popu_flow
    )
    train_slice = slice_dataset(*train_set, shuffle=True)
    val_slice = slice_dataset(*val_set)
    test_slice = slice_dataset(*test_set)

    model = FSTHGT()

    model = model.to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay or 0,
    )
    criterion = Criterion()
    best_mae = float("inf")
    best_state = None
    wait = 0
    for epoch in range(cfg.epochs):
        train_loss, train_mae, train_mape, train_rmse = train_one_epoch(
            train_slice, model, optimizer, criterion, std, mean, epoch
        )
        val_loss, val_mae, val_mape, val_rmse = evaluate_one_epoch(
            val_slice, model, criterion, std, mean, epoch
        )
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        logging.info(
            f"(Train) | Epoch={epoch:03d}, loss={train_loss:.4f}, "
            f"train_mae={(train_mae):.2f}, train_mape={(train_mape * 100):.2f}, "
            f"train_rmse={(train_rmse):.2f}"
        )

        logging.info(
            f"(Evaluate) | Epoch={epoch:03d}, loss={val_loss:.4f}, "
            f"val_mae={(val_mae):.2f}, val_mape={(val_mape * 100):.2f}, "
            f"val_rmse={(val_rmse):.2f}"
        )
        if wait >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_mae, test_mape, test_rmse = evaluate_one_epoch(
        test_slice, model, criterion, std, mean, epoch
    )
    logging.info(
        f"(Test) | loss={test_loss:.4f}, test_mae={(test_mae):.2f}, "
        f"test_mape={(test_mape * 100):.2f}, test_rmse={(test_rmse):.2f}"
    )
