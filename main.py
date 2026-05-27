import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

import config
from engine import train

logging.basicConfig(level=logging.INFO)
cfg = SimpleNamespace(**vars(config))


def set_seed(seed):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(cfg.seed)
    train()
